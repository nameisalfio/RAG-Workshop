"""Evaluation: measure retrieval and generation against a golden set.

The golden set holds, for each question, the passages that contain the answer and
the facts a correct answer must state. Retrieval is measured without the language
model; generation needs one call per question, and one more when faithfulness is judged.

``python -m src.evaluation`` asks every question and records the results in a file,
so that they can be shown without waiting for the model.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from src.config import CONFIG_NAME, EvaluationConfig, RootConfig, project_path, register_schema
from src.llm import AllowanceExhaustedError, ChatClient, LlmError
from src.rag import RagAnswer, RagPipeline, build_pipeline, format_passages
from src.retrieval.vector_store import Hit

LOGGER = logging.getLogger(__name__)

UNANSWERABLE = "unanswerable"

_JUDGE_SYSTEM = (
    "You check whether an answer is supported by the passages it was written from. "
    "Split the answer into its factual claims. A claim is supported only when the passages state it. "
    'Reply with JSON only, in this form: {"claims": <number of claims>, "supported": <number supported>}.'
)


@dataclass(frozen=True)
class ContextItem:
    """A passage that holds the answer: the start of a file name and, optionally, of a section heading."""

    document: str
    section: str

    def matches(self, hit: Hit) -> bool:
        """True when a retrieved chunk is this passage."""
        return Path(hit.chunk.source).name.startswith(self.document) and hit.chunk.section.startswith(
            self.section
        )


@dataclass(frozen=True)
class GoldenQuestion:
    """One entry of the golden set; ``facts`` holds, for each fact, its accepted wordings."""

    id: str
    type: str
    question: str
    answer: str
    facts: tuple[tuple[str, ...], ...]
    context: tuple[ContextItem, ...]

    @property
    def answerable(self) -> bool:
        """False when the documents do not hold the answer and the system is expected to say so."""
        return self.type != UNANSWERABLE


@dataclass(frozen=True)
class RetrievalResult:
    """How well the retrieved chunks cover one question; every score is between 0 and 1."""

    question_id: str
    precision: float
    recall: float
    reciprocal_rank: float
    ndcg: float
    context_recall: float
    relevant: tuple[bool, ...]


@dataclass(frozen=True)
class AnswerCheck:
    """An answer compared with the golden set."""

    question_id: str
    facts_found: tuple[bool, ...]
    abstained: bool
    passed: bool


@dataclass(frozen=True)
class Faithfulness:
    """Verdict of the judge: how many claims of an answer the passages support."""

    claims: int
    supported: int

    @property
    def score(self) -> float:
        """Share of supported claims; an answer without claims is fully faithful."""
        return self.supported / self.claims if self.claims else 1.0


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    """Read the golden set; an entry that is incomplete or inconsistent is an error."""
    entries = OmegaConf.to_container(OmegaConf.load(path))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: the golden set must be a non-empty list")
    questions: list[GoldenQuestion] = []
    for entry in entries:
        question = GoldenQuestion(
            id=str(entry["id"]),
            type=str(entry["type"]),
            question=str(entry["question"]).strip(),
            answer=str(entry["answer"]).strip(),
            facts=tuple(tuple(str(wording) for wording in fact) for fact in entry["facts"]),
            context=tuple(
                ContextItem(str(item["document"]), str(item.get("section", ""))) for item in entry["context"]
            ),
        )
        if not question.question or not question.answer:
            raise ValueError(f"{path}: entry {question.id!r} needs a question and an answer")
        if any(not fact for fact in question.facts):
            raise ValueError(f"{path}: entry {question.id!r} has a fact without wordings")
        if question.answerable != bool(question.facts and question.context):
            raise ValueError(
                f"{path}: entry {question.id!r} must have facts and context unless it is {UNANSWERABLE}"
            )
        questions.append(question)
    identifiers = [question.id for question in questions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path}: two entries share the same id")
    return questions


def _normalize(text: str) -> str:
    """Lower case with single spaces, so that wordings match across line breaks."""
    return " ".join(text.lower().split())


def facts_found(question: GoldenQuestion, text: str) -> tuple[bool, ...]:
    """For each fact of a question, whether one of its wordings occurs in a text."""
    normalized = _normalize(text)
    return tuple(any(_normalize(wording) in normalized for wording in fact) for fact in question.facts)


def retrieval_result(question: GoldenQuestion, hits: Sequence[Hit]) -> RetrievalResult:
    """Score the chunks retrieved for an answerable question against its golden passages.

    Precision and recall follow the usual definitions over the retrieved chunks and
    the golden passages; the reciprocal rank is one over the position of the first
    relevant chunk; NDCG counts each golden passage once, the first time it appears.
    Context recall is the share of the facts of the correct answer that the
    retrieved text contains.
    """
    if not question.answerable:
        raise ValueError(f"Question {question.id!r} is unanswerable: retrieval cannot be scored")
    if not hits:
        raise ValueError(f"No chunks were retrieved for question {question.id!r}")
    relevant = tuple(any(item.matches(hit) for item in question.context) for hit in hits)
    found = [any(item.matches(hit) for hit in hits) for item in question.context]
    first = next((position for position, flag in enumerate(relevant, start=1) if flag), None)
    seen: set[ContextItem] = set()
    gain = 0.0
    for position, hit in enumerate(hits, start=1):
        new = [item for item in question.context if item not in seen and item.matches(hit)]
        if new:
            seen.update(new)
            gain += 1.0 / math.log2(position + 1)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(hits), len(found)) + 1))
    facts = facts_found(question, "\n".join(hit.chunk.text for hit in hits))
    return RetrievalResult(
        question_id=question.id,
        precision=sum(relevant) / len(hits),
        recall=sum(found) / len(found),
        reciprocal_rank=1.0 / first if first else 0.0,
        ndcg=gain / ideal,
        context_recall=sum(facts) / len(facts),
        relevant=relevant,
    )


def check_answer(question: GoldenQuestion, answer: str, abstention_markers: Sequence[str]) -> AnswerCheck:
    """Compare an answer with the golden set.

    An answerable question passes when the answer states every fact. An unanswerable
    one passes when the answer says that the documents do not help, which is
    recognised by the configured wordings.
    """
    found = facts_found(question, answer)
    normalized = _normalize(answer)
    abstained = any(_normalize(marker) in normalized for marker in abstention_markers)
    passed = all(found) if question.answerable else abstained
    return AnswerCheck(question.id, found, abstained, passed)


def judge_faithfulness(llm: ChatClient, answer: str, hits: Sequence[Hit]) -> Faithfulness:
    """Ask the language model how many claims of an answer are supported by the passages."""
    # The judge reads the passages as the answering model did: an answer that names a document
    # or a section can be checked only when the passages carry those names.
    reply = llm.ask(f"{format_passages(hits)}\n\nAnswer to check:\n\n{answer}", system=_JUDGE_SYSTEM)
    found = re.search(r"\{[^{}]*\}", reply.text)
    if found is None:
        raise ValueError(f"The judge did not reply with JSON: {reply.text[:200]!r}")
    verdict = json.loads(found.group(0))
    claims, supported = int(verdict["claims"]), int(verdict["supported"])
    if not 0 <= supported <= claims:
        raise ValueError(f"The judge returned an impossible verdict: {verdict!r}")
    return Faithfulness(claims, supported)


# ------------------------------------------------------------------- reports


class RetrievedPassage(BaseModel):
    """A retrieved chunk and whether the golden set lists it for the question."""

    position: int
    document: str
    section: str
    relevant: bool


class RetrievalScores(BaseModel):
    """Retrieval metrics, each between 0 and 1."""

    precision: float
    recall: float
    mrr: float
    ndcg: float
    context_recall: float


class RetrievalRow(RetrievalScores):
    """Retrieval metrics of one question, with what was retrieved."""

    id: str
    first_relevant: int | None
    retrieved: list[RetrievedPassage]


class RetrievalGroup(RetrievalScores):
    """Average retrieval metrics of the questions of one type."""

    type: str
    questions: int


class RetrievalReport(BaseModel):
    """Body of a reply from /api/evaluation/retrieval.

    ``search`` says how the passages were found; a file recorded before the field existed has none.
    """

    search: str = "a search that was not recorded"
    top_k: int
    questions: int
    summary: RetrievalScores
    by_type: list[RetrievalGroup]
    rows: list[RetrievalRow]


class FactCheck(BaseModel):
    """A fact of the correct answer and whether the answer of the model states it."""

    wordings: list[str]
    found: bool


class FaithfulnessVerdict(BaseModel):
    """How many claims of the answer the retrieved passages support, according to the judge."""

    claims: int
    supported: int
    score: float


class GenerationRow(BaseModel):
    """Body of a reply from /api/evaluation/generation."""

    id: str
    answer: str
    model: str
    facts: list[FactCheck]
    abstained: bool
    passed: bool
    faithfulness: FaithfulnessVerdict | None
    faithfulness_error: str | None


def _average(results: Sequence[RetrievalResult]) -> dict[str, float]:
    """Average of each retrieval metric over a group of questions."""
    return {
        "precision": fmean(result.precision for result in results),
        "recall": fmean(result.recall for result in results),
        "mrr": fmean(result.reciprocal_rank for result in results),
        "ndcg": fmean(result.ndcg for result in results),
        "context_recall": fmean(result.context_recall for result in results),
    }


def retrieval_report(
    pipeline: RagPipeline, golden_set: Sequence[GoldenQuestion], top_k: int
) -> RetrievalReport:
    """Measure retrieval on every answerable question; the language model is not called."""
    results: dict[str, list[RetrievalResult]] = defaultdict(list)
    rows: list[RetrievalRow] = []
    for question in golden_set:
        if not question.answerable:
            continue
        hits = pipeline.retrieve(question.question)
        result = retrieval_result(question, hits)
        results[question.type].append(result)
        passages = [
            RetrievedPassage(
                position=position,
                document=Path(hit.chunk.source).name,
                section=hit.chunk.section,
                relevant=flag,
            )
            for position, (hit, flag) in enumerate(zip(hits, result.relevant, strict=True), start=1)
        ]
        rows.append(
            RetrievalRow(
                id=question.id,
                precision=result.precision,
                recall=result.recall,
                mrr=result.reciprocal_rank,
                ndcg=result.ndcg,
                context_recall=result.context_recall,
                first_relevant=next((passage.position for passage in passages if passage.relevant), None),
                retrieved=passages,
            )
        )
    if not rows:
        raise ValueError("The golden set has no answerable question")
    by_type = [
        RetrievalGroup(type=kind, questions=len(group), **_average(group)) for kind, group in results.items()
    ]
    everything = [result for group in results.values() for result in group]
    summary = RetrievalScores(**_average(everything))
    return RetrievalReport(
        search=pipeline.search, top_k=top_k, questions=len(rows), summary=summary, by_type=by_type, rows=rows
    )


def generation_row(
    pipeline: RagPipeline, question: GoldenQuestion, result: RagAnswer, judge: bool, markers: Sequence[str]
) -> GenerationRow:
    """Compare an answer with the golden set and, when asked, have its faithfulness judged."""
    check = check_answer(question, result.text, markers)
    verdict: FaithfulnessVerdict | None = None
    judge_error: str | None = None
    if judge:
        # A judge that fails does not hide the answer: the row reports why it has no verdict.
        try:
            judged = judge_faithfulness(pipeline.llm, result.text, result.hits)
            verdict = FaithfulnessVerdict(
                claims=judged.claims, supported=judged.supported, score=judged.score
            )
        except AllowanceExhaustedError:
            # No request can succeed until the reset: the caller has to know, not just this row.
            raise
        except (LlmError, ValueError, KeyError) as error:
            judge_error = str(error)
    facts = [
        FactCheck(wordings=list(fact), found=found)
        for fact, found in zip(question.facts, check.facts_found, strict=True)
    ]
    return GenerationRow(
        id=question.id,
        answer=result.text,
        model=result.model,
        facts=facts,
        abstained=check.abstained,
        passed=check.passed,
        faithfulness=verdict,
        faithfulness_error=judge_error,
    )


class EvaluationResults(BaseModel):
    """A recorded run over the whole golden set; ``errors`` holds the questions that got no answer."""

    created_at: str
    judged: bool
    retrieval: RetrievalReport
    generation: dict[str, GenerationRow]
    errors: dict[str, str]


def save_results(results: EvaluationResults, path: Path) -> None:
    """Write a recorded run as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results.model_dump_json(indent=2), encoding="utf-8")


def load_results(path: Path, golden_set: Sequence[GoldenQuestion], search: str) -> EvaluationResults:
    """Read a recorded run; one made with another golden set or another search is an error."""
    results = EvaluationResults.model_validate_json(path.read_text(encoding="utf-8"))
    if results.retrieval.search != search:
        raise ValueError(
            f"{path} was recorded with {results.retrieval.search}, the system now uses {search}: "
            "run python -m src.evaluation again"
        )
    asked = set(results.generation) | set(results.errors)
    measured = {row.id for row in results.retrieval.rows}
    if asked != {question.id for question in golden_set} or measured != {
        question.id for question in golden_set if question.answerable
    }:
        raise ValueError(f"{path} was recorded with another golden set: run python -m src.evaluation again")
    return results


# ---------------------------------------------------------------- recorded run


class Checkpoint(BaseModel):
    """Answers recorded by a run that has not finished yet."""

    search: str
    judged: bool
    generation: dict[str, GenerationRow]


def _load_checkpoint(
    path: Path, golden_set: Sequence[GoldenQuestion], judged: bool, search: str
) -> dict[str, GenerationRow]:
    """Read the answers of an interrupted run; one made with other settings is an error."""
    checkpoint = Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))
    known = {question.id for question in golden_set}
    same_settings = checkpoint.judged == judged and checkpoint.search == search
    if not same_settings or not set(checkpoint.generation) <= known:
        raise ValueError(
            f"{path} comes from a run with another golden set, search or judge setting: "
            "run again with evaluation.fresh=true"
        )
    return dict(checkpoint.generation)


def record_answers(
    pipeline: RagPipeline,
    golden_set: Sequence[GoldenQuestion],
    settings: EvaluationConfig,
    checkpoint: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, GenerationRow], dict[str, str]]:
    """Ask every question that has no recorded answer yet; return the answers and the failures.

    Each answer is written to the checkpoint as soon as it arrives. When the allowance
    of requests is used up the run stops with that error: what was answered stays in
    the checkpoint and the next run continues from there.
    """
    if settings.attempts <= 0:
        raise ValueError(f"evaluation.attempts must be positive, got {settings.attempts}")
    generation: dict[str, GenerationRow] = {}
    if checkpoint.is_file() and not settings.fresh:
        generation = _load_checkpoint(checkpoint, golden_set, settings.judge, pipeline.search)
        LOGGER.info("Continuing an earlier run: %d answers are already recorded", len(generation))
    errors: dict[str, str] = {}
    for number, question in enumerate(golden_set, start=1):
        if question.id in generation:
            continue
        for attempt in range(1, settings.attempts + 1):
            try:
                result = pipeline.answer(question.question, fresh=settings.fresh)
                row = generation_row(pipeline, question, result, settings.judge, settings.abstention_markers)
            except AllowanceExhaustedError:
                LOGGER.error(
                    "%d of %d questions are recorded. Run the command again after the reset to continue.",
                    len(generation),
                    len(golden_set),
                )
                raise
            except LlmError as error:
                LOGGER.warning("%s, attempt %d of %d: %s", question.id, attempt, settings.attempts, error)
                errors[question.id] = str(error)
                if attempt < settings.attempts:
                    sleep(settings.retry_pause_s)
                continue
            errors.pop(question.id, None)
            generation[question.id] = row
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            saved = Checkpoint(search=pipeline.search, judged=settings.judge, generation=generation)
            checkpoint.write_text(saved.model_dump_json(indent=2), encoding="utf-8")
            outcome = "pass" if row.passed else "fail"
            LOGGER.info("%d of %d, %s: %s", number, len(golden_set), question.id, outcome)
            break
    return generation, errors


@hydra.main(version_base=None, config_path="conf", config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """Ask every question of the golden set and record the results.

    Run it with ``python -m src.evaluation``. A run that stops early is continued
    by the next one; ``evaluation.fresh=true`` starts again from the first question.
    """
    config = cast(RootConfig, OmegaConf.to_object(cfg))
    golden_set = load_golden_set(project_path(config.paths.golden_set))
    target = project_path(config.paths.evaluation_results)
    checkpoint = target.with_suffix(".partial.json")
    pipeline = build_pipeline(config)
    retrieval = retrieval_report(pipeline, golden_set, config.reranking.top_k)
    LOGGER.info("Retrieval measured on %d questions: %s", retrieval.questions, retrieval.summary)
    generation, errors = record_answers(pipeline, golden_set, config.evaluation, checkpoint)
    results = EvaluationResults(
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        judged=config.evaluation.judge,
        retrieval=retrieval,
        generation=generation,
        errors=errors,
    )
    save_results(results, target)
    LOGGER.info("Results of %d questions written to %s", len(generation), target)
    if errors:
        # The checkpoint stays, so that the next run asks only the questions still without an answer.
        LOGGER.warning("No answer for %s: run the command again to ask only these", sorted(errors))
    else:
        checkpoint.unlink(missing_ok=True)
    unjudged = sorted(name for name, row in generation.items() if row.faithfulness_error)
    if unjudged:
        LOGGER.warning("The judge gave no verdict for %s", unjudged)


if __name__ == "__main__":
    register_schema()
    main()
