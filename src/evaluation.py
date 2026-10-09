"""Evaluation: measure retrieval and generation against a golden set.

The golden set holds, for each question, the passages that contain the answer and
the facts a correct answer must state. Retrieval is measured without the language
model; generation needs one call per question, and one more when faithfulness is judged.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from src.llm import ChatClient
from src.retrieval import Hit

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
    passages = "\n\n".join(f"[{number}] {hit.chunk.text}" for number, hit in enumerate(hits, start=1))
    reply = llm.ask(f"Passages:\n\n{passages}\n\nAnswer to check:\n\n{answer}", system=_JUDGE_SYSTEM)
    found = re.search(r"\{[^{}]*\}", reply.text)
    if found is None:
        raise ValueError(f"The judge did not reply with JSON: {reply.text[:200]!r}")
    verdict = json.loads(found.group(0))
    claims, supported = int(verdict["claims"]), int(verdict["supported"])
    if not 0 <= supported <= claims:
        raise ValueError(f"The judge returned an impossible verdict: {verdict!r}")
    return Faithfulness(claims, supported)
