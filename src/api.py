"""Backend: the HTTP API that the web interface calls."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from statistics import fmean

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.config import RootConfig, load_config, project_path
from src.evaluation import (
    GoldenQuestion,
    RetrievalResult,
    check_answer,
    judge_faithfulness,
    load_golden_set,
    retrieval_result,
)
from src.llm import LlmError
from src.rag import RagAnswer, RagPipeline, Turn, build_pipeline
from src.retrieval import Hit
from src.vector_store import EmptyCollectionError


class PastTurn(BaseModel):
    """One earlier exchange of the conversation, kept by the web interface."""

    question: str
    answer: str


class Question(BaseModel):
    """Body of a request to /api/ask: the new question and the conversation before it."""

    question: str
    history: list[PastTurn] = Field(default_factory=list)


class Source(BaseModel):
    """One passage that was given to the model."""

    position: int
    score: float
    document: str
    section: str
    page: int | None
    text: str


class Answer(BaseModel):
    """Body of a reply from /api/ask."""

    answer: str
    model: str
    sources: list[Source]


class Health(BaseModel):
    """Body of a reply from /api/health."""

    status: str
    model: str
    chunks: int
    max_question_chars: int


class GoldenPassage(BaseModel):
    """A passage of the golden set: where the answer to a question is written."""

    document: str
    section: str


class GoldenEntry(BaseModel):
    """A question of the golden set."""

    id: str
    type: str
    question: str
    answer: str
    facts: list[list[str]]
    context: list[GoldenPassage]


class GoldenSet(BaseModel):
    """Body of a reply from /api/evaluation/golden."""

    top_k: int
    questions: list[GoldenEntry]


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
    """Body of a reply from /api/evaluation/retrieval."""

    top_k: int
    questions: int
    summary: RetrievalScores
    by_type: list[RetrievalGroup]
    rows: list[RetrievalRow]


class GenerationRequest(BaseModel):
    """Body of a request to /api/evaluation/generation: one question, and whether to judge faithfulness."""

    id: str
    judge: bool = False


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


def _sources(hits: Sequence[Hit]) -> list[Source]:
    """The retrieved chunks in the form sent to the web interface."""
    return [
        Source(
            position=position,
            score=hit.score,
            document=Path(hit.chunk.source).name,
            section=hit.chunk.section,
            page=hit.chunk.page,
            text=hit.chunk.text,
        )
        for position, hit in enumerate(hits, start=1)
    ]


def _average(results: Sequence[RetrievalResult]) -> dict[str, float]:
    """Average of each retrieval metric over a group of questions."""
    return {
        "precision": fmean(result.precision for result in results),
        "recall": fmean(result.recall for result in results),
        "mrr": fmean(result.reciprocal_rank for result in results),
        "ndcg": fmean(result.ndcg for result in results),
        "context_recall": fmean(result.context_recall for result in results),
    }


def _retrieval_report(
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
    return RetrievalReport(top_k=top_k, questions=len(rows), summary=summary, by_type=by_type, rows=rows)


def _generation_row(
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


def build_app(
    config: RootConfig, pipeline_factory: Callable[[], RagPipeline], golden_set: Sequence[GoldenQuestion]
) -> FastAPI:
    """Create the application; the pipeline is built once, when the server starts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline_factory()
        yield

    app = FastAPI(title="RAG workshop API", lifespan=lifespan)
    golden = {question.id: question for question in golden_set}

    def answer_with(pipeline: RagPipeline, question: str, history: Sequence[Turn]) -> RagAnswer:
        """Run the pipeline, turning its known failures into HTTP errors."""
        try:
            return pipeline.answer(question, history)
        except EmptyCollectionError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except LlmError as error:
            detail = f"The language model did not answer. {error}"
            raise HTTPException(status_code=502, detail=detail) from error

    @app.get("/api/health")
    def health(request: Request) -> Health:
        pipeline: RagPipeline = request.app.state.pipeline
        return Health(
            status="ok",
            model=config.llm.model,
            chunks=pipeline.indexed_chunks(),
            max_question_chars=config.api.max_question_chars,
        )

    @app.post("/api/ask")
    def ask(body: Question, request: Request) -> Answer:
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="The question is empty.")
        if len(question) > config.api.max_question_chars:
            limit = config.api.max_question_chars
            raise HTTPException(status_code=422, detail=f"The question is longer than {limit} characters.")
        history = [Turn(turn.question, turn.answer) for turn in body.history]
        result = answer_with(request.app.state.pipeline, question, history)
        return Answer(answer=result.text, model=result.model, sources=_sources(result.hits))

    @app.get("/api/evaluation/golden")
    def golden_questions() -> GoldenSet:
        entries = [
            GoldenEntry(
                id=question.id,
                type=question.type,
                question=question.question,
                answer=question.answer,
                facts=[list(fact) for fact in question.facts],
                context=[
                    GoldenPassage(document=item.document, section=item.section) for item in question.context
                ],
            )
            for question in golden_set
        ]
        return GoldenSet(top_k=config.reranking.top_k, questions=entries)

    @app.post("/api/evaluation/retrieval")
    def evaluate_retrieval(request: Request) -> RetrievalReport:
        try:
            return _retrieval_report(request.app.state.pipeline, golden_set, config.reranking.top_k)
        except EmptyCollectionError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/api/evaluation/generation")
    def evaluate_generation(body: GenerationRequest, request: Request) -> GenerationRow:
        question = golden.get(body.id)
        if question is None:
            raise HTTPException(status_code=404, detail=f"The golden set has no question {body.id!r}.")
        pipeline: RagPipeline = request.app.state.pipeline
        result = answer_with(pipeline, question.question, ())
        return _generation_row(pipeline, question, result, body.judge, config.evaluation.abstention_markers)

    return app


def create_app() -> FastAPI:
    """Application factory for the server: ``uvicorn src.api:create_app --factory``."""
    config = load_config()
    golden_set = load_golden_set(project_path(config.paths.golden_set))
    return build_app(config, lambda: build_pipeline(config), golden_set)
