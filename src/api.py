"""Backend: the HTTP API that the web interface calls."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.config import RootConfig, load_config, project_path
from src.evaluation import (
    EvaluationResults,
    GenerationRow,
    GoldenQuestion,
    RetrievalReport,
    generation_row,
    load_golden_set,
    load_results,
    retrieval_report,
)
from src.llm import AllowanceExhaustedError, LlmError
from src.rag import RagAnswer, RagPipeline, Turn, build_pipeline
from src.retrieval.vector_store import EmptyCollectionError, Hit


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
    # True when the answer was saved by an earlier identical question and no model was called.
    cached: bool


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


class GenerationRequest(BaseModel):
    """Body of a request to /api/evaluation/generation: one question, and whether to judge faithfulness."""

    id: str
    judge: bool = False


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


def build_app(
    config: RootConfig,
    pipeline_factory: Callable[[], RagPipeline],
    golden_set: Sequence[GoldenQuestion],
    results_path: Path,
) -> FastAPI:
    """Create the application; the pipeline is built once, when the server starts.

    ``results_path`` is the file with the recorded evaluation results; it may not exist yet.
    """

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
        except AllowanceExhaustedError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
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
        return Answer(
            answer=result.text, model=result.model, sources=_sources(result.hits), cached=result.cached
        )

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

    @app.get("/api/evaluation/results")
    def recorded_results(request: Request) -> EvaluationResults:
        """The results recorded by ``python -m src.evaluation``, read again at every request."""
        if not results_path.is_file():
            raise HTTPException(status_code=404, detail="No results were recorded yet.")
        pipeline: RagPipeline = request.app.state.pipeline
        try:
            return load_results(results_path, golden_set, pipeline.search)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/evaluation/retrieval")
    def evaluate_retrieval(request: Request) -> RetrievalReport:
        try:
            return retrieval_report(request.app.state.pipeline, golden_set, config.reranking.top_k)
        except EmptyCollectionError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/api/evaluation/generation")
    def evaluate_generation(body: GenerationRequest, request: Request) -> GenerationRow:
        question = golden.get(body.id)
        if question is None:
            raise HTTPException(status_code=404, detail=f"The golden set has no question {body.id!r}.")
        pipeline: RagPipeline = request.app.state.pipeline
        result = answer_with(pipeline, question.question, ())
        try:
            markers = config.evaluation.abstention_markers
            return generation_row(pipeline, question, result, body.judge, markers)
        except AllowanceExhaustedError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error

    return app


def create_app() -> FastAPI:
    """Application factory for the server: ``uvicorn src.api:create_app --factory``."""
    config = load_config()
    golden_set = load_golden_set(project_path(config.paths.golden_set))
    results_path = project_path(config.paths.evaluation_results)
    return build_app(config, lambda: build_pipeline(config), golden_set, results_path)
