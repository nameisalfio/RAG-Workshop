"""Backend: the HTTP API that the web interface calls."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.config import RootConfig, load_config
from src.llm import LlmError
from src.rag import RagPipeline, build_pipeline
from src.vector_store import EmptyCollectionError


class Question(BaseModel):
    """Body of a request to /api/ask."""

    question: str


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


def build_app(config: RootConfig, pipeline_factory: Callable[[], RagPipeline]) -> FastAPI:
    """Create the application; the pipeline is built once, when the server starts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline_factory()
        yield

    app = FastAPI(title="RAG workshop API", lifespan=lifespan)

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
        pipeline: RagPipeline = request.app.state.pipeline
        try:
            result = pipeline.answer(question)
        except EmptyCollectionError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except LlmError as error:
            detail = f"The language model did not answer. {error}"
            raise HTTPException(status_code=502, detail=detail) from error
        sources = [
            Source(
                position=position,
                score=hit.score,
                document=Path(hit.chunk.source).name,
                section=hit.chunk.section,
                page=hit.chunk.page,
                text=hit.chunk.text,
            )
            for position, hit in enumerate(result.hits, start=1)
        ]
        return Answer(answer=result.text, model=result.model, sources=sources)

    return app


def create_app() -> FastAPI:
    """Application factory for the server: ``uvicorn src.api:create_app --factory``."""
    config = load_config()
    return build_app(config, lambda: build_pipeline(config))
