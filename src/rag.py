"""Retrieve, augment, generate: a question goes in, an answer with its sources comes out."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.config import RootConfig, project_path
from src.embedding import BgeM3Embedder, Embedder
from src.env import load_env_file, require_env
from src.llm import ChatClient
from src.retrieval import Hit
from src.vector_store import QdrantStore


@dataclass(frozen=True)
class RagAnswer:
    """Answer of the model, the model that wrote it and the passages it was given."""

    text: str
    model: str
    hits: tuple[Hit, ...]


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    """The prompt for the model: the retrieved passages, numbered and with their source, then the question."""
    passages = [
        f"[{number}] {Path(hit.chunk.source).name} | {hit.chunk.section}\n{hit.chunk.text}"
        for number, hit in enumerate(hits, start=1)
    ]
    return "Passages:\n\n" + "\n\n".join(passages) + f"\n\nQuestion: {question}"


class RagPipeline:
    """Answers questions from the chunks held in the vector store."""

    def __init__(
        self, embedder: Embedder, store: QdrantStore, llm: ChatClient, system_prompt: str, top_k: int
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._system_prompt = system_prompt
        self._top_k = top_k

    def indexed_chunks(self) -> int:
        """Number of chunks available for retrieval."""
        return self._store.count()

    def answer(self, question: str) -> RagAnswer:
        """Retrieve the closest chunks, put them in the prompt and let the model answer."""
        if not question.strip():
            raise ValueError("The question is empty")
        query = self._embedder.embed([question]).dense[0]
        hits = self._store.search(query, self._top_k)
        # The model needs the date to tell which of two versions of a rule is in force.
        system = f"{self._system_prompt} Today is {date.today():%d %B %Y}."
        reply = self._llm.ask(build_prompt(question, hits), system=system)
        return RagAnswer(reply.text, reply.model, tuple(hits))


def build_pipeline(config: RootConfig) -> RagPipeline:
    """Create the pipeline described by the configuration; the API key comes from the environment."""
    load_env_file(project_path(config.paths.env_file))
    return RagPipeline(
        embedder=BgeM3Embedder(config.embedding),
        store=QdrantStore.connect(config.vector_store, config.reranking.similarity_metric),
        llm=ChatClient(config.llm, require_env(config.llm.api_key_env)),
        system_prompt=config.rag.system_prompt,
        top_k=config.reranking.top_k,
    )
