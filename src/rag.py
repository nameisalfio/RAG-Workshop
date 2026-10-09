"""Retrieve, augment, generate: a question goes in, an answer with its sources comes out."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.config import RagConfig, RootConfig, project_path
from src.embedding import BgeM3Embedder, Embedder
from src.env import load_env_file, require_env
from src.llm import ChatClient
from src.retrieval import Hit
from src.vector_store import QdrantStore


@dataclass(frozen=True)
class Turn:
    """One earlier exchange of a conversation."""

    question: str
    answer: str


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
        self, embedder: Embedder, store: QdrantStore, llm: ChatClient, config: RagConfig, top_k: int
    ) -> None:
        if config.retrieval_history_turns < 0 or config.prompt_history_turns < 0:
            raise ValueError("The numbers of history turns cannot be negative")
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._config = config
        self._top_k = top_k

    @property
    def llm(self) -> ChatClient:
        """The client of the language model, for callers that need a call of their own."""
        return self._llm

    def indexed_chunks(self) -> int:
        """Number of chunks available for retrieval."""
        return self._store.count()

    def retrieve(self, question: str, history: Sequence[Turn] = ()) -> list[Hit]:
        """The chunks closest to a question, searched together with the last questions before it."""
        if not question.strip():
            raise ValueError("The question is empty")
        searched = _last(history, self._config.retrieval_history_turns)
        search_text = "\n".join([*(turn.question for turn in searched), question])
        return self._store.search(self._embedder.embed([search_text]).dense[0], self._top_k)

    def answer(self, question: str, history: Sequence[Turn] = ()) -> RagAnswer:
        """Retrieve the closest chunks, put them in the prompt and let the model answer.

        ``history`` holds the earlier exchanges of the conversation, oldest first. The last
        questions are searched together with the new one, so that a follow-up such as
        "and for the supplier?" still finds the passages of the topic under discussion.
        """
        hits = self.retrieve(question, history)
        # The model needs the date to tell which of two versions of a rule is in force.
        system = f"{self._config.system_prompt} Today is {date.today():%d %B %Y}."
        recalled = _last(history, self._config.prompt_history_turns)
        earlier = [(turn.question, turn.answer) for turn in recalled]
        reply = self._llm.ask(build_prompt(question, hits), system=system, history=earlier)
        return RagAnswer(reply.text, reply.model, tuple(hits))


def _last(history: Sequence[Turn], count: int) -> Sequence[Turn]:
    """The most recent turns of a conversation, oldest first; none when count is zero."""
    return history[max(len(history) - count, 0) :] if count > 0 else ()


def build_pipeline(config: RootConfig) -> RagPipeline:
    """Create the pipeline described by the configuration; the API key comes from the environment."""
    load_env_file(project_path(config.paths.env_file))
    return RagPipeline(
        embedder=BgeM3Embedder(config.embedding),
        store=QdrantStore.connect(config.vector_store, config.reranking.similarity_metric),
        llm=ChatClient(config.llm, require_env(config.llm.api_key_env)),
        config=config.rag,
        top_k=config.reranking.top_k,
    )
