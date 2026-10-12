"""Retrieve, augment, generate: a question goes in, an answer with its sources comes out."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from src.cache import DiskCache
from src.config import RagConfig, RootConfig, load_env_file, project_path, require_env
from src.ingestion.chunking import Chunk
from src.llm import ChatClient
from src.retrieval.embedding import BgeM3Embedder, Embedder
from src.retrieval.reranking import CrossEncoderReranker, Reranker
from src.retrieval.vector_store import Hit, QdrantStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Turn:
    """One earlier exchange of a conversation."""

    question: str
    answer: str


@dataclass(frozen=True)
class RagAnswer:
    """Answer of the model, the model that wrote it and the passages it was given.

    ``cached`` is True when the answer was read from disk and no model was called.
    """

    text: str
    model: str
    hits: tuple[Hit, ...]
    cached: bool = False


def passage_text(hit: Hit) -> str:
    """A chunk with the name of its document and section, which often say what the text leaves out."""
    return f"{Path(hit.chunk.source).name} | {hit.chunk.section}\n{hit.chunk.text}"


def format_passages(hits: Sequence[Hit]) -> str:
    """The retrieved passages as the model reads them: numbered, each with its document and section."""
    passages = [f"[{number}] {passage_text(hit)}" for number, hit in enumerate(hits, start=1)]
    return "Passages:\n\n" + "\n\n".join(passages)


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    """The prompt for the model: the retrieved passages, numbered and with their source, then the question."""
    return format_passages(hits) + f"\n\nQuestion: {question}"


class RagPipeline:
    """Answers questions from the chunks held in the vector store."""

    def __init__(
        self,
        embedder: Embedder,
        store: QdrantStore,
        llm: ChatClient,
        config: RagConfig,
        top_k: int,
        reranker: Reranker | None = None,
        candidates: int = 0,
        retrieval_cache: DiskCache | None = None,
        answer_cache: DiskCache | None = None,
    ) -> None:
        """``reranker`` orders ``candidates`` chunks again and keeps ``top_k``; None skips that step.

        Each cache, when given, saves the results of its step for the questions already asked.
        """
        if config.retrieval_history_turns < 0 or config.prompt_history_turns < 0:
            raise ValueError("The numbers of history turns cannot be negative")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if reranker is not None and candidates < top_k:
            raise ValueError(
                f"The re-ranker needs at least top_k candidates: {candidates} candidates, top_k {top_k}"
            )
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._config = config
        self._top_k = top_k
        self._reranker = reranker
        self._candidates = candidates
        self._retrieval_cache = retrieval_cache
        self._answer_cache = answer_cache

    @property
    def llm(self) -> ChatClient:
        """The client of the language model, for callers that need a call of their own."""
        return self._llm

    @property
    def search(self) -> str:
        """How the passages are found, in words; recorded with every measurement made with it."""
        described = f"{self._store.mode} search"
        if self._reranker is not None:
            described += f", {self._candidates} candidates re-ranked by {self._reranker.name}"
        return f"{described}, top {self._top_k}"

    def indexed_chunks(self) -> int:
        """Number of chunks available for retrieval."""
        return self._store.count()

    def retrieve(self, question: str, history: Sequence[Turn] = ()) -> list[Hit]:
        """The best chunks for a question, searched together with the last questions before it.

        Without a re-ranker these are the first chunks of the search. With one, the search
        returns more candidates, the re-ranker scores each against the question and the
        best are kept: the score of a hit is then the one given by the re-ranker.
        """
        if not question.strip():
            raise ValueError("The question is empty")
        searched = _last(history, self._config.retrieval_history_turns)
        search_text = "\n".join([*(turn.question for turn in searched), question])
        if self._retrieval_cache is None:
            return self._search(search_text)
        # The same text, searched in the same way in the same collection, gives the same chunks.
        key = (search_text, self.search, self._store.fingerprint())
        saved = self._retrieval_cache.get(*key)
        if saved is not None:
            return [Hit(Chunk(**entry["chunk"]), entry["score"]) for entry in saved]
        hits = self._search(search_text)
        self._retrieval_cache.put([{"chunk": asdict(hit.chunk), "score": hit.score} for hit in hits], *key)
        return hits

    def _search(self, search_text: str) -> list[Hit]:
        """Search the vector store and, with a re-ranker, order the candidates again."""
        embedded = self._embedder.embed([search_text])
        limit = self._top_k if self._reranker is None else self._candidates
        hits = self._store.search(embedded.dense[0], embedded.sparse[0], limit)
        if self._reranker is None:
            return hits
        scores = self._reranker.score(search_text, [passage_text(hit) for hit in hits])
        ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [Hit(hit.chunk, score) for hit, score in ranked[: self._top_k]]

    def answer(self, question: str, history: Sequence[Turn] = (), fresh: bool = False) -> RagAnswer:
        """Retrieve the closest chunks, put them in the prompt and let the model answer.

        ``history`` holds the earlier exchanges of the conversation, oldest first. The last
        questions are searched together with the new one, so that a follow-up such as
        "and for the supplier?" still finds the passages of the topic under discussion.
        With an answer cache, a prompt that was answered before gets the saved answer;
        ``fresh`` asks the model anyway and replaces what was saved.
        """
        hits = self.retrieve(question, history)
        prompt = build_prompt(question, hits)
        recalled = _last(history, self._config.prompt_history_turns)
        earlier = [(turn.question, turn.answer) for turn in recalled]
        # The key holds everything the answer depends on except the date, which changes every day:
        # a saved answer keeps the date of the day it was written.
        key = (self._llm.model, self._config.system_prompt, json.dumps(earlier, ensure_ascii=False), prompt)
        if self._answer_cache is not None and not fresh:
            saved = self._answer_cache.get(*key)
            if saved is not None:
                LOGGER.info("Answer read from the cache: the language model is not called")
                return RagAnswer(saved["text"], saved["model"], tuple(hits), cached=True)
        # The model needs the date to tell which of two versions of a rule is in force.
        system = f"{self._config.system_prompt} Today is {date.today():%d %B %Y}."
        reply = self._llm.ask(prompt, system=system, history=earlier)
        if self._answer_cache is not None:
            self._answer_cache.put({"text": reply.text, "model": reply.model}, *key)
        return RagAnswer(reply.text, reply.model, tuple(hits))


def _last(history: Sequence[Turn], count: int) -> Sequence[Turn]:
    """The most recent turns of a conversation, oldest first; none when count is zero."""
    return history[max(len(history) - count, 0) :] if count > 0 else ()


def build_pipeline(config: RootConfig) -> RagPipeline:
    """Create the pipeline described by the configuration; the API key comes from the environment."""
    load_env_file(project_path(config.paths.env_file))
    reranker = CrossEncoderReranker(config.reranking) if config.reranking.enabled else None
    cache_dir = project_path(config.cache.dir)
    return RagPipeline(
        embedder=BgeM3Embedder(config.embedding),
        store=QdrantStore.connect(config.vector_store, config.retrieval),
        llm=ChatClient(config.llm, require_env(config.llm.api_key_env)),
        config=config.rag,
        top_k=config.reranking.top_k,
        reranker=reranker,
        candidates=config.retrieval.candidates,
        retrieval_cache=DiskCache(cache_dir / "retrieval") if config.cache.retrieval else None,
        answer_cache=DiskCache(cache_dir / "answers") if config.cache.answers else None,
    )
