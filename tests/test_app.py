"""Vector store, RAG pipeline, ingestion and API tested together, with no server, no model and no network.

The vector store runs inside the test process, the embedder returns hand-made
vectors and the language model is replaced by a function.
"""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.api import build_app
from src.cache import DiskCache
from src.config import EvaluationConfig, RagConfig, RetrievalConfig, VectorStoreConfig, load_config
from src.evaluation import (
    ContextItem,
    EvaluationResults,
    GoldenQuestion,
    generation_row,
    record_answers,
    retrieval_report,
    save_results,
)
from src.ingestion.chunking import Chunk
from src.ingestion.main import fill_store
from src.llm import AllowanceExhaustedError, ChatClient
from src.rag import RagPipeline, Turn, build_prompt
from src.retrieval.embedding import EmbeddingBatch
from src.retrieval.reranking import Reranker
from src.retrieval.vector_store import EmptyCollectionError, Hit, QdrantStore, StoreUnavailableError

CONFIG = load_config()
NO_RESULTS = Path("results-that-do-not-exist.json")
RAG = RagConfig(system_prompt="Use only the passages.", retrieval_history_turns=1, prompt_history_turns=2)
OLD_TURNS = [
    Turn("Che ore sono?", "Non lo so."),
    Turn("Chi firma il contratto?", "Elena Sartori."),
    Turn("Con quanto preavviso di recesso?", "45 giorni."),
]
CHUNKS = [
    Chunk("01_contratti/CTR-2024-017_Beta.pdf", 0, "Il preavviso di recesso è di 60 giorni.", "Art. 7", 2),
    Chunk("02_addenda/CTR-2024-017_addendum-01.pdf", 0, "Il preavviso diventa di 45 giorni.", "Art. 1", 1),
    Chunk("03_manuali_tecnici/MM-P7.pdf", 0, "E-4012: infiltrazione d'acqua.", "E-4012", 9),
]


GOLDEN = [
    GoldenQuestion(
        id="notice",
        type="superseded",
        question="Con quanto preavviso di recesso?",
        answer="45 giorni.",
        facts=(("45 giorni",),),
        context=(ContextItem("CTR-2024-017_addendum-01", "Art. 1"),),
    ),
    GoldenQuestion("phone", "unanswerable", "Qual è il numero di cellulare?", "Non si sa.", (), ()),
]


class KeywordEmbedder:
    """Stand-in for the embedding model: one axis for withdrawal, one for alarms, one for the rest."""

    dimension = 3

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        rows = [
            [1.0 if "recesso" in text or "preavviso" in text else 0.0, 1.0 if "E-4012" in text else 0.0, 0.1]
            for text in texts
        ]
        # The sparse side knows two terms only: token 7 is the alarm code, token 3 the word "giorni".
        terms = [{7: 1.0} if "E-4012" in text else {3: 0.5} if "giorni" in text else {} for text in texts]
        return EmbeddingBatch(np.array(rows, dtype=np.float32), tuple(terms))


WITHDRAWAL = np.array([1.0, 0.0, 0.1], dtype=np.float32)
ALARM = np.array([0.0, 1.0, 0.1], dtype=np.float32)


def make_store(
    metric: str = "cosine", mode: str = "dense", client: QdrantClient | None = None
) -> QdrantStore:
    """An empty store that lives in the memory of the test process."""
    retrieval = RetrievalConfig(mode=mode, similarity_metric=metric, candidates=3, rrf_k=60)
    return QdrantStore(client or QdrantClient(":memory:"), "test_chunks", retrieval)


def filled_store(metric: str = "cosine", mode: str = "dense") -> QdrantStore:
    """A store that holds the three chunks."""
    store = make_store(metric, mode)
    fill_store(CHUNKS, KeywordEmbedder(), store, batch_size=2)
    return store


class NumberReranker:
    """Stand-in for the cross-encoder: a text that names the addendum wins, and every input is kept."""

    name = "test-reranker"

    def __init__(self) -> None:
        self.seen: list[tuple[str, list[str]]] = []

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        self.seen.append((question, list(texts)))
        return [0.9 if "addendum" in text else 0.2 for text in texts]


def make_pipeline(
    store: QdrantStore,
    handler: Callable[[httpx.Request], httpx.Response],
    reranker: Reranker | None = None,
    top_k: int = 2,
) -> RagPipeline:
    """A pipeline whose language model is a function."""
    llm = ChatClient(CONFIG.llm, "secret-key", transport=httpx.MockTransport(handler))
    return RagPipeline(KeywordEmbedder(), store, llm, RAG, top_k, reranker, candidates=3)


def make_app(
    pipeline: RagPipeline, golden: Sequence[GoldenQuestion] = (), results: Path = NO_RESULTS
) -> FastAPI:
    """The application around a pipeline that already exists."""
    return build_app(CONFIG, lambda: pipeline, golden, results)


def reply_45_days(request: httpx.Request) -> httpx.Response:
    """A model that always gives the same answer."""
    choices = [{"message": {"content": "45 giorni."}}]
    return httpx.Response(200, json={"model": "test-model", "choices": choices})


def test_store_keeps_chunks_and_ranks_them() -> None:
    store = filled_store()
    assert store.count() == 3
    hits = store.search(WITHDRAWAL, {}, limit=2)
    assert {hit.chunk for hit in hits} == set(CHUNKS[:2])
    assert hits[0].score >= hits[1].score > 0.9
    alarm = store.search(ALARM, {}, limit=1)
    assert alarm[0].chunk == CHUNKS[2]


def test_sparse_search_finds_the_exact_term_whatever_the_meaning() -> None:
    hits = filled_store(mode="sparse").search(WITHDRAWAL, {7: 1.0}, limit=3)
    assert [hit.chunk for hit in hits] == [CHUNKS[2]]


def test_hybrid_search_fuses_the_two_rankings_by_position() -> None:
    store = filled_store(mode="hybrid")
    hits = store.search(WITHDRAWAL, {7: 1.0}, limit=3)
    # The alarm chunk is first by terms and last by meaning; the others are found by meaning only.
    assert {hit.chunk for hit in hits} == set(CHUNKS)
    assert hits[0].chunk == CHUNKS[2]
    assert hits[0].score == pytest.approx(1 / (60 + 1) + 1 / (60 + 3), rel=0.05)
    assert hits[0].score > hits[1].score >= hits[2].score
    agreed = store.search(ALARM, {7: 1.0}, limit=1)
    assert agreed[0].chunk == CHUNKS[2]
    assert agreed[0].score == pytest.approx(2 / (60 + 1), rel=0.05)


def test_reranker_reorders_the_candidates_and_keeps_the_best() -> None:
    reranker = NumberReranker()
    pipeline = make_pipeline(filled_store(), reply_45_days, reranker, top_k=1)
    hits = pipeline.retrieve("Con quanto preavviso di recesso?")
    assert [(hit.chunk, hit.score) for hit in hits] == [(CHUNKS[1], 0.9)]
    question, texts = reranker.seen[0]
    assert question == "Con quanto preavviso di recesso?"
    assert len(texts) == 3
    assert "CTR-2024-017_addendum-01.pdf | Art. 1\nIl preavviso diventa di 45 giorni." in texts
    assert pipeline.search == "dense search, 3 candidates re-ranked by test-reranker, top 1"
    assert make_pipeline(filled_store(mode="hybrid"), reply_45_days).search == "hybrid search, top 2"
    with pytest.raises(ValueError, match="at least top_k candidates"):
        make_pipeline(filled_store(), reply_45_days, reranker, top_k=4)


def test_euclidean_scores_are_higher_for_closer_chunks() -> None:
    hits = filled_store("euclidean").search(ALARM, {}, limit=3)
    assert hits[0].chunk == CHUNKS[2]
    assert hits[0].score == pytest.approx(0.0, abs=1e-6)
    assert hits[0].score > hits[1].score


def test_store_refuses_what_it_cannot_answer() -> None:
    assert make_store().count() == 0
    with pytest.raises(EmptyCollectionError, match="run the ingestion"):
        make_store().search(WITHDRAWAL, {}, limit=2)
    with pytest.raises(ValueError, match="limit"):
        filled_store().search(WITHDRAWAL, {}, limit=0)
    with pytest.raises(ValueError, match="similarity metric"):
        make_store("manhattan")
    with pytest.raises(ValueError, match="retrieval mode"):
        make_store(mode="keyword")
    client = QdrantClient(":memory:")
    fill_store(CHUNKS, KeywordEmbedder(), make_store(client=client), batch_size=3)
    with pytest.raises(ValueError, match=r"not built for the dot metric.*ingestion\.force=true"):
        make_store("dot", client=client).search(WITHDRAWAL, {}, limit=2)


def test_collection_without_sparse_vectors_is_refused() -> None:
    client = QdrantClient(":memory:")
    client.create_collection("test_chunks", vectors_config=VectorParams(size=3, distance=Distance.COSINE))
    client.upsert("test_chunks", [PointStruct(id=0, vector=[1.0, 0.0, 0.1], payload={})])
    store = make_store(client=client)
    assert store.layout_problem() == "it does not hold a dense and a sparse vector for every chunk"
    with pytest.raises(ValueError, match=r"cannot be searched.*ingestion\.force=true"):
        store.search(WITHDRAWAL, {}, limit=2)


def test_filling_the_store_replaces_what_was_there() -> None:
    store = filled_store()
    fill_store(CHUNKS[:1], KeywordEmbedder(), store, batch_size=8)
    assert store.count() == 1
    with pytest.raises(ValueError, match="batch_size"):
        fill_store(CHUNKS, KeywordEmbedder(), store, batch_size=0)


def test_prompt_shows_each_passage_with_its_source() -> None:
    prompt = build_prompt("Quanti giorni?", [Hit(CHUNKS[0], 0.9), Hit(CHUNKS[1], 0.8)])
    assert "[1] CTR-2024-017_Beta.pdf | Art. 7\nIl preavviso di recesso è di 60 giorni." in prompt
    assert "[2] CTR-2024-017_addendum-01.pdf | Art. 1" in prompt
    assert prompt.endswith("Question: Quanti giorni?")


def test_pipeline_retrieves_augments_and_generates() -> None:
    seen: list[dict[str, list[dict[str, str]]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return reply_45_days(request)

    pipeline = make_pipeline(filled_store(), handler)
    result = pipeline.answer("Con quanto preavviso di recesso?")
    assert (result.text, result.model) == ("45 giorni.", "test-model")
    assert {hit.chunk for hit in result.hits} == set(CHUNKS[:2])
    system, user = seen[0]["messages"]
    assert system["content"].startswith("Use only the passages. Today is ")
    assert "45 giorni" in user["content"] and "E-4012" not in user["content"]
    assert pipeline.indexed_chunks() == 3
    with pytest.raises(ValueError, match="empty"):
        pipeline.answer("   ")


def test_follow_up_is_searched_with_the_previous_question_and_sent_with_the_last_turns() -> None:
    seen: list[dict[str, list[dict[str, str]]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return reply_45_days(request)

    pipeline = make_pipeline(filled_store(), handler)
    alone = pipeline.answer("E il fornitore?")
    follow_up = pipeline.answer("E il fornitore?", OLD_TURNS)
    # Alone, the follow-up says nothing about withdrawal; with the previous question it does.
    assert {hit.chunk for hit in follow_up.hits} == set(CHUNKS[:2])
    assert alone.hits[0].score < follow_up.hits[0].score
    roles = [message["role"] for message in seen[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    contents = [message["content"] for message in seen[1]["messages"]]
    expected = [text for turn in OLD_TURNS[1:] for text in (turn.question, turn.answer)]
    assert contents[1:5] == expected
    assert contents[5].endswith("Question: E il fornitore?")
    assert [message["role"] for message in seen[0]["messages"]] == ["system", "user"]


def test_api_passes_the_conversation_to_the_pipeline() -> None:
    seen: list[dict[str, list[dict[str, str]]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return reply_45_days(request)

    history = [{"question": "Con quanto preavviso di recesso?", "answer": "45 giorni."}]
    with TestClient(make_app(make_pipeline(filled_store(), handler))) as client:
        response = client.post("/api/ask", json={"question": "E il fornitore?", "history": history})
    assert response.status_code == 200
    assert [message["role"] for message in seen[0]["messages"]] == ["system", "user", "assistant", "user"]
    assert {source["document"] for source in response.json()["sources"]} == {
        "CTR-2024-017_Beta.pdf",
        "CTR-2024-017_addendum-01.pdf",
    }


def test_api_answers_with_sources() -> None:
    pipeline = make_pipeline(filled_store(), reply_45_days)
    with TestClient(make_app(pipeline)) as client:
        health = client.get("/api/health").json()
        assert (health["status"], health["chunks"], health["model"]) == ("ok", 3, CONFIG.llm.model)
        response = client.post("/api/ask", json={"question": "Con quanto preavviso di recesso?"})
    assert response.status_code == 200
    body = response.json()
    assert (body["answer"], body["model"]) == ("45 giorni.", "test-model")
    assert [source["position"] for source in body["sources"]] == [1, 2]
    documents = {source["document"] for source in body["sources"]}
    assert documents == {"CTR-2024-017_Beta.pdf", "CTR-2024-017_addendum-01.pdf"}
    assert {"score", "section", "page", "text"} <= set(body["sources"][0])


def test_api_reports_each_kind_of_failure() -> None:
    limit = CONFIG.api.max_question_chars
    with TestClient(make_app(make_pipeline(filled_store(), reply_45_days))) as client:
        assert client.post("/api/ask", json={"question": "  "}).status_code == 422
        too_long = client.post("/api/ask", json={"question": "x" * (limit + 1)})
        assert too_long.status_code == 422 and str(limit) in too_long.json()["detail"]
    with TestClient(make_app(make_pipeline(make_store(), reply_45_days))) as client:
        empty = client.post("/api/ask", json={"question": "Quanti giorni?"})
        assert empty.status_code == 503 and "run the ingestion" in empty.json()["detail"]
    down = make_pipeline(filled_store(), lambda request: httpx.Response(401, text="invalid key"))
    with TestClient(make_app(down)) as client:
        failed = client.post("/api/ask", json={"question": "Quanti giorni?"})
        assert failed.status_code == 502 and "HTTP 401" in failed.json()["detail"]


def test_api_measures_retrieval_on_the_golden_set_without_calling_the_model() -> None:
    def no_model(request: httpx.Request) -> httpx.Response:
        raise AssertionError("retrieval evaluation must not call the language model")

    with TestClient(make_app(make_pipeline(filled_store(), no_model), GOLDEN)) as client:
        golden = client.get("/api/evaluation/golden").json()
        report = client.post("/api/evaluation/retrieval").json()
    assert [entry["id"] for entry in golden["questions"]] == ["notice", "phone"]
    passage = {"document": "CTR-2024-017_addendum-01", "section": "Art. 1"}
    assert golden["questions"][0]["context"] == [passage]
    assert report["questions"] == 1
    row = report["rows"][0]
    assert (row["id"], row["recall"], row["precision"], row["context_recall"]) == ("notice", 1.0, 0.5, 1.0)
    assert row["mrr"] == 1 / row["first_relevant"]
    assert [passage["relevant"] for passage in row["retrieved"]].count(True) == 1
    assert report["summary"]["recall"] == 1.0
    assert report["by_type"] == [{"type": "superseded", "questions": 1, **report["summary"]}]


def test_api_checks_one_answer_and_reports_the_judge() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        system = json.loads(request.content)["messages"][0]["content"]
        text = '{"claims": 2, "supported": 1}' if system.startswith("You check") else "Sono 45 giorni."
        return httpx.Response(200, json={"model": "test-model", "choices": [{"message": {"content": text}}]})

    with TestClient(make_app(make_pipeline(filled_store(), handler), GOLDEN)) as client:
        plain = client.post("/api/evaluation/generation", json={"id": "notice"}).json()
        judged = client.post("/api/evaluation/generation", json={"id": "notice", "judge": True}).json()
        invented = client.post("/api/evaluation/generation", json={"id": "phone"}).json()
        unknown = client.post("/api/evaluation/generation", json={"id": "missing"})
    assert (plain["passed"], plain["answer"], plain["faithfulness"]) == (True, "Sono 45 giorni.", None)
    assert plain["facts"] == [{"wordings": ["45 giorni"], "found": True}]
    assert judged["faithfulness"] == {"claims": 2, "supported": 1, "score": 0.5}
    assert (invented["passed"], invented["abstained"]) == (False, False)
    assert unknown.status_code == 404


def test_judge_that_fails_leaves_the_answer_and_says_why() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        system = json.loads(request.content)["messages"][0]["content"]
        text = "I cannot count the claims." if system.startswith("You check") else "Sono 45 giorni."
        return httpx.Response(200, json={"model": "test-model", "choices": [{"message": {"content": text}}]})

    with TestClient(make_app(make_pipeline(filled_store(), handler), GOLDEN)) as client:
        row = client.post("/api/evaluation/generation", json={"id": "notice", "judge": True}).json()
    assert (row["passed"], row["faithfulness"]) == (True, None)
    assert "did not reply with JSON" in row["faithfulness_error"]


def test_api_serves_recorded_results_and_refuses_those_of_another_golden_set(tmp_path: Path) -> None:
    pipeline = make_pipeline(filled_store(), reply_45_days)
    rows = {
        question.id: generation_row(pipeline, question, pipeline.answer(question.question), False, [])
        for question in GOLDEN
    }
    results = EvaluationResults(
        created_at="2026-10-09T12:00:00+02:00",
        judged=False,
        retrieval=retrieval_report(pipeline, GOLDEN, 2),
        generation=rows,
        errors={},
    )
    path = tmp_path / "results.json"
    with TestClient(make_app(pipeline, GOLDEN, path)) as client:
        assert client.get("/api/evaluation/results").status_code == 404
        save_results(results, path)
        recorded = client.get("/api/evaluation/results").json()
        assert recorded["created_at"] == "2026-10-09T12:00:00+02:00"
        assert sorted(recorded["generation"]) == ["notice", "phone"]
        assert recorded["retrieval"]["summary"]["recall"] == 1.0
        assert recorded["retrieval"]["search"] == "dense search, top 2"
    with TestClient(make_app(pipeline, GOLDEN[:1], path)) as client:
        stale = client.get("/api/evaluation/results")
    assert stale.status_code == 409 and "another golden set" in stale.json()["detail"]
    reranked = make_pipeline(filled_store(), reply_45_days, NumberReranker())
    with TestClient(make_app(reranked, GOLDEN, path)) as client:
        stale = client.get("/api/evaluation/results")
    assert stale.status_code == 409
    assert "recorded with dense search, top 2, the system now uses" in stale.json()["detail"]


def evaluation_settings(fresh: bool = False) -> EvaluationConfig:
    """The shipped evaluation settings, without the judge and with two tries."""
    markers = list(CONFIG.evaluation.abstention_markers)
    return EvaluationConfig(False, 2, 3.0, fresh, markers)


def used_up(request: httpx.Request) -> httpx.Response:
    """The provider says that no request is left until tomorrow."""
    reset = str(int((time.time() + 6 * 3600) * 1000))
    headers = {"X-RateLimit-Limit": "50", "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset}
    return httpx.Response(429, headers=headers, json={"error": {"message": "Rate limit exceeded"}})


def test_recorded_run_stops_when_the_allowance_ends_and_the_next_run_continues(tmp_path: Path) -> None:
    checkpoint = tmp_path / "results.partial.json"
    asked: list[str] = []

    def one_answer_then_no_allowance(request: httpx.Request) -> httpx.Response:
        asked.append(json.loads(request.content)["messages"][-1]["content"])
        return reply_45_days(request) if len(asked) == 1 else used_up(request)

    pipeline = make_pipeline(filled_store(), one_answer_then_no_allowance)
    with pytest.raises(AllowanceExhaustedError):
        record_answers(pipeline, GOLDEN, evaluation_settings(), checkpoint, sleep=lambda _: None)
    assert len(asked) == 2
    assert checkpoint.is_file()

    asked.clear()

    def does_not_know(request: httpx.Request) -> httpx.Response:
        asked.append(json.loads(request.content)["messages"][-1]["content"])
        body = {"model": "test-model", "choices": [{"message": {"content": "Non lo so."}}]}
        return httpx.Response(200, json=body)

    pipeline = make_pipeline(filled_store(), does_not_know)
    settings = evaluation_settings()
    generation, errors = record_answers(pipeline, GOLDEN, settings, checkpoint, sleep=lambda _: None)
    assert len(asked) == 1 and "cellulare" in asked[0]
    assert errors == {}
    assert generation["notice"].passed and generation["phone"].passed

    asked.clear()
    record_answers(pipeline, GOLDEN, evaluation_settings(fresh=True), checkpoint, sleep=lambda _: None)
    assert len(asked) == 2


def test_recorded_run_keeps_the_failures_and_refuses_a_checkpoint_of_other_settings(tmp_path: Path) -> None:
    checkpoint = tmp_path / "results.partial.json"
    pauses: list[float] = []

    def busy_on_the_phone_question(request: httpx.Request) -> httpx.Response:
        if "cellulare" in json.loads(request.content)["messages"][-1]["content"]:
            return httpx.Response(400, text="bad request")
        return reply_45_days(request)

    pipeline = make_pipeline(filled_store(), busy_on_the_phone_question)
    settings = evaluation_settings()
    generation, errors = record_answers(pipeline, GOLDEN, settings, checkpoint, sleep=pauses.append)
    assert set(generation) == {"notice"}
    assert "HTTP 400" in errors["phone"]
    assert pauses == [3.0]

    judged = EvaluationConfig(True, 2, 3.0, False, list(CONFIG.evaluation.abstention_markers))
    with pytest.raises(ValueError, match=r"evaluation\.fresh=true"):
        record_answers(pipeline, GOLDEN, judged, checkpoint, sleep=pauses.append)
    reranked = make_pipeline(filled_store(), busy_on_the_phone_question, NumberReranker())
    with pytest.raises(ValueError, match=r"another golden set, search or judge"):
        record_answers(reranked, GOLDEN, settings, checkpoint, sleep=pauses.append)


def test_server_that_does_not_answer_is_reported_at_once_with_the_command_that_starts_it() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = VectorStoreConfig(f"http://127.0.0.1:{port}", "chunks", 2.0, "chunks.snapshot")
    retrieval = RetrievalConfig(mode="dense", similarity_metric="cosine", candidates=3, rrf_k=60)
    with pytest.raises(StoreUnavailableError, match="docker compose up -d qdrant"):
        QdrantStore.connect(config, retrieval)


def test_question_asked_before_is_answered_from_disk(tmp_path: Path) -> None:
    calls: list[str] = []

    def counting(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["messages"][-1]["content"])
        return reply_45_days(request)

    reranker = NumberReranker()
    llm = ChatClient(CONFIG.llm, "secret-key", transport=httpx.MockTransport(counting))
    caches = DiskCache(tmp_path / "retrieval"), DiskCache(tmp_path / "answers")

    def pipeline(rag: RagConfig = RAG) -> RagPipeline:
        return RagPipeline(KeywordEmbedder(), filled_store(), llm, rag, 2, reranker, 3, *caches)

    first = pipeline().answer("Con quanto preavviso di recesso?")
    again = pipeline().answer("Con quanto preavviso di recesso?")
    assert (first.cached, again.cached) == (False, True)
    assert (again.text, again.model, again.hits) == (first.text, first.model, first.hits)
    assert len(calls) == 1 and len(reranker.seen) == 1

    # Another question, another conversation or other instructions are not the same request.
    pipeline().answer("Cosa significa E-4012?")
    pipeline().answer("Con quanto preavviso di recesso?", [Turn("Chi firma?", "Elena Sartori.")])
    other_rules = RagConfig("Answer in one word.", RAG.retrieval_history_turns, RAG.prompt_history_turns)
    assert not pipeline(other_rules).answer("Con quanto preavviso di recesso?").cached
    assert len(calls) == 4

    # Asking afresh calls the model and replaces what was saved; the passages are still the saved ones.
    assert not pipeline().answer("Con quanto preavviso di recesso?", fresh=True).cached
    assert len(calls) == 5 and len(reranker.seen) == 3

    with TestClient(make_app(pipeline())) as client:
        body = client.post("/api/ask", json={"question": "Con quanto preavviso di recesso?"}).json()
    assert body["cached"] is True and len(calls) == 5


def test_saved_passages_are_not_reused_for_another_search_or_collection(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    llm = ChatClient(CONFIG.llm, "secret-key", transport=httpx.MockTransport(reply_45_days))
    reranker = NumberReranker()
    store = filled_store()
    question = "Con quanto preavviso di recesso?"
    RagPipeline(KeywordEmbedder(), store, llm, RAG, 2, reranker, 3, cache).retrieve(question)
    RagPipeline(KeywordEmbedder(), store, llm, RAG, 1, reranker, 3, cache).retrieve(question)
    assert len(reranker.seen) == 2
    fill_store(CHUNKS[:2], KeywordEmbedder(), store, batch_size=2)
    hits = RagPipeline(KeywordEmbedder(), store, llm, RAG, 2, reranker, 2, cache).retrieve(question)
    assert len(reranker.seen) == 3 and len(hits) == 2
    # Same number of chunks, other content: still another collection for the cache.
    renamed = [Chunk("09_fornitori_esteri/MSA.pdf", 0, "Il preavviso è di 90 giorni.", "7.", 1), CHUNKS[1]]
    before = store.fingerprint()
    fill_store(renamed, KeywordEmbedder(), store, batch_size=2)
    assert store.fingerprint() != before and store.count() == 2
    cache.clear()
    assert cache.get("anything") is None


def test_points_carry_the_folder_of_their_document_and_still_give_back_the_chunk() -> None:
    client = QdrantClient(":memory:")
    store = make_store(client=client)
    fill_store(CHUNKS, KeywordEmbedder(), store, batch_size=3)
    folders = {
        point.payload["folder"] for point in client.scroll("test_chunks", limit=10)[0] if point.payload
    }
    assert folders == {"01_contratti", "02_addenda", "03_manuali_tecnici"}
    assert {hit.chunk for hit in store.search(WITHDRAWAL, {}, limit=3)} == set(CHUNKS)
