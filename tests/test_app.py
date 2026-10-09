"""Vector store, RAG pipeline, ingestion and API tested together, with no server, no model and no network.

The vector store runs inside the test process, the embedder returns hand-made
vectors and the language model is replaced by a function.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from src.api import build_app
from src.chunking import Chunk
from src.config import RagConfig, load_config
from src.embedding import EmbeddingBatch
from src.evaluation import ContextItem, GoldenQuestion
from src.ingest import fill_store
from src.llm import ChatClient
from src.rag import RagPipeline, Turn, build_prompt
from src.retrieval import Hit
from src.vector_store import EmptyCollectionError, QdrantStore

CONFIG = load_config()
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
        return EmbeddingBatch(np.array(rows, dtype=np.float32), tuple({} for _ in texts))


def make_store(metric: str = "cosine") -> QdrantStore:
    """An empty store that lives in the memory of the test process."""
    return QdrantStore(QdrantClient(":memory:"), "test_chunks", metric)


def filled_store(metric: str = "cosine") -> QdrantStore:
    """A store that holds the three chunks."""
    store = make_store(metric)
    fill_store(CHUNKS, KeywordEmbedder(), store, batch_size=2)
    return store


def make_pipeline(store: QdrantStore, handler: Callable[[httpx.Request], httpx.Response]) -> RagPipeline:
    """A pipeline whose language model is a function."""
    llm = ChatClient(CONFIG.llm, "secret-key", transport=httpx.MockTransport(handler))
    return RagPipeline(KeywordEmbedder(), store, llm, RAG, top_k=2)


def reply_45_days(request: httpx.Request) -> httpx.Response:
    """A model that always gives the same answer."""
    choices = [{"message": {"content": "45 giorni."}}]
    return httpx.Response(200, json={"model": "test-model", "choices": choices})


def test_store_keeps_chunks_and_ranks_them() -> None:
    store = filled_store()
    assert store.count() == 3
    hits = store.search(np.array([1.0, 0.0, 0.1], dtype=np.float32), top_k=2)
    assert {hit.chunk for hit in hits} == set(CHUNKS[:2])
    assert hits[0].score >= hits[1].score > 0.9
    alarm = store.search(np.array([0.0, 1.0, 0.1], dtype=np.float32), top_k=1)
    assert alarm[0].chunk == CHUNKS[2]


def test_euclidean_scores_are_higher_for_closer_chunks() -> None:
    hits = filled_store("euclidean").search(np.array([0.0, 1.0, 0.1], dtype=np.float32), top_k=3)
    assert hits[0].chunk == CHUNKS[2]
    assert hits[0].score == pytest.approx(0.0, abs=1e-6)
    assert hits[0].score > hits[1].score


def test_store_refuses_what_it_cannot_answer() -> None:
    query = np.array([1.0, 0.0, 0.1], dtype=np.float32)
    assert make_store().count() == 0
    with pytest.raises(EmptyCollectionError, match="run the ingestion"):
        make_store().search(query, top_k=2)
    with pytest.raises(ValueError, match="top_k"):
        filled_store().search(query, top_k=0)
    with pytest.raises(ValueError, match="similarity metric"):
        make_store("manhattan")
    client = QdrantClient(":memory:")
    fill_store(CHUNKS, KeywordEmbedder(), QdrantStore(client, "test_chunks", "cosine"), batch_size=3)
    with pytest.raises(ValueError, match="not built for the dot metric"):
        QdrantStore(client, "test_chunks", "dot").search(query, top_k=2)


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
    with TestClient(build_app(CONFIG, lambda: make_pipeline(filled_store(), handler), ())) as client:
        response = client.post("/api/ask", json={"question": "E il fornitore?", "history": history})
    assert response.status_code == 200
    assert [message["role"] for message in seen[0]["messages"]] == ["system", "user", "assistant", "user"]
    assert {source["document"] for source in response.json()["sources"]} == {
        "CTR-2024-017_Beta.pdf",
        "CTR-2024-017_addendum-01.pdf",
    }


def test_api_answers_with_sources() -> None:
    pipeline = make_pipeline(filled_store(), reply_45_days)
    with TestClient(build_app(CONFIG, lambda: pipeline, ())) as client:
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
    with TestClient(build_app(CONFIG, lambda: make_pipeline(filled_store(), reply_45_days), ())) as client:
        assert client.post("/api/ask", json={"question": "  "}).status_code == 422
        too_long = client.post("/api/ask", json={"question": "x" * (limit + 1)})
        assert too_long.status_code == 422 and str(limit) in too_long.json()["detail"]
    with TestClient(build_app(CONFIG, lambda: make_pipeline(make_store(), reply_45_days), ())) as client:
        empty = client.post("/api/ask", json={"question": "Quanti giorni?"})
        assert empty.status_code == 503 and "run the ingestion" in empty.json()["detail"]
    down = make_pipeline(filled_store(), lambda request: httpx.Response(401, text="invalid key"))
    with TestClient(build_app(CONFIG, lambda: down, ())) as client:
        failed = client.post("/api/ask", json={"question": "Quanti giorni?"})
        assert failed.status_code == 502 and "HTTP 401" in failed.json()["detail"]


def test_api_measures_retrieval_on_the_golden_set_without_calling_the_model() -> None:
    def no_model(request: httpx.Request) -> httpx.Response:
        raise AssertionError("retrieval evaluation must not call the language model")

    with TestClient(build_app(CONFIG, lambda: make_pipeline(filled_store(), no_model), GOLDEN)) as client:
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

    with TestClient(build_app(CONFIG, lambda: make_pipeline(filled_store(), handler), GOLDEN)) as client:
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

    with TestClient(build_app(CONFIG, lambda: make_pipeline(filled_store(), handler), GOLDEN)) as client:
        row = client.post("/api/evaluation/generation", json={"id": "notice", "judge": True}).json()
    assert (row["passed"], row["faithfulness"]) == (True, None)
    assert "did not reply with JSON" in row["faithfulness_error"]
