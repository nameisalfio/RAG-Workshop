"""The embedder must reproduce the reference outputs of the model and refuse inputs it cannot represent.

These tests load the real model (about 2.3 GB, downloaded on first use), so they are
selected with ``pytest -m model`` and left out of the default run.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.chunking import load_tokenizer
from src.config import load_config
from src.embedding import BgeM3Embedder

pytestmark = pytest.mark.model

# Inputs and outputs published in the BAAI/bge-m3 model card (computed there in half precision).
QUERIES = ["What is BGE M3?", "Defination of BM25"]
PASSAGES = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms "
    "appearing in each document",
]
CARD_DENSE = [[0.6265, 0.3477], [0.3499, 0.678]]
CARD_SPARSE = {
    "What": 0.08356,
    "is": 0.0814,
    "B": 0.1296,
    "GE": 0.252,
    "M": 0.1702,
    "3": 0.2695,
    "?": 0.04092,
}
CARD_LEXICAL_SCORE = 0.1955
TOLERANCE = 2e-3


@pytest.fixture(scope="module")
def embedder() -> BgeM3Embedder:
    """The configured embedding model, loaded once for the module."""
    return BgeM3Embedder(load_config().embedding)


def test_dense_and_sparse_outputs_match_the_model_card(embedder: BgeM3Embedder) -> None:
    queries, passages = embedder.embed(QUERIES), embedder.embed(PASSAGES)
    assert np.allclose(queries.dense @ passages.dense.T, CARD_DENSE, atol=TOLERANCE)
    tokenizer = load_tokenizer(load_config().embedding.model_name)
    weights: dict[str, float] = {}
    for token_id, weight in queries.sparse[0].items():
        token = tokenizer.id_to_token(token_id)
        assert token is not None
        weights[token.lstrip("▁")] = weight
    assert weights == pytest.approx(CARD_SPARSE, abs=TOLERANCE)
    shared = set(queries.sparse[0]) & set(passages.sparse[0])
    score = sum(queries.sparse[0][token] * passages.sparse[0][token] for token in shared)
    assert score == pytest.approx(CARD_LEXICAL_SCORE, abs=TOLERANCE)
    assert not set(queries.sparse[0]) & set(queries.sparse[1])


def test_dense_vectors_are_normalized_and_keep_input_order(embedder: BgeM3Embedder) -> None:
    batch = embedder.embed(["un cane", "un gatto", "una macchina"])
    assert batch.dense.shape == (3, embedder.dimension)
    assert batch.dense.dtype == np.float32
    assert np.allclose(np.linalg.norm(batch.dense, axis=1), 1.0, atol=1e-5)
    dog, cat, car = batch.dense
    assert cat @ dog > cat @ car


def test_batching_does_not_change_the_result(embedder: BgeM3Embedder) -> None:
    texts = ["Il Committente può recedere con preavviso di 60 giorni.", "un cane", "Allarme E-4012 sulla P7."]
    together = embedder.embed(texts)
    one_by_one = BgeM3Embedder(replace(load_config().embedding, batch_size=1)).embed(texts)
    assert np.allclose(together.dense, one_by_one.dense, atol=1e-4)
    for joint, single in zip(together.sparse, one_by_one.sparse, strict=True):
        assert sorted(joint) == sorted(single)


def test_empty_and_oversized_inputs_are_errors(embedder: BgeM3Embedder) -> None:
    with pytest.raises(ValueError, match="at least one text"):
        embedder.embed([])
    with pytest.raises(ValueError, match="empty"):
        embedder.embed(["un cane", "  "])
    short = BgeM3Embedder(replace(load_config().embedding, max_length=8))
    with pytest.raises(ValueError, match="over the limit"):
        short.embed(["Il Committente può recedere dal contratto con preavviso di 60 giorni."])
