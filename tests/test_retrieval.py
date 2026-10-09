"""The vector index must rank chunks by the chosen metric and survive a round trip to disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.chunking import Chunk
from src.retrieval import VectorIndex


def make_index() -> VectorIndex:
    """Three chunks whose vectors point right, up, and far to the right."""
    chunks = [Chunk("doc.pdf", index, text, "", 1) for index, text in enumerate(["right", "up", "far right"])]
    return VectorIndex(chunks, np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 0.3]], dtype=np.float32))


def texts(index: VectorIndex, metric: str, top_k: int = 3) -> list[str]:
    """Texts returned for a query that points right, best first."""
    return [hit.chunk.text for hit in index.search(np.array([1.0, 0.0], dtype=np.float32), top_k, metric)]


def test_each_metric_has_its_own_ranking() -> None:
    index = make_index()
    assert texts(index, "cosine") == ["right", "far right", "up"]
    assert texts(index, "dot") == ["far right", "right", "up"]
    assert texts(index, "euclidean") == ["right", "up", "far right"]


def test_scores_are_sorted_and_top_k_limits_the_result() -> None:
    hits = make_index().search(np.array([1.0, 0.0], dtype=np.float32), 2, "cosine")
    assert [hit.chunk.text for hit in hits] == ["right", "far right"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score > hits[1].score
    assert len(texts(make_index(), "cosine", top_k=10)) == 3


def test_index_is_read_back_from_disk(tmp_path: Path) -> None:
    index = make_index()
    index.save(tmp_path / "index")
    loaded = VectorIndex.load(tmp_path / "index")
    assert loaded.chunks == index.chunks
    assert np.array_equal(loaded.vectors, index.vectors)


def test_invalid_inputs_are_errors() -> None:
    index = make_index()
    query = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="similarity metric"):
        index.search(query, 3, "manhattan")
    with pytest.raises(ValueError, match="top_k"):
        index.search(query, 0, "cosine")
    with pytest.raises(ValueError, match="does not match"):
        index.search(np.zeros(3, dtype=np.float32), 3, "cosine")
    with pytest.raises(ValueError, match="one vector per chunk"):
        VectorIndex(index.chunks, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="at least one chunk"):
        VectorIndex([], np.zeros((0, 2), dtype=np.float32))
