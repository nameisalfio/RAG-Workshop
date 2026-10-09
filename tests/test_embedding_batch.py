"""The embedding batch contract must reject inconsistent shapes."""

from __future__ import annotations

import numpy as np
import pytest

from src.embedding import EmbeddingBatch


def test_valid_batch() -> None:
    batch = EmbeddingBatch(dense=np.zeros((2, 4), dtype=np.float32), sparse=({1: 0.5}, {}))
    assert len(batch) == 2


def test_dense_must_be_two_dimensional() -> None:
    with pytest.raises(ValueError, match="2D"):
        EmbeddingBatch(dense=np.zeros(4, dtype=np.float32), sparse=({},))


def test_sparse_must_match_rows() -> None:
    with pytest.raises(ValueError, match="sparse"):
        EmbeddingBatch(dense=np.zeros((2, 4), dtype=np.float32), sparse=({},))
