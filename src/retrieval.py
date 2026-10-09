"""Store and retrieve stages: keep the vector of every chunk and find the chunks closest to a query."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.chunking import Chunk

METRICS = ("cosine", "dot", "euclidean")
_VECTORS_FILE = "vectors.npy"
_CHUNKS_FILE = "chunks.json"


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk and its score; with every metric a higher score is a better match."""

    chunk: Chunk
    score: float


class VectorIndex:
    """Chunks with one dense vector each, searched by comparing the query with every vector."""

    def __init__(self, chunks: Sequence[Chunk], vectors: NDArray[np.float32]) -> None:
        if not chunks:
            raise ValueError("An index needs at least one chunk")
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Expected one vector per chunk: {len(chunks)} chunks, vectors of shape {vectors.shape}"
            )
        self.chunks = tuple(chunks)
        self.vectors = vectors

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: NDArray[np.float32], top_k: int, metric: str) -> list[Hit]:
        """Return at most top_k chunks, best first.

        The euclidean score is the distance with the sign changed, so that higher is
        better as with the other metrics.
        """
        if metric not in METRICS:
            raise ValueError(f"Unknown similarity metric {metric!r}; expected one of {METRICS}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if query.shape != self.vectors.shape[1:]:
            size = self.vectors.shape[1]
            raise ValueError(f"Query of shape {query.shape} does not match vectors of size {size}")
        if metric == "euclidean":
            scores = -np.linalg.norm(self.vectors - query, axis=1)
        else:
            scores = self.vectors @ query
            if metric == "cosine":
                scores = scores / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query))
        best = np.argsort(-scores, kind="stable")[:top_k]
        return [Hit(self.chunks[position], float(scores[position])) for position in best]

    def save(self, directory: Path) -> None:
        """Write the vectors and the chunks into a directory."""
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / _VECTORS_FILE, self.vectors)
        records = [asdict(chunk) for chunk in self.chunks]
        (directory / _CHUNKS_FILE).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> VectorIndex:
        """Read an index written by save."""
        records = json.loads((directory / _CHUNKS_FILE).read_text(encoding="utf-8"))
        return cls([Chunk(**record) for record in records], np.load(directory / _VECTORS_FILE))
