"""Vector database: the chunks and their vectors live in a Qdrant collection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import httpx
import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient, models

from src.chunking import Chunk
from src.config import VectorStoreConfig
from src.retrieval import Hit

_DISTANCES = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclidean": models.Distance.EUCLID,
}


class EmptyCollectionError(RuntimeError):
    """Raised when a search is attempted before the corpus has been ingested."""


class QdrantStore:
    """One Qdrant collection with a dense vector and the chunk as payload for every point.

    ``server`` is the HTTP client of the Qdrant server; it is None for a store that
    lives in the memory of the process, which cannot write or read snapshots.
    """

    def __init__(
        self, client: QdrantClient, collection: str, metric: str, server: httpx.Client | None = None
    ) -> None:
        if metric not in _DISTANCES:
            raise ValueError(f"Unknown similarity metric {metric!r}; expected one of {tuple(_DISTANCES)}")
        self._client = client
        self._collection = collection
        self._metric = metric
        self._server = server

    @classmethod
    def connect(cls, config: VectorStoreConfig, metric: str) -> QdrantStore:
        """Open the configured collection on a Qdrant server."""
        client = QdrantClient(url=config.url, timeout=int(config.timeout_s))
        server = httpx.Client(base_url=config.url, timeout=config.timeout_s)
        return cls(client, config.collection, metric, server)

    def count(self) -> int:
        """Number of chunks in the collection; zero when the collection does not exist yet."""
        if not self._client.collection_exists(self._collection):
            return 0
        return self._client.count(self._collection, exact=True).count

    def recreate(self, dimension: int) -> None:
        """Drop the collection with everything in it and create it again, empty."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        vectors = models.VectorParams(size=dimension, distance=_DISTANCES[self._metric])
        self._client.create_collection(self._collection, vectors_config=vectors)

    def add(self, chunks: Sequence[Chunk], vectors: NDArray[np.float32], first_id: int) -> None:
        """Store chunks with their vectors; ids are consecutive numbers starting at first_id."""
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Expected one vector per chunk: {len(chunks)} chunks, vectors of shape {vectors.shape}"
            )
        points = [
            models.PointStruct(id=first_id + offset, vector=vector.tolist(), payload=asdict(chunk))
            for offset, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self._client.upsert(self._collection, points=points, wait=True)

    def search(self, query: NDArray[np.float32], top_k: int) -> list[Hit]:
        """Return at most top_k chunks, best first; a higher score is a better match with every metric."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if self.count() == 0:
            raise EmptyCollectionError(
                f"Collection {self._collection!r} holds no chunks: run the ingestion before asking questions"
            )
        stored = self._client.get_collection(self._collection).config.params.vectors
        if not isinstance(stored, models.VectorParams) or stored.distance != _DISTANCES[self._metric]:
            raise ValueError(
                f"Collection {self._collection!r} was not built for the {self._metric} metric: "
                "ingest it again"
            )
        found = self._client.query_points(
            self._collection, query=query.tolist(), limit=top_k, with_payload=True
        )
        # Qdrant reports a distance for the euclidean metric: the sign is changed so that higher is better.
        sign = -1.0 if self._metric == "euclidean" else 1.0
        return [Hit(Chunk(**(point.payload or {})), sign * point.score) for point in found.points]

    def export_snapshot(self, path: Path) -> None:
        """Save the whole collection into one file that another Qdrant server can load."""
        server = self._require_server()
        created = self._client.create_snapshot(self._collection, wait=True)
        if created is None:
            raise RuntimeError(f"Qdrant did not create a snapshot of collection {self._collection!r}")
        location = f"/collections/{self._collection}/snapshots/{created.name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        with server.stream("GET", location) as response:
            response.raise_for_status()
            with path.open("wb") as file:
                for block in response.iter_bytes():
                    file.write(block)
        # The copy kept by the server is no longer needed once the file is on disk.
        self._client.delete_snapshot(self._collection, created.name, wait=True)

    def import_snapshot(self, path: Path) -> None:
        """Replace the collection with the content of a file written by export_snapshot."""
        server = self._require_server()
        with path.open("rb") as file:
            response = server.post(
                f"/collections/{self._collection}/snapshots/upload",
                params={"priority": "snapshot", "wait": "true"},
                files={"snapshot": (path.name, file, "application/octet-stream")},
            )
        if response.status_code != httpx.codes.OK:
            detail = " ".join(response.text.split())[:300]
            raise RuntimeError(f"Qdrant refused the snapshot {path}: HTTP {response.status_code} {detail}")

    def _require_server(self) -> httpx.Client:
        """The HTTP client of the server, needed by the operations that move snapshot files."""
        if self._server is None:
            raise RuntimeError("Snapshots need a Qdrant server: this store lives in memory")
        return self._server
