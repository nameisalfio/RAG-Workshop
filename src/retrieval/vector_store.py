"""Vector database: a Qdrant collection of chunks, searched by meaning, by terms or by both."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient, models

from src.config import RetrievalConfig, VectorStoreConfig
from src.ingestion.chunking import Chunk
from src.retrieval.embedding import EmbeddingBatch

_DISTANCES = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclidean": models.Distance.EUCLID,
}


# Names of the two vectors of a point, and the ways to search them.
_DENSE = "dense"
_SPARSE = "sparse"
# Payload field with the family of the document. The search does not use it: it is there to
# filter and to colour the points in the dashboard of the vector database.
_FOLDER = "folder"
MODES = ("dense", "sparse", "hybrid")


def _folder_of(chunk: Chunk) -> str:
    """The folder of the raw documents a chunk comes from, which is its family: contracts, manuals, ..."""
    parts = PurePosixPath(chunk.source).parts
    return parts[0] if len(parts) > 1 else ""


def _chunk_of(payload: Mapping[str, Any] | None) -> Chunk:
    """The chunk stored in a point; the payload also holds fields that are not part of the chunk."""
    stored = payload or {}
    return Chunk(**{field.name: stored[field.name] for field in fields(Chunk)})


def _sparse_vector(weights: Mapping[int, float]) -> models.SparseVector:
    """Token weights in the form Qdrant stores and searches."""
    return models.SparseVector(indices=list(weights), values=[float(value) for value in weights.values()])


@dataclass(frozen=True)
class Hit:
    """A chunk returned by a search, with its score; a higher score is a better match."""

    chunk: Chunk
    score: float


class EmptyCollectionError(RuntimeError):
    """Raised when a search is attempted before the corpus has been ingested."""


class StoreUnavailableError(RuntimeError):
    """Raised when the Qdrant server does not answer."""


class QdrantStore:
    """One Qdrant collection: every point holds a dense vector, a sparse vector and the chunk.

    The dense vector finds chunks close in meaning, the sparse one chunks with the same
    terms; the hybrid mode runs both searches and fuses the two rankings.
    ``server`` is the HTTP client of the Qdrant server; it is None for a store that
    lives in the memory of the process, which cannot write or read snapshots.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        retrieval: RetrievalConfig,
        server: httpx.Client | None = None,
    ) -> None:
        if retrieval.similarity_metric not in _DISTANCES:
            raise ValueError(
                f"Unknown similarity metric {retrieval.similarity_metric!r}; "
                f"expected one of {tuple(_DISTANCES)}"
            )
        if retrieval.mode not in MODES:
            raise ValueError(f"Unknown retrieval mode {retrieval.mode!r}; expected one of {MODES}")
        if retrieval.candidates <= 0 or retrieval.rrf_k <= 0:
            raise ValueError("retrieval.candidates and retrieval.rrf_k must be positive")
        self._client = client
        self._collection = collection
        self._retrieval = retrieval
        self._metric = retrieval.similarity_metric
        self._server = server

    @classmethod
    def connect(cls, config: VectorStoreConfig, retrieval: RetrievalConfig) -> QdrantStore:
        """Open the configured collection on a Qdrant server."""
        server = httpx.Client(base_url=config.url, timeout=config.timeout_s)
        try:
            server.get("/readyz").raise_for_status()
        except httpx.HTTPError as error:
            # The cause is in the message: its long network traceback would only hide it.
            raise StoreUnavailableError(
                f"The vector database does not answer at {config.url} ({error}). "
                "Start it with: docker compose up -d qdrant"
            ) from None
        client = QdrantClient(url=config.url, timeout=int(config.timeout_s))
        return cls(client, config.collection, retrieval, server)

    @property
    def mode(self) -> str:
        """Which vectors a search uses: dense, sparse or hybrid."""
        return self._retrieval.mode

    def fingerprint(self) -> str:
        """A short signature of what the collection holds: its name, its size and two of its chunks.

        Results saved for a search are valid only for the content they were found in. The first
        and the last chunk change whenever the documents or the way they are cut change.
        """
        count = self.count()
        if count == 0:
            return f"{self._collection}:0"
        sample = self._client.retrieve(self._collection, ids=[0, count - 1], with_payload=True)
        chunks = sorted(
            (asdict(_chunk_of(point.payload)) for point in sample), key=lambda chunk: chunk["source"]
        )
        digest = hashlib.sha256(json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        return f"{self._collection}:{count}:{digest.hexdigest()[:16]}"

    def count(self) -> int:
        """Number of chunks in the collection; zero when the collection does not exist yet."""
        if not self._client.collection_exists(self._collection):
            return 0
        return self._client.count(self._collection, exact=True).count

    def layout_problem(self) -> str | None:
        """Why the existing collection cannot be searched as configured; None when it can."""
        params = self._client.get_collection(self._collection).config.params
        dense = params.vectors.get(_DENSE) if isinstance(params.vectors, dict) else None
        if dense is None or _SPARSE not in (params.sparse_vectors or {}):
            return "it does not hold a dense and a sparse vector for every chunk"
        if dense.distance != _DISTANCES[self._metric]:
            return f"it was not built for the {self._metric} metric"
        return None

    def recreate(self, dimension: int) -> None:
        """Drop the collection with everything in it and create it again, empty."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            self._collection,
            vectors_config={_DENSE: models.VectorParams(size=dimension, distance=_DISTANCES[self._metric])},
            sparse_vectors_config={_SPARSE: models.SparseVectorParams()},
        )

    def add(self, chunks: Sequence[Chunk], embedded: EmbeddingBatch, first_id: int) -> None:
        """Store chunks with their vectors; ids are consecutive numbers starting at first_id."""
        if len(embedded) != len(chunks):
            raise ValueError(
                f"Expected one embedding per chunk: {len(chunks)} chunks, {len(embedded)} embeddings"
            )
        points = [
            models.PointStruct(
                id=first_id + offset,
                vector={_DENSE: dense.tolist(), _SPARSE: _sparse_vector(sparse)},
                payload={**asdict(chunk), _FOLDER: _folder_of(chunk)},
            )
            for offset, (chunk, dense, sparse) in enumerate(
                zip(chunks, embedded.dense, embedded.sparse, strict=True)
            )
        ]
        self._client.upsert(self._collection, points=points, wait=True)

    def search(self, dense: NDArray[np.float32], sparse: Mapping[int, float], limit: int) -> list[Hit]:
        """Return at most ``limit`` chunks for an embedded question, best first.

        A higher score is a better match in every mode. In the hybrid mode the score is
        the one of Reciprocal Rank Fusion, which depends on positions only: the scores of
        the two searches live on different scales and cannot be added.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if self.count() == 0:
            raise EmptyCollectionError(
                f"Collection {self._collection!r} holds no chunks: run the ingestion before asking questions"
            )
        problem = self.layout_problem()
        if problem is not None:
            raise ValueError(
                f"Collection {self._collection!r} cannot be searched: {problem}. "
                "Build it again with: python -m src.ingestion.main ingestion.force=true"
            )
        mode = self._retrieval.mode
        if mode == "dense":
            found = self._client.query_points(
                self._collection, query=dense.tolist(), using=_DENSE, limit=limit, with_payload=True
            )
        elif mode == "sparse":
            found = self._client.query_points(
                self._collection, query=_sparse_vector(sparse), using=_SPARSE, limit=limit, with_payload=True
            )
        else:
            candidates = max(self._retrieval.candidates, limit)
            found = self._client.query_points(
                self._collection,
                prefetch=[
                    models.Prefetch(query=dense.tolist(), using=_DENSE, limit=candidates),
                    models.Prefetch(query=_sparse_vector(sparse), using=_SPARSE, limit=candidates),
                ],
                query=models.RrfQuery(rrf=models.Rrf(k=self._retrieval.rrf_k)),
                limit=limit,
                with_payload=True,
            )
        # Qdrant reports a distance for the euclidean metric: the sign is changed so that higher is better.
        sign = -1.0 if mode == "dense" and self._metric == "euclidean" else 1.0
        return [Hit(_chunk_of(point.payload), sign * point.score) for point in found.points]

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
