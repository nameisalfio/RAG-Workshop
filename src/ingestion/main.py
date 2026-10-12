"""Ingestion: bring the vector store in line with the raw documents.

The cheapest path that works is taken, in this order:

1. the collection already holds chunks, with the vectors the search needs: nothing is done;
2. a snapshot file exists: it is loaded, which takes seconds;
3. otherwise the raw documents are parsed, cut into chunks, embedded and stored,
   and a snapshot file is written for the next machine.

Run it with ``python -m src.ingestion.main``. ``python -m src.ingestion.main ingestion.force=true``
goes straight to step 3 and replaces whatever the collection holds.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.cache import DiskCache
from src.config import CONFIG_NAME, RootConfig, project_path, register_schema
from src.ingestion.chunking import Chunk, chunk_document, load_tokenizer
from src.ingestion.parsing import DocumentRouter, ParsedDocument, list_documents, load_corpus, parse_corpus
from src.retrieval.embedding import BgeM3Embedder, Embedder
from src.retrieval.vector_store import QdrantStore

LOGGER = logging.getLogger(__name__)

register_schema()


def parsed_documents(config: RootConfig) -> list[ParsedDocument]:
    """Parse the raw documents, or read the saved result when it covers every raw document."""
    raw_dir = project_path(config.paths.raw_documents_dir)
    parsed_dir = project_path(config.paths.parsed_dir)
    expected = len(list_documents(raw_dir))
    saved = len(list_documents(parsed_dir)) if parsed_dir.is_dir() and any(parsed_dir.iterdir()) else 0
    if saved == expected:
        LOGGER.info("Reading %d parsed documents from %s; delete it to parse again", saved, parsed_dir)
    else:
        LOGGER.info("Parsing %d raw documents (%d found already parsed)", expected, saved)
        parse_corpus(raw_dir, parsed_dir, DocumentRouter(config.parsing))
    return load_corpus(parsed_dir)


def fill_store(chunks: Sequence[Chunk], embedder: Embedder, store: QdrantStore, batch_size: int) -> None:
    """Empty the collection, then embed and store the chunks one batch at a time."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    store.recreate(embedder.dimension)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        store.add(batch, embedder.embed([chunk.text for chunk in batch]), first_id=start)
        LOGGER.info("Stored %d of %d chunks", start + len(batch), len(chunks))


@hydra.main(version_base=None, config_path="../conf", config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """Fill the vector store unless it is already filled."""
    config = cast(RootConfig, OmegaConf.to_object(cfg))
    store = QdrantStore.connect(config.vector_store, config.retrieval)
    snapshot = project_path(config.vector_store.snapshot_file)
    stored = store.count()
    problem = store.layout_problem() if stored else None
    if stored and problem is None and not config.ingestion.force:
        LOGGER.info("The vector store already holds %d chunks: nothing to do", stored)
        return
    if problem is not None:
        LOGGER.warning("The collection in the vector store is replaced: %s", problem)
    if snapshot.is_file() and not config.ingestion.force:
        LOGGER.info("Loading the vector store from the snapshot %s", snapshot)
        store.import_snapshot(snapshot)
        problem = store.layout_problem()
        if problem is not None:
            raise RuntimeError(
                f"The snapshot {snapshot} cannot be used: {problem}. "
                "Build the collection again with: python -m src.ingestion.main ingestion.force=true"
            )
        LOGGER.info("The vector store now holds %d chunks", store.count())
        return
    # New vectors can rank the same chunks in another order: what was saved for the old ones is dropped.
    DiskCache(project_path(config.cache.dir)).clear()
    documents = parsed_documents(config)
    tokenizer = load_tokenizer(config.embedding.model_name)
    chunks: list[Chunk] = []
    for document in documents:
        chunks += chunk_document(document, tokenizer, config.chunking)
    LOGGER.info("Embedding %d chunks from %d documents", len(chunks), len(documents))
    fill_store(chunks, BgeM3Embedder(config.embedding), store, config.ingestion.batch_size)
    store.export_snapshot(snapshot)
    LOGGER.info("Snapshot written to %s", snapshot)


if __name__ == "__main__":
    main()
