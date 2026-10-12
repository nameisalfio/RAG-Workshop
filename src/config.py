"""Settings of the pipeline: the Hydra schema of conf/config.yaml and the secrets of the environment."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import MISSING, OmegaConf

SCHEMA_NAME = "base_config"
CONFIG_MODULE = "src.conf"
CONFIG_NAME = "config"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathsConfig:
    """Input and output locations; relative paths start at the project root."""

    raw_documents_dir: str = MISSING
    artifacts_dir: str = MISSING
    parsed_dir: str = MISSING
    golden_set: str = MISSING
    evaluation_results: str = MISSING
    env_file: str = MISSING


@dataclass
class ParsingConfig:
    """Parsers used for each file type and options of the text-based readers."""

    pdf_backend: str = MISSING
    generic_backend: str = MISSING
    text_encoding: str = MISSING
    csv_delimiters: str = MISSING


@dataclass
class ChunkingConfig:
    """How parsed documents are cut into chunks; sizes are in tokens of the embedding model."""

    strategy: str = MISSING
    max_tokens: int = MISSING
    min_tokens: int = MISSING
    overlap_tokens: int = MISSING


@dataclass
class LlmConfig:
    """Generation model reached through an OpenAI-compatible API."""

    provider: str = MISSING
    base_url: str = MISSING
    model: str = MISSING
    attempts: int = MISSING
    retry_pause_s: float = MISSING
    fallback_model: str = MISSING
    api_key_env: str = MISSING
    temperature: float = MISSING
    max_tokens: int = MISSING
    timeout_s: float = MISSING


@dataclass
class EmbeddingConfig:
    """Embedding model and its inference parameters."""

    model_name: str = MISSING
    device: str = MISSING
    batch_size: int = MISSING
    max_length: int = MISSING
    normalize: bool = MISSING


@dataclass
class RetrievalConfig:
    """How the vector store is searched: which vectors, how many candidates, how they are fused."""

    mode: str = MISSING
    similarity_metric: str = MISSING
    candidates: int = MISSING
    rrf_k: int = MISSING


@dataclass
class RerankingConfig:
    """The cross-encoder that orders the candidates again, and how many chunks reach the prompt."""

    enabled: bool = MISSING
    model_name: str = MISSING
    device: str = MISSING
    batch_size: int = MISSING
    max_length: int = MISSING
    top_k: int = MISSING


@dataclass
class VectorStoreConfig:
    """Vector database that holds the chunks and their vectors."""

    url: str = MISSING
    collection: str = MISSING
    timeout_s: float = MISSING
    snapshot_file: str = MISSING


@dataclass
class IngestionConfig:
    """How the corpus is loaded into the vector store."""

    batch_size: int = MISSING
    force: bool = MISSING


@dataclass
class RagConfig:
    """Instructions that frame every answer and how much of the conversation is used."""

    system_prompt: str = MISSING
    retrieval_history_turns: int = MISSING
    prompt_history_turns: int = MISSING


@dataclass
class EvaluationConfig:
    """How answers are compared with the golden set and how a recorded run behaves."""

    judge: bool = MISSING
    attempts: int = MISSING
    retry_pause_s: float = MISSING
    fresh: bool = MISSING
    abstention_markers: list[str] = MISSING


@dataclass
class CacheConfig:
    """What is saved on disk so that a question asked before is answered at once."""

    dir: str = MISSING
    retrieval: bool = MISSING
    answers: bool = MISSING


@dataclass
class ApiConfig:
    """Limits enforced by the web API."""

    max_question_chars: int = MISSING


@dataclass
class RootConfig:
    """Root configuration of the demo."""

    seed: int = MISSING
    paths: PathsConfig = MISSING
    parsing: ParsingConfig = MISSING
    chunking: ChunkingConfig = MISSING
    llm: LlmConfig = MISSING
    embedding: EmbeddingConfig = MISSING
    retrieval: RetrievalConfig = MISSING
    reranking: RerankingConfig = MISSING
    vector_store: VectorStoreConfig = MISSING
    ingestion: IngestionConfig = MISSING
    rag: RagConfig = MISSING
    evaluation: EvaluationConfig = MISSING
    cache: CacheConfig = MISSING
    api: ApiConfig = MISSING


def project_path(path: str) -> Path:
    """Resolve a configured path: absolute paths are kept, relative ones start at the project root."""
    return Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path


def register_schema() -> None:
    """Register the schema so that Hydra validates conf/config.yaml and every override against it."""
    ConfigStore.instance().store(name=SCHEMA_NAME, node=RootConfig)


def load_config(overrides: Sequence[str] = ()) -> RootConfig:
    """Compose the packaged configuration outside the command line, for notebooks and tests.

    It resets Hydra's global state, so it must not be called from inside a running Hydra application.
    """
    GlobalHydra.instance().clear()
    register_schema()
    with initialize_config_module(config_module=CONFIG_MODULE, version_base=None):
        composed = compose(config_name=CONFIG_NAME, overrides=list(overrides))
    return cast(RootConfig, OmegaConf.to_object(composed))


class MissingEnvironmentVariableError(RuntimeError):
    """Raised when a required environment variable is unset or empty."""


def load_env_file(env_file: Path) -> None:
    """Load variables from an env file if it exists; variables already set in the environment win."""
    if env_file.is_file():
        load_dotenv(env_file, override=False)


def require_env(name: str) -> str:
    """Return the value of an environment variable; an unset or empty variable is an error."""
    value = os.environ.get(name, "")
    if not value.strip():
        raise MissingEnvironmentVariableError(
            f"Environment variable {name} is not set. Copy .env.example to .env and fill in {name}."
        )
    return value
