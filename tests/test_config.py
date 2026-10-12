"""The shipped configuration must load and validate; a secret that is missing must fail loudly."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException

from src.config import MissingEnvironmentVariableError, load_config, load_env_file, require_env
from src.ingestion.parsing import PDF_BACKENDS


def test_default_configuration_loads() -> None:
    config = load_config()
    assert config.embedding.model_name == "BAAI/bge-m3"
    assert config.llm.api_key_env == "OPENROUTER_API_KEY"
    assert config.parsing.pdf_backend in PDF_BACKENDS


def test_override_changes_value() -> None:
    assert load_config(["embedding.batch_size=4"]).embedding.batch_size == 4


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigCompositionException):
        load_config(["embedding.unknown_option=1"])


KEY_NAME = "RAG_WORKSHOP_TEST_KEY"


def test_unset_or_blank_variable_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_NAME, raising=False)
    with pytest.raises(MissingEnvironmentVariableError, match=KEY_NAME):
        require_env(KEY_NAME)
    monkeypatch.setenv(KEY_NAME, "   ")
    with pytest.raises(MissingEnvironmentVariableError):
        require_env(KEY_NAME)


def test_variable_is_read_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"{KEY_NAME}=from-file\n", encoding="utf-8")
    monkeypatch.delenv(KEY_NAME, raising=False)
    load_env_file(env_file)
    assert require_env(KEY_NAME) == "from-file"
    monkeypatch.delenv(KEY_NAME)
