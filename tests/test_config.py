"""The shipped configuration must load and validate against the schema."""

from __future__ import annotations

import pytest
from hydra.errors import ConfigCompositionException

from src.config import load_config
from src.parsing import PDF_BACKENDS


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
