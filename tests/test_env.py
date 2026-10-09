"""Secrets must come from the environment or the env file, and a missing one must fail loudly."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.env import MissingEnvironmentVariableError, load_env_file, require_env

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
