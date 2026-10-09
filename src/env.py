"""Access to secrets held in the environment or in a local .env file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


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
