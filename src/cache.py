"""Results kept on disk, so that a question asked before costs nothing the second time."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


class DiskCache:
    """One JSON file per entry, named after a hash of what the entry depends on.

    An entry is found again only when every part of its key is the same: a change
    of the question, of the search settings or of the model gives another file.
    The files are plain JSON and can be read or deleted by hand.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def _file(self, key: tuple[str, ...]) -> Path:
        digest = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()
        return self._directory / f"{digest}.json"

    def get(self, *key: str) -> Any | None:  # noqa: ANN401 - a value is whatever JSON can hold
        """The value saved under a key, or None when nothing was saved; a damaged file is an error."""
        file = self._file(key)
        if not file.is_file():
            return None
        return json.loads(file.read_text(encoding="utf-8"))["value"]

    def put(self, value: object, *key: str) -> None:
        """Save a value under a key; the key is written next to it so that the file explains itself."""
        file = self._file(key)
        file.parent.mkdir(parents=True, exist_ok=True)
        # Written aside and renamed, so that a reader never finds half a file.
        draft = file.with_suffix(".tmp")
        draft.write_text(
            json.dumps({"key": key, "value": value}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        draft.replace(file)

    def clear(self) -> None:
        """Delete every entry."""
        if self._directory.is_dir():
            shutil.rmtree(self._directory)
