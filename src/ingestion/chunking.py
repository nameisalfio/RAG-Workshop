"""Chunking stage: cut a parsed document into pieces that can be retrieved on their own.

Three strategies share one size unit, the tokens of the embedding model: fixed
windows, recursive splitting on separators, and structural splitting on Markdown
headings with a recursive fallback for sections that are too long.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

from tokenizers import Tokenizer

from src.config import ChunkingConfig
from src.ingestion.parsing import ParsedDocument

STRATEGIES = ("fixed", "recursive", "structural")
_SEPARATORS = ("\n\n", "\n", ". ", " ")
_HEADING = re.compile(r"^#{1,6} +(.+)$", re.MULTILINE)

# A piece of text and the offset of its first character in the document.
_Span = tuple[int, str]


@dataclass(frozen=True)
class Chunk:
    """A piece of a document together with what is needed to cite it.

    ``section`` is empty for the strategies that ignore the document structure;
    ``page`` is None for formats without pages.
    """

    source: str
    index: int
    text: str
    section: str
    page: int | None


def load_tokenizer(model_name: str) -> Tokenizer:
    """Load the tokenizer of the embedding model, the unit in which chunk sizes are measured."""
    return Tokenizer.from_pretrained(model_name)


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    """Number of tokens of a text, without the special tokens the model adds around it."""
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def _trim(span: _Span) -> _Span:
    """Strip a piece and move its offset to the first character that is kept."""
    offset, text = span
    return offset + len(text) - len(text.lstrip()), text.strip()


def _split_fixed(text: str, start: int, tokenizer: Tokenizer, max_tokens: int, overlap: int) -> list[_Span]:
    """Cut every max_tokens tokens, wherever that falls; consecutive pieces share ``overlap`` tokens.

    A piece that starts inside a word can count one token more when tokenized on its own.
    """
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    spans: list[_Span] = []
    for first in range(0, len(offsets), max_tokens - overlap):
        window = offsets[first : first + max_tokens]
        begin, end = window[0][0], window[-1][1]
        spans.append((start + begin, text[begin:end]))
        if first + max_tokens >= len(offsets):
            break
    return spans


def _split_recursive(
    text: str, start: int, tokenizer: Tokenizer, max_tokens: int, separators: tuple[str, ...]
) -> list[_Span]:
    """Cut at the coarsest separator that keeps every piece within max_tokens.

    Paragraphs are tried first, then lines, sentences and words; pieces are filled
    greedily and do not overlap. ``start`` is the offset of ``text`` in its document.
    """
    if count_tokens(tokenizer, text) <= max_tokens:
        return [(start, text)]
    if not separators:
        return _split_fixed(text, start, tokenizer, max_tokens, 0)
    separator, finer = separators[0], separators[1:]
    parts = text.split(separator)
    # Every part keeps the separator that follows it, so that no character is lost.
    parts = [part + separator for part in parts[:-1]] + parts[-1:]
    groups: list[_Span] = []
    group, group_start, position = "", 0, 0
    for part in parts:
        if group and count_tokens(tokenizer, group + part) > max_tokens:
            groups.append((group_start, group))
            group, group_start = "", position
        group += part
        position += len(part)
    groups.append((group_start, group))
    spans: list[_Span] = []
    for offset, piece in groups:
        spans += _split_recursive(piece, start + offset, tokenizer, max_tokens, finer)
    return spans


def _split_structural(
    text: str, tokenizer: Tokenizer, max_tokens: int, min_tokens: int
) -> list[tuple[int, str, str]]:
    """One piece per Markdown section, as (offset, text, heading).

    A section below min_tokens, typically a title with no body, is joined to the
    section that follows. A section over max_tokens is split recursively and every
    part repeats the heading.
    """
    bounds = [0]
    for match in _HEADING.finditer(text):
        if count_tokens(tokenizer, text[bounds[-1] : match.start()]) >= min_tokens:
            bounds.append(match.start())
    bounds.append(len(text))
    pieces: list[tuple[int, str, str]] = []
    for begin, end in pairwise(bounds):
        block = text[begin:end]
        heading = _HEADING.search(block)
        section = heading.group(1).strip("* ") if heading else ""
        if count_tokens(tokenizer, block) <= max_tokens:
            pieces.append((begin, block, section))
            continue
        body_start = heading.end() if heading else 0
        prefix = f"{block[:body_start].strip()}\n\n" if heading else ""
        budget = max_tokens - count_tokens(tokenizer, prefix)
        if budget <= 0:
            raise ValueError(f"Heading {section!r} alone exceeds the limit of {max_tokens} tokens")
        body = _split_recursive(block[body_start:], begin + body_start, tokenizer, budget, _SEPARATORS)
        pieces += [(offset, prefix + part, section) for offset, part in map(_trim, body) if part]
    return pieces


def chunk_document(document: ParsedDocument, tokenizer: Tokenizer, config: ChunkingConfig) -> list[Chunk]:
    """Cut a parsed document into chunks with the configured strategy.

    ``overlap_tokens`` is used by the fixed strategy only, ``min_tokens`` by the
    structural strategy only.
    """
    if config.strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy {config.strategy!r}; expected one of {STRATEGIES}")
    if config.max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {config.max_tokens}")
    for name, value in (("overlap_tokens", config.overlap_tokens), ("min_tokens", config.min_tokens)):
        if not 0 <= value < config.max_tokens:
            raise ValueError(f"{name} must be between 0 and max_tokens - 1, got {value}")
    text = document.text
    if config.strategy == "fixed":
        spans = _split_fixed(text, 0, tokenizer, config.max_tokens, config.overlap_tokens)
        pieces = [(offset, piece, "") for offset, piece in spans]
    elif config.strategy == "recursive":
        spans = _split_recursive(text, 0, tokenizer, config.max_tokens, _SEPARATORS)
        pieces = [(offset, piece, "") for offset, piece in spans]
    else:
        pieces = _split_structural(text, tokenizer, config.max_tokens, config.min_tokens)
    source = document.path.as_posix()
    chunks: list[Chunk] = []
    for offset, piece, section in pieces:
        start, body = _trim((offset, piece))
        if body:
            chunks.append(Chunk(source, len(chunks), body, section, document.page_of(start)))
    return chunks
