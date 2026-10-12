"""Chunking strategies must respect the size limit and keep what is needed to cite a chunk."""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from src.config import ChunkingConfig
from src.ingestion.chunking import chunk_document, count_tokens
from src.ingestion.parsing import ParsedDocument

CONTRACT = (
    "## Art. 6 - Penalties\n\nA penalty of 150 euro applies for every hour of delay.\n\n"
    "## Art. 7 - Withdrawal\n\nThe customer may withdraw with 45 days of notice.\n\n"
    "The supplier may withdraw with 180 days of notice.",
    "## Art. 8 - Notices\n\nNotices are sent by certified e-mail.",
)


@pytest.fixture
def tokenizer() -> Tokenizer:
    """A tokenizer that needs no download: one token per word or punctuation mark."""
    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def document(pages: tuple[str, ...] = CONTRACT, paginated: bool = True) -> ParsedDocument:
    """A two-page contract with three articles."""
    return ParsedDocument(Path("contracts/doc.pdf"), "test", pages, paginated)


def config(strategy: str, max_tokens: int, overlap_tokens: int = 0, min_tokens: int = 0) -> ChunkingConfig:
    """A chunking configuration."""
    return ChunkingConfig(
        strategy=strategy, max_tokens=max_tokens, min_tokens=min_tokens, overlap_tokens=overlap_tokens
    )


def test_structural_gives_one_chunk_per_article_with_section_and_page(tokenizer: Tokenizer) -> None:
    chunks = chunk_document(document(), tokenizer, config("structural", 60))
    sections = ["Art. 6 - Penalties", "Art. 7 - Withdrawal", "Art. 8 - Notices"]
    assert [chunk.section for chunk in chunks] == sections
    assert [chunk.page for chunk in chunks] == [1, 1, 2]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert "45 days" in chunks[1].text and "180 days" in chunks[1].text
    assert chunks[0].source == "contracts/doc.pdf"


def test_structural_splits_a_long_section_and_repeats_its_heading(tokenizer: Tokenizer) -> None:
    chunks = chunk_document(document(), tokenizer, config("structural", 18))
    withdrawal = [chunk for chunk in chunks if chunk.section == "Art. 7 - Withdrawal"]
    assert len(withdrawal) == 2
    assert all(chunk.text.startswith("## Art. 7 - Withdrawal") for chunk in withdrawal)
    assert "45 days" in withdrawal[0].text and "180 days" in withdrawal[1].text


def test_structural_joins_a_title_without_body_to_the_next_section(tokenizer: Tokenizer) -> None:
    pages = ("## MAINTENANCE CONTRACT\n\n## n. CTR-2024-017\n\n## between\n\nTorrevalle Acque and Beta.",)
    alone = chunk_document(document(pages), tokenizer, config("structural", 60))
    assert [chunk.text for chunk in alone][:2] == ["## MAINTENANCE CONTRACT", "## n. CTR-2024-017"]
    joined = chunk_document(document(pages), tokenizer, config("structural", 60, min_tokens=12))
    assert [chunk.section for chunk in joined] == ["MAINTENANCE CONTRACT"]
    assert "CTR-2024-017" in joined[0].text and "Torrevalle Acque and Beta." in joined[0].text


def test_fixed_ignores_structure_and_overlaps(tokenizer: Tokenizer) -> None:
    chunks = chunk_document(document(), tokenizer, config("fixed", 20, overlap_tokens=5))
    assert all(count_tokens(tokenizer, chunk.text) <= 20 for chunk in chunks)
    assert all(chunk.section == "" for chunk in chunks)
    shared = chunks[0].text[chunks[0].text.index(chunks[1].text[:10]) :]
    assert chunks[1].text.startswith(shared)
    assert count_tokens(tokenizer, shared) == 5
    # The cut falls inside the heading of Art. 7: structure is ignored.
    assert chunks[0].text.endswith("## Art")


def test_recursive_respects_the_limit_and_cuts_between_paragraphs(tokenizer: Tokenizer) -> None:
    chunks = chunk_document(document(), tokenizer, config("recursive", 30))
    assert all(count_tokens(tokenizer, chunk.text) <= 30 for chunk in chunks)
    assert " ".join(" ".join(chunk.text.split()) for chunk in chunks) == " ".join(document().text.split())
    # Paragraphs stay whole, but a heading can end up far from its body.
    assert chunks[0].text.endswith("## Art. 7 - Withdrawal")
    assert chunks[1].text.startswith("The customer may withdraw")
    assert chunks[1].page == 1 and chunks[2].page == 2


def test_recursive_falls_back_to_sentences_without_losing_characters(tokenizer: Tokenizer) -> None:
    pages = ("First sentence here. Second sentence here. Third one.",)
    chunks = chunk_document(document(pages), tokenizer, config("recursive", 5))
    assert [chunk.text for chunk in chunks] == ["First sentence here.", "Second sentence here.", "Third one."]


def test_documents_without_pages_have_no_page_number(tokenizer: Tokenizer) -> None:
    chunks = chunk_document(document(("Just a note.",), paginated=False), tokenizer, config("structural", 50))
    assert [(chunk.section, chunk.page) for chunk in chunks] == [("", None)]


def test_invalid_configuration_is_an_error(tokenizer: Tokenizer) -> None:
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_document(document(), tokenizer, config("semantic", 50))
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_document(document(), tokenizer, config("fixed", 10, overlap_tokens=10))
    with pytest.raises(ValueError, match="min_tokens"):
        chunk_document(document(), tokenizer, config("structural", 10, min_tokens=10))
