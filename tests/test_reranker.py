"""The real re-ranker must put the chunk that answers the question first; it needs the model on disk."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.config import load_config
from src.retrieval.reranking import CrossEncoderReranker

pytestmark = pytest.mark.model


def test_cross_encoder_prefers_the_chunk_that_answers() -> None:
    config = load_config().reranking
    reranker = CrossEncoderReranker(config)
    texts = [
        "Il Fornitore interviene entro 4 ore dalla chiamata.",
        "Il Committente può recedere dal contratto con un preavviso di 45 giorni.",
        "E-4012: infiltrazione d'acqua nella camera olio.",
    ]
    scores = reranker.score("Con quanto preavviso si può recedere dal contratto?", texts)
    assert len(scores) == 3
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores[1] == max(scores)
    with pytest.raises(ValueError, match="at least one text"):
        reranker.score("Con quanto preavviso?", [])
    with pytest.raises(ValueError, match="nothing is truncated silently"):
        CrossEncoderReranker(replace(config, max_length=8)).score("Con quanto preavviso?", texts)
