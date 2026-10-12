"""Re-ranking stage: a cross-encoder reads the question together with each candidate chunk."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from huggingface_hub.utils.tqdm import disable_progress_bars
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as transformers_logging

from src.config import RerankingConfig


class Reranker(Protocol):
    """Scores how well each text answers a question; a higher score is a better match."""

    @property
    def name(self) -> str:
        """Name of the model, shown with the results it produced."""
        ...

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        """One score per text, in input order."""
        ...


class CrossEncoderReranker:
    """A cross-encoder run locally.

    The embedding model encodes question and chunk separately, so the two never read
    each other. Here they go through the model as one input: every token of the question
    attends to every token of the chunk. It is more precise and far slower, which is why
    it only sees the few candidates that the search has already selected.
    """

    def __init__(self, config: RerankingConfig) -> None:
        if config.batch_size <= 0:
            raise ValueError(f"reranking.batch_size must be positive, got {config.batch_size}")
        # Loading prints progress bars that only clutter the output of a command or of a server.
        disable_progress_bars()
        transformers_logging.disable_progress_bar()  # type: ignore[no-untyped-call]
        self._config = config
        self._device = torch.device(config.device)
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
        self._model = model.to(self._device).eval()

    @property
    def name(self) -> str:
        """Name of the model, shown with the results it produced."""
        return self._config.model_name

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        """Relevance of each text to the question, between 0 and 1, in input order."""
        if not question.strip():
            raise ValueError("The question is empty")
        if not texts:
            raise ValueError("score needs at least one text")
        # A batch is padded to its longest text: texts of similar length go together,
        # so that a short chunk does not pay for a long one.
        by_length = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        scores = [0.0] * len(texts)
        for start in range(0, len(texts), self._config.batch_size):
            positions = by_length[start : start + self._config.batch_size]
            batch = [texts[index] for index in positions]
            encoded = self._tokenizer([question] * len(batch), batch, padding=True, return_tensors="pt")
            length = int(encoded["input_ids"].shape[1])
            if length > self._config.max_length:
                raise ValueError(
                    f"A question with its chunk has {length} tokens, over the limit of "
                    f"{self._config.max_length}; nothing is truncated silently"
                )
            with torch.inference_mode():
                logits = self._model(**encoded.to(self._device)).logits.squeeze(-1)
            for index, score in zip(positions, torch.sigmoid(logits).tolist(), strict=True):
                scores[index] = score
        return scores
