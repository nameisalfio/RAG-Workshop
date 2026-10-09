"""Embedding stage: texts in, dense and sparse representations out."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer

from src.config import EmbeddingConfig

# File of the BGE-M3 repository that holds the linear layer of the sparse head.
_SPARSE_HEAD_FILE = "sparse_linear.pt"


@dataclass(frozen=True)
class EmbeddingBatch:
    """Representations of a batch of texts, one row per text, in input order.

    ``dense`` has shape (n, dimension); ``sparse`` holds one mapping from token id
    to weight for each text and is empty for text without lexical weights.
    """

    dense: NDArray[np.float32]
    sparse: tuple[Mapping[int, float], ...]

    def __post_init__(self) -> None:
        if self.dense.ndim != 2:
            raise ValueError(f"dense must be a 2D array, got {self.dense.ndim} dimensions")
        if len(self.sparse) != self.dense.shape[0]:
            raise ValueError(
                f"dense has {self.dense.shape[0]} rows but sparse has {len(self.sparse)} entries"
            )

    def __len__(self) -> int:
        return int(self.dense.shape[0])


class Embedder(Protocol):
    """Turns texts into vectors; every implementation must return rows in input order."""

    @property
    def dimension(self) -> int:
        """Size of each dense vector."""
        ...

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed a non-empty batch of non-empty texts; inputs longer than the model limit are rejected."""
        ...


class BgeM3Embedder:
    """BGE-M3 run locally: one pass of the model gives the dense vector and the sparse weights.

    The dense vector is the hidden state of the first token; the sparse weight of a
    token is a linear layer applied to its hidden state, keeping the highest value
    when a token occurs more than once.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._device = torch.device(config.device)
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self._model = AutoModel.from_pretrained(config.model_name).to(self._device).eval()
        head = torch.nn.Linear(self.dimension, 1)
        weights = torch.load(
            hf_hub_download(config.model_name, _SPARSE_HEAD_FILE), map_location="cpu", weights_only=True
        )
        head.load_state_dict(weights)
        self._sparse_head = head.to(self._device).eval()
        self._special_ids = frozenset(self._tokenizer.all_special_ids)

    @property
    def dimension(self) -> int:
        """Size of each dense vector."""
        return int(self._model.config.hidden_size)

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed a non-empty batch of non-empty texts; a text over max_length tokens is an error."""
        if not texts:
            raise ValueError("embed needs at least one text")
        blank = [index for index, text in enumerate(texts) if not text.strip()]
        if blank:
            raise ValueError(f"Texts at positions {blank} are empty")
        dense: list[NDArray[np.float32]] = []
        sparse: list[Mapping[int, float]] = []
        for start in range(0, len(texts), self._config.batch_size):
            encoded = self._tokenizer(
                list(texts[start : start + self._config.batch_size]), padding=True, return_tensors="pt"
            )
            length = int(encoded["input_ids"].shape[1])
            if length > self._config.max_length:
                raise ValueError(
                    f"A text has {length} tokens, over the limit of {self._config.max_length}; "
                    "nothing is truncated silently"
                )
            with torch.inference_mode():
                hidden = self._model(**encoded.to(self._device)).last_hidden_state
                vectors = hidden[:, 0]
                if self._config.normalize:
                    vectors = torch.nn.functional.normalize(vectors, dim=-1)
                token_weights = torch.relu(self._sparse_head(hidden)).squeeze(-1)
            dense.append(vectors.cpu().numpy().astype(np.float32))
            for token_ids, weights in zip(encoded["input_ids"].tolist(), token_weights.tolist(), strict=True):
                sparse.append(self._lexical_weights(token_ids, weights))
        return EmbeddingBatch(np.concatenate(dense), tuple(sparse))

    def _lexical_weights(self, token_ids: Sequence[int], weights: Sequence[float]) -> dict[int, float]:
        """Weight of each distinct token of a text, without special tokens and zero weights."""
        lexical: dict[int, float] = {}
        for token_id, weight in zip(token_ids, weights, strict=True):
            if token_id not in self._special_ids and weight > lexical.get(token_id, 0.0):
                lexical[token_id] = weight
        return lexical
