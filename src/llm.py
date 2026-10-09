"""Generate stage: ask a chat model reached through an OpenAI-compatible API."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from src.config import LlmConfig

LOGGER = logging.getLogger(__name__)

# Statuses that say the model cannot answer right now, not that the request is wrong.
_UNAVAILABLE = frozenset({404, 408, 429, 500, 502, 503, 504})


class LlmError(RuntimeError):
    """Raised when the API does not return an answer; ``status`` is the HTTP status, if any."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Reply:
    """Text written by the model and the name of the model that wrote it."""

    text: str
    model: str


class ChatClient:
    """Sends one question at a time to the configured model."""

    def __init__(self, config: LlmConfig, api_key: str, transport: httpx.BaseTransport | None = None) -> None:
        self._config = config
        self._http = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=config.timeout_s,
            transport=transport,
        )

    def ask(self, prompt: str, system: str = "", history: Sequence[tuple[str, str]] = ()) -> Reply:
        """Ask a question, optionally preceded by a system instruction and by earlier exchanges.

        ``history`` holds the earlier questions with their answers, oldest first.

        When the main model is unavailable the fallback model is tried; the switch
        is logged as a warning and the reply names the model that answered.
        """
        messages = [{"role": "system", "content": system}] if system else []
        for question, answer in history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompt})
        try:
            return self._complete(self._config.model, messages)
        except LlmError as error:
            if error.status not in _UNAVAILABLE:
                raise
            LOGGER.warning(
                "Main model unavailable, the fallback model %s answers instead. Reason: %s",
                self._config.fallback_model,
                error,
            )
        return self._complete(self._config.fallback_model, messages)

    def _complete(self, model: str, messages: list[dict[str, str]]) -> Reply:
        """Call the chat completions endpoint with one model."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        response = self._http.post("/chat/completions", json=payload)
        if response.status_code != httpx.codes.OK:
            detail = " ".join(response.text.split())[:300]
            message = f"Model {model} answered HTTP {response.status_code}: {detail}"
            raise LlmError(message, response.status_code)
        body = response.json()
        # A provider failure can come back inside a successful response.
        failure = body.get("error")
        if failure:
            code = failure.get("code")
            raise LlmError(
                f"Model {model} failed with code {code}: {failure.get('message')}",
                code if isinstance(code, int) else None,
            )
        choices = body.get("choices") or []
        text = (choices[0]["message"].get("content") or "").strip() if choices else ""
        if not text:
            raise LlmError(f"Model {model} returned no text: {' '.join(str(body).split())[:300]}")
        if choices[0].get("finish_reason") == "length":
            limit = self._config.max_tokens
            LOGGER.warning("Model %s stopped at the limit of %d tokens: the answer is cut", model, limit)
        return Reply(text, str(body.get("model", model)))
