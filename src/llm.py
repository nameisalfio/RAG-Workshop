"""Generate stage: ask a chat model reached through an OpenAI-compatible API."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

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


class AllowanceExhaustedError(LlmError):
    """Raised when the allowance of requests is used up and resets too late for a retry to help."""


@dataclass(frozen=True)
class Reply:
    """Text written by the model and the name of the model that wrote it."""

    text: str
    model: str


class ChatClient:
    """Sends one question at a time to the configured model."""

    def __init__(
        self,
        config: LlmConfig,
        api_key: str,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if config.attempts <= 0:
            raise ValueError(f"llm.attempts must be positive, got {config.attempts}")
        self._config = config
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=config.timeout_s,
            transport=transport,
        )

    @property
    def model(self) -> str:
        """Name of the model that is asked first."""
        return self._config.model

    def ask(self, prompt: str, system: str = "", history: Sequence[tuple[str, str]] = ()) -> Reply:
        """Ask a question, optionally preceded by a system instruction and by earlier exchanges.

        ``history`` holds the earlier questions with their answers, oldest first.

        A main model that is unavailable is tried again after a pause. When it stays
        unavailable, the fallback model answers if one is configured, and the switch
        is logged; otherwise the last failure is raised.
        """
        messages = [{"role": "system", "content": system}] if system else []
        for question, answer in history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompt})
        attempts = self._config.attempts
        for attempt in range(1, attempts + 1):
            try:
                return self._complete(self._config.model, messages)
            except LlmError as error:
                if isinstance(error, AllowanceExhaustedError) or error.status not in _UNAVAILABLE:
                    raise
                failure = error
            if attempt < attempts:
                pause = self._config.retry_pause_s
                LOGGER.info("The model is busy, try %d of %d: waiting %.0f s", attempt, attempts, pause)
                self._sleep(pause)
        LOGGER.warning("The model did not answer in %d tries. %s", attempts, failure)
        if not self._config.fallback_model:
            raise failure
        LOGGER.warning("The fallback model %s answers instead", self._config.fallback_model)
        return self._complete(self._config.fallback_model, messages)

    def _raise_if_allowance_is_exhausted(self, response: httpx.Response) -> None:
        """Tell a used-up allowance from a short rate limit, using the reset time sent by the provider.

        A limit that resets within the time the retries would wait is left to them.
        """
        remaining = response.headers.get("x-ratelimit-remaining")
        reset = response.headers.get("x-ratelimit-reset")
        if remaining != "0" or reset is None or not reset.isdigit():
            return
        resets_at = datetime.fromtimestamp(int(reset) / 1000).astimezone()
        retry_window = self._config.attempts * self._config.retry_pause_s
        if (resets_at - datetime.now().astimezone()).total_seconds() <= retry_window:
            return
        limit = response.headers.get("x-ratelimit-limit", "the allowed")
        message = f"The allowance of {limit} model requests is used up; "
        raise AllowanceExhaustedError(
            message + f"it resets on {resets_at:%d %B at %H:%M %Z}.", response.status_code
        )

    def _complete(self, model: str, messages: list[dict[str, str]]) -> Reply:
        """Call the chat completions endpoint with one model."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        response = self._http.post("/chat/completions", json=payload)
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            self._raise_if_allowance_is_exhausted(response)
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
