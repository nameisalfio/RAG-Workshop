"""The chat client must send the right request and surface every failure; no network is used."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import replace

import httpx
import pytest

from src.config import LlmConfig
from src.llm import AllowanceExhaustedError, ChatClient, LlmError

CONFIG = LlmConfig(
    provider="openrouter",
    base_url="https://example.test/api/v1",
    model="main-model",
    attempts=2,
    retry_pause_s=7.0,
    fallback_model="spare-model",
    api_key_env="OPENROUTER_API_KEY",
    temperature=0.0,
    max_tokens=64,
    timeout_s=5.0,
)


PAUSES: list[float] = []


def client(handler: Callable[[httpx.Request], httpx.Response], config: LlmConfig = CONFIG) -> ChatClient:
    """A client answered by a function instead of the network; its pauses are recorded, not waited."""
    PAUSES.clear()
    return ChatClient(config, "secret-key", transport=httpx.MockTransport(handler), sleep=PAUSES.append)


def answer(text: str, model: str) -> httpx.Response:
    """A successful chat completion."""
    return httpx.Response(200, json={"model": model, "choices": [{"message": {"content": text}}]})


def test_request_carries_key_model_and_messages() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer(" Rome. ", "main-model")

    reply = client(handler).ask("Capital of Italy?", system="Answer briefly.")
    assert (reply.text, reply.model) == ("Rome.", "main-model")
    assert seen[0].url == "https://example.test/api/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer secret-key"
    payload = json.loads(seen[0].content)
    assert payload["model"] == "main-model"
    assert payload["messages"] == [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Capital of Italy?"},
    ]
    assert (payload["temperature"], payload["max_tokens"]) == (0.0, 64)


def test_unavailable_model_switches_to_the_fallback_and_says_so(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        return httpx.Response(429, text="rate limited") if model == "main-model" else answer("Rome.", model)

    with caplog.at_level(logging.WARNING):
        reply = client(handler).ask("Capital of Italy?")
    assert reply.model == "spare-model"
    assert "main-model answered HTTP 429" in caplog.text and "spare-model" in caplog.text


def test_failure_inside_a_successful_response_also_switches_to_the_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        if model == "main-model":
            return httpx.Response(200, json={"error": {"code": 503, "message": "Service overloaded"}})
        return answer("Rome.", model)

    assert client(handler).ask("Capital of Italy?").model == "spare-model"
    with pytest.raises(LlmError, match="failed with code 400") as error:
        client(lambda request: httpx.Response(200, json={"error": {"code": 400, "message": "bad"}})).ask("Hi")
    assert error.value.status == 400


def test_answer_cut_at_the_token_limit_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        choice = {"message": {"content": "Rome is"}, "finish_reason": "length"}
        return httpx.Response(200, json={"model": "main-model", "choices": [choice]})

    with caplog.at_level(logging.WARNING):
        assert client(handler).ask("Capital of Italy?").text == "Rome is"
    assert "limit of 64 tokens" in caplog.text


def test_rejected_request_is_an_error_without_fallback() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(401, text="invalid key")

    with pytest.raises(LlmError, match="HTTP 401") as error:
        client(handler).ask("Capital of Italy?")
    assert error.value.status == 401
    assert calls == ["main-model"]


def test_empty_answer_and_double_failure_are_errors() -> None:
    with pytest.raises(LlmError, match="returned no text"):
        client(lambda request: answer("  ", "main-model")).ask("Capital of Italy?")
    with pytest.raises(LlmError, match="spare-model answered HTTP 503"):
        client(lambda request: httpx.Response(503, text="down")).ask("Capital of Italy?")


def test_busy_model_is_tried_again_before_anything_else() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(503, text="overloaded") if len(calls) == 1 else answer("Rome.", calls[-1])

    reply = client(handler).ask("Capital of Italy?")
    assert (reply.model, calls, PAUSES) == ("main-model", ["main-model", "main-model"], [7.0])


def test_without_a_fallback_model_the_last_failure_is_raised() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(503, text="overloaded")

    with pytest.raises(LlmError, match="main-model answered HTTP 503"):
        client(handler, replace(CONFIG, fallback_model="")).ask("Capital of Italy?")
    assert calls == ["main-model", "main-model"]
    with pytest.raises(ValueError, match="attempts must be positive"):
        client(handler, replace(CONFIG, attempts=0))


def rate_limited(resets_in_s: float) -> httpx.Response:
    """A refusal for too many requests, with the reset time the provider sends."""
    headers = {
        "X-RateLimit-Limit": "50",
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": str(int((time.time() + resets_in_s) * 1000)),
    }
    return httpx.Response(429, headers=headers, json={"error": {"message": "Rate limit exceeded"}})


def test_used_up_allowance_is_reported_at_once_without_retries_or_fallback() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        return rate_limited(resets_in_s=6 * 3600)

    with pytest.raises(AllowanceExhaustedError, match="allowance of 50 model requests is used up") as raised:
        client(handler).ask("Capital of Italy?")
    assert raised.value.status == 429
    assert calls == ["main-model"]
    assert PAUSES == []


def test_limit_that_resets_within_the_retries_is_retried() -> None:
    replies = [rate_limited(resets_in_s=5), answer("Rome.", "main-model")]

    reply = client(lambda _: replies.pop(0)).ask("Capital of Italy?")
    assert reply.text == "Rome."
    assert PAUSES == [7.0]
