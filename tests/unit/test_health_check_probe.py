"""One health check probe: what passes, what fails, and what the stored error may say."""

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from gateway.services.health_checks._probe import HealthCheckOffering, is_chat_model, probe

_PROBE = "gateway.services.health_checks._probe"
_TARGET = HealthCheckOffering(
    "openai:gpt-test",
    "openai",
    reasoning=False,
    chat=True,
    implementation="openai",
    model="gpt-test",
    call_kwargs={"api_key": "sk-never-stored-anywhere"},
)


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


async def _run(completion: Any, target: HealthCheckOffering = _TARGET, timeout: float = 5.0) -> Any:
    with patch(f"{_PROBE}.acompletion", new=completion):
        return await probe(target, prompt="ping", max_tokens=8, timeout=timeout)


@pytest.mark.asyncio
async def test_any_answer_passes() -> None:
    async def answer(**_kwargs: Any) -> object:
        return object()

    outcome = await _run(answer)

    assert (outcome.status, outcome.status_code, outcome.error) == ("pass", None, None)


@pytest.mark.asyncio
async def test_a_wrapped_status_is_found_on_the_chain() -> None:
    async def refuse(**_kwargs: Any) -> object:
        try:
            raise _StatusError("upstream said no", 503)
        except _StatusError as exc:
            msg = "provider failed"
            raise RuntimeError(msg) from exc

    outcome = await _run(refuse)

    assert (outcome.status, outcome.status_code) == ("fail", 503)


@pytest.mark.asyncio
async def test_a_credential_quoted_in_an_error_is_not_stored() -> None:
    async def refuse(**_kwargs: Any) -> object:
        msg = "Incorrect API key provided: sk-proj-abcdefghijklmnop"
        raise _StatusError(msg, 401)

    outcome = await _run(refuse)

    assert outcome.error == "Incorrect API key provided: [redacted]"


@pytest.mark.asyncio
async def test_a_slow_model_fails_on_the_timeout() -> None:
    async def hang(**_kwargs: Any) -> object:
        await asyncio.sleep(5)
        return object()

    outcome = await _run(hang, timeout=0.05)

    assert outcome.status == "fail"
    assert outcome.error == "no answer within 0.05s"


@pytest.mark.asyncio
async def test_a_model_that_is_not_for_chat_is_skipped_without_a_call() -> None:
    async def never(**_kwargs: Any) -> object:
        raise AssertionError

    outcome = await _run(
        never, HealthCheckOffering("openai:text-embedding-3", "openai", False, False, "openai", "text-embedding-3")
    )

    assert outcome.status == "skip"


@pytest.mark.parametrize(
    ("model_id", "output_modalities", "chat"),
    [
        ("gpt-4o", ["text"], True),
        ("gpt-4o", [], True),
        ("text-embedding-3-small", [], False),
        ("bge-m3", ["text"], False),
        ("gpt-image-1", ["image"], False),
        ("rerank-v3", [], False),
    ],
)
def test_which_models_are_asked_for_a_chat_completion(model_id: str, output_modalities: list[str], chat: bool) -> None:
    assert is_chat_model(model_id, output_modalities) is chat


@pytest.mark.parametrize(
    ("message", "stored"),
    [
        ("bad key AIzaSyA-1234567890abcdefghijklmnopqrstu", "bad key [redacted]"),
        ("refused Authorization: Bearer abc.def-ghi", "refused Authorization: [redacted]"),
        ("GET https://api.example.com/v1/models?key=s3cr3t&alt=json failed", "GET [redacted] failed"),
        ("POST /v1?api_key=s3cr3t", "POST /v1?[redacted]"),
        ("invalid key gsk_abcdefghijklmnopqrstuv1234", "invalid key [redacted]"),
        ("rejected xai-abcdefghijklmnop1234", "rejected [redacted]"),
        ("key 0123456789abcdef0123456789abcdef refused", "key [redacted] refused"),
        ("Basic dXNlcjpwYXNzd29yZA== rejected", "[redacted] rejected"),
        ("body api_key: hunter2hunter2", "body [redacted]"),
        ("otari tk_abcdefghijklmnop", "otari [redacted]"),
    ],
)
@pytest.mark.asyncio
async def test_other_credential_shapes_are_not_stored(message: str, stored: str) -> None:
    async def refuse(**_kwargs: Any) -> object:
        raise _StatusError(message, 401)

    outcome = await _run(refuse)

    assert outcome.error == stored


@pytest.mark.asyncio
async def test_an_echoed_request_is_stored_as_the_error_class() -> None:
    async def refuse(**_kwargs: Any) -> object:
        msg = 'bad request: {"messages": [{"role": "user", "content": "ping"}]}'
        raise _StatusError(msg, 400)

    outcome = await _run(refuse)

    assert outcome.error == "_StatusError"
