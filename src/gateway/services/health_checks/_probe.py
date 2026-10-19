"""One tiny completion to one offering, sent with the deployment's own credentials.

A side call like ``services/vision.py``: it goes straight to the provider, so it
writes no usage row, holds no budget and is counted by no rate limit.
"""

import asyncio
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from any_llm.exceptions import ModelNotFoundError

from gateway.services.providers import acompletion

PROBE_TIMEOUT_SECONDS = 60.0
PROBE_CONCURRENCY = 8
_ERROR_LIMIT = 300
_MAX_EXCEPTION_CHAIN = 5
# Provider errors sometimes quote the credential they refused.
_SECRET = re.compile(
    r"\b(?:sk|gw|tk|key)-[A-Za-z0-9_\-]{8,}"
    r"|\bAIza[0-9A-Za-z_\-]{30,}"
    r"|(?i:\bbearer\s+)\S+"
    r"|(?i:[?&](?:api[_-]?)?key=)[^&\s'\"]+"
)
# The catalog has no field for what a model is for, so the id decides where models.dev is silent.
_NOT_CHAT_ID = re.compile(r"embed|rerank|bge-|whisper|tts|dall-e|image", re.IGNORECASE)


@dataclass(frozen=True)
class ProbeTarget:
    """One offering to check: its selector, the hosted provider it names, and how to reach it."""

    selector: str
    provider: str
    reasoning: bool
    chat: bool
    implementation: str
    model: str
    call_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ProbeOutcome:
    target: ProbeTarget
    status: Literal["pass", "fail", "skip"]
    status_code: int | None
    error: str | None
    latency_ms: int


def is_chat_model(model_id: str, output_modalities: Sequence[str]) -> bool:
    """Whether a chat completion is the way to reach this model."""
    if output_modalities:
        return "text" in output_modalities and not _NOT_CHAT_ID.search(model_id)
    return not _NOT_CHAT_ID.search(model_id)


def _status_code(exc: BaseException) -> int | None:
    """The HTTP status somewhere on the exception chain, with any-llm's own 404 counted as one."""
    queue: list[BaseException] = [exc]
    seen: set[int] = set()
    while queue and len(seen) < _MAX_EXCEPTION_CHAIN:
        candidate = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if isinstance(candidate, ModelNotFoundError):
            return 404
        for code in (
            getattr(candidate, "status_code", None),
            getattr(getattr(candidate, "response", None), "status_code", None),
        ):
            if isinstance(code, int) and 400 <= code <= 599:
                return code
        queue.extend(
            wrapped
            for wrapped in (getattr(candidate, "original_exception", None), candidate.__cause__)
            if isinstance(wrapped, BaseException)
        )
    return None


def _describe(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    text = _SECRET.sub("[redacted]", text)
    return text if len(text) <= _ERROR_LIMIT else text[: _ERROR_LIMIT - 1] + "…"


async def probe(
    target: ProbeTarget, *, prompt: str, max_tokens: int, timeout: float = PROBE_TIMEOUT_SECONDS
) -> ProbeOutcome:
    """Send ``prompt`` to one offering. Any answer passes; an error or a timeout fails."""
    if not target.chat:
        return ProbeOutcome(target, "skip", None, "not a chat model", 0)
    started = time.monotonic()
    try:
        await asyncio.wait_for(
            acompletion(
                model=target.model,
                provider=target.implementation,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                **target.call_kwargs,
            ),
            timeout=timeout,
        )
    except TimeoutError:
        return ProbeOutcome(target, "fail", None, f"no answer within {timeout:g}s", _elapsed(started))
    except Exception as exc:  # noqa: BLE001 - every failure is a result, not an error of the run
        return ProbeOutcome(target, "fail", _status_code(exc), _describe(exc), _elapsed(started))
    return ProbeOutcome(target, "pass", None, None, _elapsed(started))


async def probe_all(
    targets: Sequence[ProbeTarget],
    *,
    prompt: str,
    max_tokens: int,
    reasoning_max_tokens: int,
) -> list[ProbeOutcome]:
    """Probe every target, a few at a time, so a run does not burst a provider's rate limit."""
    gate = asyncio.Semaphore(PROBE_CONCURRENCY)

    async def one(target: ProbeTarget) -> ProbeOutcome:
        async with gate:
            cap = reasoning_max_tokens if target.reasoning else max_tokens
            return await probe(target, prompt=prompt, max_tokens=cap)

    return list(await asyncio.gather(*(one(target) for target in targets)))


def _elapsed(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
