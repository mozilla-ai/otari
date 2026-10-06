"""Reuse a provider's client across requests.

any-llm's module-level calls build a provider, and with it a new SDK and HTTP
client, on every call: an SSL context loaded on the event loop, then a new
connection and TLS handshake to the provider for each request. These calls take
the same arguments and keep one provider per provider, credential and client
options instead, so its connection pool is reused.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator, Hashable
from typing import Any
from weakref import WeakKeyDictionary

from any_llm import AnyLLM, LLMProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.types.messages import MessageResponse, MessageStreamEvent, ParsedBetaMessage, ParsedMessage
from any_llm.types.responses import ParsedResponse, Response, ResponseStreamEvent
from openresponses_types import ResponseResource

# Per event loop, since an SDK client's connections belong to the loop that
# opened them. Bounded, so rotating credentials cannot grow it without limit; an
# evicted provider is closed by the garbage collector, as every one was before.
_MAX_PROVIDERS = 256
_providers: WeakKeyDictionary[asyncio.AbstractEventLoop, OrderedDict[Hashable, AnyLLM]] = WeakKeyDictionary()


class _Uncacheable(Exception):
    """A client option that is an object rather than a value, so it cannot key the cache."""


def _freeze(value: Any) -> Hashable:
    """A hashable stand-in for a client option made of plain values.

    An object (a prebuilt HTTP client, a credentials object) raises instead: it may
    be built per request, and one keyed by identity could match a later object
    given the same id, handing a request another's client.
    """
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise _Uncacheable


def provider_for(
    provider: str | LLMProvider,
    api_key: str | None,
    api_base: str | None,
    client_args: dict[str, Any] | None,
) -> AnyLLM:
    """Return the provider for these credentials and options, building it on first use.

    Options holding an object rather than plain values build a provider every
    call, as any-llm's own calls do.
    """
    try:
        key = (str(provider), api_key, api_base, _freeze(client_args or {}))
    except _Uncacheable:
        return AnyLLM.create(provider, api_key=api_key, api_base=api_base, **client_args or {})
    cache = _providers.setdefault(asyncio.get_running_loop(), OrderedDict())
    llm = cache.get(key)
    if llm is not None:
        cache.move_to_end(key)
        return llm
    llm = AnyLLM.create(provider, api_key=api_key, api_base=api_base, **client_args or {})
    cache[key] = llm
    if len(cache) > _MAX_PROVIDERS:
        cache.popitem(last=False)
    return llm


def reset_provider_clients() -> None:
    """Forget every cached provider (tests)."""
    _providers.clear()


def _split(model: str, provider: str | LLMProvider | None) -> tuple[str | LLMProvider, str]:
    if provider is None:
        return AnyLLM.split_model_provider(model)
    return AnyLLM.resolve_provider_key(provider), model


async def acompletion(
    model: str,
    *,
    provider: str | LLMProvider | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
    """``any_llm.acompletion`` on a reused provider."""
    provider_key, model_id = _split(model, provider)
    llm = provider_for(provider_key, api_key, api_base, client_args)
    result: ChatCompletion | AsyncIterator[ChatCompletionChunk] = await llm.acompletion(model=model_id, **kwargs)
    return result


async def aresponses(
    model: str,
    *args: Any,
    provider: str | LLMProvider | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ResponseResource | Response | ParsedResponse[Any] | AsyncIterator[ResponseStreamEvent]:
    """``any_llm.aresponses`` on a reused provider."""
    if args:
        kwargs["input_data"] = args[0]
    provider_key, model_id = _split(model, provider)
    llm = provider_for(provider_key, api_key, api_base, client_args)
    result: ResponseResource | Response | ParsedResponse[Any] | AsyncIterator[ResponseStreamEvent]
    result = await llm.aresponses(model=model_id, **kwargs)
    return result


async def amessages(
    model: str,
    *,
    provider: str | LLMProvider | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> MessageResponse | ParsedMessage[Any] | ParsedBetaMessage[Any] | AsyncIterator[MessageStreamEvent]:
    """``any_llm.amessages`` on a reused provider."""
    provider_key, model_id = _split(model, provider)
    return await provider_for(provider_key, api_key, api_base, client_args).amessages(model=model_id, **kwargs)
