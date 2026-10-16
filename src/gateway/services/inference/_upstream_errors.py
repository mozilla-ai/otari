"""What an upstream provider exception says about itself, and whether resending it can help."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Mapping
from typing import Literal, TypeVar

import httpx
from anthropic import APIConnectionError as _AnthropicAPIConnectionError
from anthropic import APITimeoutError as _AnthropicAPITimeoutError
from openai import APIConnectionError as _OpenAIAPIConnectionError
from openai import APITimeoutError as _OpenAIAPITimeoutError

from gateway.core.retry_after import bounded_retry_after
from gateway.streaming import IncompleteStreamError

# Statuses on which the billing-message probe runs. A 402 is payment required by
# definition, so it is handled directly in ``is_provider_billing_error`` without
# depending on a provider's error text. Anthropic reports "credit balance is too
# low" as a 400 ``invalid_request_error`` and OpenAI's "billing hard limit
# reached" is a 400. 429 is deliberately excluded: OpenAI's
# ``insufficient_quota`` arrives as a 429, which is already both failover-eligible
# and surfaced to the caller as a rate limit, so backing off remains a sane client
# action and reclassifying it would only change the wording.
_BILLING_MESSAGE_CANDIDATE_STATUS_CODES = {400, 422}

# Lowercase substrings that identify account-level billing exhaustion in an
# upstream provider message. Deliberately narrow: a false positive turns a
# genuinely malformed request into wasted failover attempts against every
# provider in the route, so a phrase only earns a place here if it cannot plausibly
# describe a malformed request. Best-effort against current provider phrasing; a
# reworded message degrades safely to the generic bad-request handling rather
# than misclassifying.
_BILLING_MESSAGE_PROBES: tuple[str, ...] = (
    "credit balance is too low",  # anthropic
    "purchase credits",  # anthropic
    "plans & billing",  # anthropic
    "billing hard limit",  # openai
    "insufficient balance",  # deepseek, moonshot
    "insufficient_quota",  # openai (error code, echoed in some messages)
    "exceeded your current quota",  # openai
    "insufficient credits",  # openrouter, together
)

# Exceptions that only a bug or a mismatch in the code making the call raises,
# never an upstream: the request fails the same way however often it is resent.
_LOCAL_FAULTS: tuple[type[BaseException], ...] = (LookupError, AttributeError, TypeError)

T = TypeVar("T")


UpstreamErrorKind = Literal["timeout", "conn_err"]

EMPTY_COMPLETION_DETAIL = "The provider returned a completion with no choices"


class EmptyCompletionError(Exception):
    """The provider answered a chat completion with no choices.

    A success status around nothing to return is a provider failure, not an
    answer: the attempt is recorded as failed and a routing policy moves on to
    its next candidate.
    """

    def __init__(self) -> None:
        super().__init__(EMPTY_COMPLETION_DETAIL)


def require_choices(completion: T) -> T:
    """Return ``completion``, or raise :class:`EmptyCompletionError` when it carries no choices."""
    if not getattr(completion, "choices", None):
        raise EmptyCompletionError
    return completion


def is_empty_completion(exc: BaseException) -> bool:
    """Whether ``exc`` is a provider's completion that came back with no choices."""
    return any(isinstance(link, EmptyCompletionError) for link in upstream_exception_chain(exc))


def _error_status_code(exc: BaseException) -> int | None:
    """The HTTP error status an upstream exception carries, or ``None``.

    A status on the exception itself wins over one on its attached response.

    Only a 4xx/5xx counts.
    An attached stream can report the 200 it opened with, and some SDKs put a non-HTTP code where others put the status.
    """
    response = getattr(exc, "response", None)
    for value in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(response, "status_code", None),
        _response_metadata_status(response),
    ):
        if isinstance(value, int) and 400 <= value <= 599:
            return value
    return None


def _response_metadata_status(response: object) -> int | None:
    """The HTTP status in an AWS SDK error response, or ``None``."""
    status_code = _response_metadata(response).get("HTTPStatusCode")
    return status_code if isinstance(status_code, int) else None


def _response_metadata_headers(response: object) -> Mapping[str, str] | None:
    """The HTTP headers in an AWS SDK error response, or ``None``."""
    headers = _response_metadata(response).get("HTTPHeaders")
    return headers if isinstance(headers, Mapping) else None


def _response_metadata(response: object) -> Mapping[str, object]:
    metadata = response.get("ResponseMetadata") if isinstance(response, Mapping) else None
    return metadata if isinstance(metadata, Mapping) else {}


def upstream_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its ``original_exception`` chain once each."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "original_exception", None)


def upstream_exception_shape(exc: BaseException) -> tuple[UpstreamErrorKind | None, int | None]:
    """Classify the *shape* of an upstream exception, independent of retry policy.

    Returns ``(kind, status_code)`` where ``kind`` is ``"timeout"``,
    ``"conn_err"``, or ``None``, and ``status_code`` is the HTTP status the
    exception carries, or ``None``. At most one of the two is set. ``kind`` is
    a ``Literal``, not a bare ``str``, so a typo at a ``kind == "..."``
    comparison site (e.g. in :func:`_classify_upstream_error`) is a mypy error
    instead of a silent no-op. Shared by :func:`_classify_upstream_error`
    (drives the hybrid-mode fallback decision) and
    :func:`gateway.api.routes._pipeline.classify_provider_error` (drives the
    final client-facing status/detail), so the two stay in sync.

    Recognizes, in order, walking ``.original_exception`` when the current
    exception carries no signal of its own:

    1. Stdlib/httpx timeout and network exceptions directly.
    2. The OpenAI and Anthropic SDKs' own ``APITimeoutError`` /
       ``APIConnectionError``. any-llm calls these SDKs directly (it does not
       wrap httpx itself unless ``ANY_LLM_UNIFIED_EXCEPTIONS=1``), and both
       SDKs catch ``httpx.TimeoutException`` / network errors internally and
       re-raise as their own types. Those wrapped types are not instances of
       any httpx exception and carry no ``status_code``/``response``, so a
       real provider timeout or "provider unreachable" would otherwise be
       unclassifiable. ``APITimeoutError`` subclasses ``APIConnectionError``
       in both SDKs, so the timeout check must run first. This covers the
       majority of any-llm providers, which reuse ``BaseOpenAIProvider`` or
       ``BaseAnthropicProvider``.
    3. An HTTP error status the exception carries (see :func:`_error_status_code`).
    4. A conservative duck-typed fallback, by exception class name, for the
       remaining any-llm provider SDKs that don't reuse the OpenAI/Anthropic
       base classes (e.g. cohere, mistral, groq, bedrock) and whose own
       timeout/connection exception types aren't imported here. Mirrors the
       heuristic ``any_llm.utils.exception_handler.convert_exception`` already
       uses internally. Only applies when the exception carries no status code
       at all, so it never shadows a real HTTP-status classification.
    5. If none of the above match, ``.original_exception`` (set by any-llm's
       ``AnyLLMError`` family, e.g. via ``convert_exception``) is unwrapped and
       reclassified from step 1. This keeps detection working once
       ``ANY_LLM_UNIFIED_EXCEPTIONS=1`` becomes the default: any-llm's
       ``convert_exception`` re-wraps a raw SDK timeout/connection error into
       a generic ``any_llm.exceptions.ProviderError`` (class name matches
       neither ``*TimeoutError`` nor ``*ConnectionError``, no ``status_code``),
       but always preserves the original SDK exception on
       ``original_exception``. Bounded by an ``id()``-based seen-set so a
       (pathological, self-referential) cycle terminates instead of looping.
    """
    for current in upstream_exception_chain(exc):
        if isinstance(
            current,
            (
                asyncio.TimeoutError,
                TimeoutError,
                httpx.TimeoutException,
                _OpenAIAPITimeoutError,
                _AnthropicAPITimeoutError,
            ),
        ):
            return "timeout", None
        if isinstance(current, (httpx.NetworkError, _OpenAIAPIConnectionError, _AnthropicAPIConnectionError)):
            return "conn_err", None

        status_code = _error_status_code(current)
        if status_code is not None:
            return None, status_code

        class_name = type(current).__name__
        if class_name.endswith("TimeoutError"):
            return "timeout", None
        if class_name.endswith("ConnectionError"):
            return "conn_err", None

    return None, None


def upstream_error_message(exc: BaseException) -> str:
    """Best-effort human-readable text from an upstream exception (and its wrapper).

    Concatenates the exception's ``message`` attribute and ``str()`` across its
    ``original_exception`` chain, so a substring probe still matches whether
    otari receives the raw SDK error today or a wrapped ``AnyLLMError`` after
    any-llm flips to unified exceptions.

    Used both for classification and, on a caller-fault rejection, as the text
    the caller sees. Run it through :func:`redact_upstream_message` before it
    reaches a response body.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for candidate in upstream_exception_chain(exc):
        message = getattr(candidate, "message", None)
        candidates = (message, str(candidate)) if isinstance(message, str) else (str(candidate),)
        for part in candidates:
            # ``message`` and ``str()`` are usually the same text on a raw SDK
            # error, and a wrapper usually restringifies what it wraps. Keeping
            # both would read as a stutter now that this text can reach the
            # caller, so each distinct part is kept once, in order.
            stripped = part.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                parts.append(stripped)
    return " ".join(parts)


def upstream_body_messages(exc: BaseException) -> list[str] | None:
    """The messages in the error bodies provider SDKs attached across the chain, or ``None``.

    ``None`` means no exception in the ``original_exception`` chain carries a
    ``body``. A list, even an empty one, means at least one does, and then the
    provider's message is only what is inside a body. The OpenAI, Anthropic, Groq
    and Cohere SDKs keep the parsed body there, and their ``message`` and string
    forms are framing around it: a status prefix and a Python dict repr, or
    Cohere's response headers.
    """
    bodied = [current for current in upstream_exception_chain(exc) if getattr(current, "body", None) is not None]
    if not bodied:
        return None
    return [message for current in bodied if (message := upstream_body_message(current)) is not None]


def upstream_body_message(exc: BaseException) -> str | None:
    """The human message in the error body a provider SDK attached to ``exc``, or ``None``.

    Reads ``error.message`` ahead of a top-level ``message``, because Anthropic
    nests the error in an envelope while the OpenAI SDK unwraps it first and
    Cohere never had one. A string ``error`` comes last: beside a ``message`` it
    is usually a type name rather than the explanation. A body the SDK left as text (Mistral's) is parsed when
    it is JSON and read whole otherwise, unless it is markup such as a proxy's
    HTML error page.
    """
    raw = getattr(exc, "body", None)
    if isinstance(raw, str) and not raw.strip().startswith("{"):
        text = raw.strip()
        return text if text and not text.startswith("<") else None
    body = _parsed_body(exc)
    if body is None:
        return None
    error = body.get("error")
    nested = error.get("message") if isinstance(error, Mapping) else None
    for candidate in (nested, body.get("message"), error):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _parsed_body(exc: BaseException) -> Mapping[str, object] | None:
    """The error body a provider SDK attached to ``exc`` as a mapping, parsing a JSON string, or ``None``."""
    body = getattr(exc, "body", None)
    if isinstance(body, str):
        try:
            body = json.loads(body.strip())
        except ValueError:
            return None
    return body if isinstance(body, Mapping) else None


def _body_error_field(body: Mapping[str, object], name: str) -> str | None:
    """``error.<name>`` in an error body, else a top-level ``<name>``, when it is a string."""
    error = body.get("error")
    for candidate in (error.get(name) if isinstance(error, Mapping) else None, body.get(name)):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


# Lowercase substrings of a 403's message that point at the gateway's key, its
# project or its organization rather than at the request, as OpenAI's "Project
# `proj_...` does not have access to model" does under an invalid_request_error.
_CREDENTIAL_MESSAGE_PROBES: tuple[str, ...] = ("api key", "api_key", "project", "organization", "scope")


def is_upstream_request_fault(exc: BaseException) -> bool:
    """Whether an upstream 403 refuses the request rather than the credential the gateway sent.

    OpenAI answers some request faults with a 403 whose body's error ``type`` is
    ``invalid_request_error``: logprobs from a model that does not return them, an
    embedding model sent to chat. The key is fine; the request is not, and resending it
    to another provider or with another key changes nothing about that. A 403 of any
    other type (Anthropic's ``permission_error``, OpenAI's ``request_forbidden``), with
    no body, or whose message names the key, project or organization stays a credential
    rejection.
    """
    _kind, status_code = upstream_exception_shape(exc)
    if status_code != 403:
        return False
    for current in upstream_exception_chain(exc):
        body = _parsed_body(current)
        if body is None:
            continue
        if _body_error_field(body, "type") != "invalid_request_error":
            continue
        message = (_body_error_field(body, "message") or "").lower()
        code = (_body_error_field(body, "code") or "").lower()
        return not any(probe in message or probe in code for probe in _CREDENTIAL_MESSAGE_PROBES)
    return False


def upstream_retry_after(exc: BaseException) -> str | None:
    """Return the upstream ``Retry-After`` as whole seconds capped at one day, or ``None``.

    The value comes from the first exception in the ``original_exception`` chain
    whose own headers or response headers carry a usable one.
    """
    for current in upstream_exception_chain(exc):
        response = getattr(current, "response", None)
        for headers in (
            getattr(current, "headers", None),
            getattr(response, "headers", None),
            _response_metadata_headers(response),
        ):
            # Mapping-like but not necessarily a Mapping: httpx.Headers here, a
            # plain dict where an SDK copies them out.
            get_header = getattr(headers, "get", None)
            if not callable(get_header):
                continue
            seconds = bounded_retry_after(get_header("retry-after") or get_header("Retry-After"))
            if seconds is not None:
                return seconds
    return None


def is_provider_billing_error(exc: BaseException) -> bool:
    """True when a 4xx is the provider saying "this account is out of money".

    Providers disagree on the status for account billing exhaustion. Anthropic
    returns a 400 ``invalid_request_error`` carrying "Your credit balance is too
    low to access the Anthropic API"; OpenAI has a 400 "billing hard limit
    reached"; DeepSeek uses 402 "Insufficient Balance". A 400 normally means the
    request itself is malformed, so these would otherwise be reported to the
    caller as "check the model name and parameters" and, worse, skip failover on
    the assumption that every provider would reject the same request. The
    account balance is per-provider, so the next attempt in the route can very
    plausibly succeed.

    A 402 is always a billing error: HTTP defines it as payment required, even
    when a provider SDK discards the response reason and body. The 400/422 probe
    stays deliberately narrow so error reports do not falsely label malformed
    requests as account billing exhaustion.
    """
    _kind, status_code = upstream_exception_shape(exc)
    if status_code == 402:
        return True
    if status_code not in _BILLING_MESSAGE_CANDIDATE_STATUS_CODES:
        return False
    message = upstream_error_message(exc).lower()
    return any(probe in message for probe in _BILLING_MESSAGE_PROBES)


def is_retry_futile(exc: BaseException) -> bool:
    """Whether resending the same request cannot change how ``exc`` ends it.

    True for an exhausted provider account (:func:`is_provider_billing_error`),
    which no retry within seconds tops up, and for a local fault: an exception
    the code making the call raised with no upstream signal anywhere behind it,
    such as a ``KeyError`` in a provider adapter's message conversion before the
    request left. A timeout, a dropped connection, an upstream 5xx or 429, and a
    stream the provider cut short are transient, and so is any failure this
    cannot place, so those stay retryable.
    """
    if is_provider_billing_error(exc):
        return True
    links = list(_causal_chain(exc))
    if any(_carries_upstream_signal(link) for link in links):
        return False
    return any(isinstance(link, _LOCAL_FAULTS) for link in links)


def is_upstream_connect_failure(exc: BaseException) -> bool:
    """Whether the connection to the provider was never opened, as opposed to dropped once open."""
    return any(isinstance(link, httpx.ConnectError) for link in _causal_chain(exc))


def is_upstream_stream_interrupted(exc: BaseException) -> bool:
    """Whether a provider's stream ended before it finished.

    It ended without the chunk that completes it, or the connection carrying it
    was cut, which httpx reports as a protocol error (a chunked body cut short)
    or a network error. A connection that never opened carried no stream.
    """
    if is_upstream_connect_failure(exc):
        return False
    if upstream_exception_shape(exc)[0] == "conn_err":
        return True
    return any(
        isinstance(link, (IncompleteStreamError, httpx.RemoteProtocolError, httpx.NetworkError))
        for link in _causal_chain(exc)
    )


def _causal_chain(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and every exception behind it, through any-llm's wrapper and Python's own chaining."""
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        linked = (getattr(current, "original_exception", None), current.__cause__, current.__context__)
        pending.extend(link for link in linked if isinstance(link, BaseException))


def _carries_upstream_signal(exc: BaseException) -> bool:
    """Whether ``exc`` came from talking to the provider, and so may not recur."""
    if isinstance(exc, (IncompleteStreamError, OSError, httpx.HTTPError)):
        return True
    return upstream_exception_shape(exc) != (None, None)
