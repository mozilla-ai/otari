from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request, Response, status

from gateway.core.config import CONVERSATION_HEADER, ROUTER_HEADER, ROUTER_TASK_HEADER
from gateway.core.env import otari_env
from gateway.models.guardrails import GuardrailConfig
from gateway.services.guardrails import GuardrailsNotReachableError, run_input_guardrails
from gateway.services.routing.decide import RoutingSignal
from gateway.services.url_safety import UnsafeURLError

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig
    from gateway.db import APIKey


GUARDRAILS_RESULT_HEADER = "X-Otari-Guardrails"
"""Response header carrying a compact JSON summary of guardrail verdicts when a
``monitor``-mode (or otherwise non-blocking) check ran."""


def resolve_user_id(
    user_id_from_request: str | None,
    api_key: APIKey | None,
    is_master_key: bool,
    *,
    master_key_error: HTTPException,
    no_api_key_error: HTTPException,
    no_user_error: HTTPException,
    forbidden_user_error: HTTPException,
    reject_mismatch: bool = True,
) -> str:
    """Resolve the effective user_id from request context.

    The resolution order is:
    1. If master key is used, the request *must* supply a user_id, and may
       name any user (the master key is trusted to act on behalf of others).
    2. For a non-master key, spend is *always* bound to the key's own user.
       The request may echo the same user_id (e.g. OpenAI's ``user`` field for
       tracking), but naming a *different* user is rejected — otherwise any key
       could charge spend to, and exhaust the budget of, another user.

    Args:
        user_id_from_request: User identifier extracted from the request body
        api_key: Authenticated API key object (None when using master key)
        is_master_key: Whether the request was authenticated with a master key
        master_key_error: Raised when master key is used but no user_id is provided
        no_api_key_error: Raised when no API key is available
        no_user_error: Raised when the API key has no associated user
        forbidden_user_error: Raised when a non-master key names a user other
            than its own (only when ``reject_mismatch`` is True)
        reject_mismatch: The deployment-wide default. When True, a non-master key
            naming a different user is rejected. When False, the mismatch is
            ignored and spend is still bound to the key's own user (the client
            ``user`` is treated as a provider-side tag only). A key whose own
            ``reject_user_mismatch`` is not None overrides this in either
            direction. Spend is bound to the key's user however this resolves;
            leniency never lets a key charge another user.

    Returns:
        Resolved user_id string

    """
    if is_master_key:
        if not user_id_from_request:
            raise master_key_error
        return user_id_from_request

    if api_key is None:
        raise no_api_key_error
    if not api_key.user_id:
        raise no_user_error
    key_user_id = str(api_key.user_id)

    # A non-master key is bound to its own user. Allow the request to echo that
    # same id; a different id is rejected (strict) or ignored (lenient) — either
    # way spend binds to key_user_id, so a key can never charge another user.
    # A key may override the deployment-wide default in either direction (NULL =
    # inherit), so one client whose ``user`` field is telemetry can be let through
    # without relaxing the check for every key, and a deployment that relaxed it
    # globally can still pin an individual key strict.
    if api_key.reject_user_mismatch is not None:
        reject_mismatch = api_key.reject_user_mismatch
    if reject_mismatch and user_id_from_request and user_id_from_request != key_user_id:
        raise forbidden_user_error

    return key_user_id


def text_from_content(content: Any) -> str:
    """Flatten a message ``content`` value to plain text for guardrail checks.

    Handles the two wire shapes shared across the chat and Anthropic-messages
    formats: a bare string, or a list of content parts where text parts look
    like ``{"type": "text", "text": "..."}``. Non-text parts (images, tool
    results, etc.) are ignored — guardrails like prompt-injection detection
    operate on the textual prompt.

    Returns:
        The flattened text, or an empty string for unrecognized shapes.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def latest_user_text(messages: Sequence[Any]) -> str:
    """Return the text of the most recent ``role == "user"`` message.

    Falls back to the last message of any role if no user message is present.
    Used to feed input-direction guardrails the prompt the model is about to
    see.

    Returns:
        The latest user message's text, or an empty string if ``messages`` is
        empty.
    """
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return text_from_content(message.get("content"))
    if messages and isinstance(messages[-1], dict):
        return text_from_content(messages[-1].get("content"))
    return ""


_ROUTER_HEADER_OFF = frozenset({"off", "false", "0", "no", "none", "disabled"})
_ROUTER_HEADER_ON = frozenset({"on", "true", "1", "yes", "auto", "default"})


def routing_signal_from_messages(messages: Sequence[Any], raw_request: Request, *, has_tools: bool) -> RoutingSignal:
    """Build the router's view of a chat-shaped request.

    Flattens the prompt the same way guardrails do and reads the three routing
    headers. Called on every request through the endpoint, whether or not the
    model names a policy with a router, so it stays cheap: three header lookups
    and one pass over the messages.
    """
    return RoutingSignal(
        task_signal=latest_user_text(messages),
        trace_signal=first_user_text(messages),
        trace_anchor=conversation_opening_text(messages),
        conversation_id=_header_value(raw_request, CONVERSATION_HEADER),
        task_id=_header_value(raw_request, ROUTER_TASK_HEADER),
        has_tools=has_tools,
        # A conversation with an assistant turn is one whose routing decision has
        # already been made, so trace-sticky reuse applies rather than a fresh
        # decision.
        is_continuation=any(isinstance(message, dict) and message.get("role") == "assistant" for message in messages),
        opted_out=routing_opted_out(raw_request),
    )


def routing_signal_from_text(text: str, raw_request: Request, *, has_tools: bool) -> RoutingSignal:
    """Build the router's view of a request with no message list (the responses API).

    One text blob serves as all three signals: there is no turn structure to draw
    a conversation opening from, so a routed responses request re-decides per call
    unless the client sends a conversation id.
    """
    return RoutingSignal(
        task_signal=text,
        trace_signal=text,
        trace_anchor=text,
        conversation_id=_header_value(raw_request, CONVERSATION_HEADER),
        task_id=_header_value(raw_request, ROUTER_TASK_HEADER),
        has_tools=has_tools,
        opted_out=routing_opted_out(raw_request),
    )


def _header_value(raw_request: Request, header: str) -> str | None:
    """A trimmed header value, or ``None`` when absent or blank.

    Blank is treated as absent so a client that always sends the header with an
    empty value gets the default behavior rather than an empty-string identity.
    """
    raw = raw_request.headers.get(header)
    if raw is None:
        return None
    return raw.strip() or None


def routing_opted_out(raw_request: Request) -> bool:
    """Whether the caller asked to skip routing for this request (``Otari-Router``).

    Absent or an "on" value means "use the policy as written". An "off" value
    serves the policy's default target without consulting its router, which is the
    escape hatch for a request the caller knows is hard. Anything else is a 400:
    silently ignoring an unrecognized value would leave a client believing it had
    opted out.
    """
    raw = raw_request.headers.get(ROUTER_HEADER)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _ROUTER_HEADER_OFF:
        return True
    if value in _ROUTER_HEADER_ON:
        return False
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid {ROUTER_HEADER} header '{raw}': expected 'on' or 'off'.",
    )


def first_user_text(messages: Sequence[Any]) -> str:
    """Return the text of the *first* ``role == "user"`` message.

    The stable task signal for a conversation: it does not change as turns are
    appended, so trace-sticky routing embeds the same thing on every turn and a
    router that has forgotten its decision reproduces it rather than drifting.
    Falls back to :func:`latest_user_text` when no user message is present.
    """
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            text = text_from_content(message.get("content"))
            if text:
                return text
    return latest_user_text(messages)


def conversation_opening_text(messages: Sequence[Any]) -> str:
    """Every turn before the first assistant reply, flattened.

    Used to identify a conversation when the client sends no
    ``Otari-Conversation-Id``. Richer than the first user turn alone, so it
    separates conversations that share an opening question but differ in their
    system preamble. Two conversations whose entire opening is identical still
    collapse to one identity; only a client-supplied id can tell those apart.
    """
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            break
        text = text_from_content(message.get("content"))
        if text:
            parts.append(text)
    if parts:
        return "\n".join(parts)
    return first_user_text(messages)


async def apply_input_guardrails(
    guardrails: list[GuardrailConfig] | None,
    input_text: str,
    *,
    response: Response,
    config: GatewayConfig | None = None,
) -> None:
    """Enforce the input guardrails for a request before the provider call.

    ``guardrails`` is the effective list: the caller's own, merged with any a
    routing policy mandates (see
    :func:`gateway.api.routes._pipeline.merge_policy_guardrails`, which is where
    the merge happens so every completion endpoint enforces a mandate alike).

    No-op when ``guardrails`` is empty/None (zero overhead for the common
    case). On a ``block``-mode flag, raises ``403`` and the provider is never
    called. On a non-blocking flag (``monitor`` mode), attaches a compact
    summary to the :data:`GUARDRAILS_RESULT_HEADER` response header and lets
    the request proceed.

    Service-failure handling depends on ``mode`` and ``on_unavailable`` (see
    :func:`gateway.services.guardrails.run_input_guardrails`): a ``block``
    guardrail that can't be evaluated fails closed (``502``) unless it sets
    ``on_unavailable="monitor"``; a ``monitor`` guardrail fails open (logged,
    request proceeds).

    Note:
        The header is set on the injected ``response``, so it reaches
        non-streaming responses. For streamed responses (where the route
        returns its own ``StreamingResponse``) the ``monitor`` annotation is
        not currently propagated; ``block`` still applies (it raises before any
        bytes are streamed).

    Raises:
        HTTPException: ``400`` when a guardrail's ``url`` override fails the
            SSRF/scheme safety check; ``403`` when a ``block`` guardrail flags
            the input; ``502`` when a ``block`` guardrail that fails closed can't
            be evaluated.
    """
    if not guardrails:
        return

    # Effective guardrails URL: dashboard override / config / env, falling back to
    # the env var when no config is threaded in (e.g. unit tests). A dashboard
    # override mutates config, so it hot-applies on the next request.
    default_url = (config.guardrails_url if config is not None else None) or otari_env("GUARDRAILS_URL") or None
    try:
        verdict = await run_input_guardrails(guardrails, input_text, default_url=default_url)
    except UnsafeURLError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GuardrailsNotReachableError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    if verdict.blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": "Request blocked by guardrail policy.",
                "code": "guardrail_violation",
                "guardrails": [
                    {
                        "profile": r.profile,
                        "explanation": r.explanation,
                        "score": r.score,
                    }
                    for r in verdict.flagged
                    if r.mode == "block"
                ],
            },
        )

    if verdict.results:
        # Non-blocking: surface the verdict for observability (monitor mode, or
        # a passing block-mode check). Header value is kept compact and free of
        # the freeform `explanation` to avoid oversized / non-ASCII headers.
        summary = [
            {"profile": r.profile, "mode": r.mode, "valid": r.valid, "score": r.score}
            for r in verdict.results
        ]
        response.headers[GUARDRAILS_RESULT_HEADER] = json.dumps(summary, separators=(",", ":"))
