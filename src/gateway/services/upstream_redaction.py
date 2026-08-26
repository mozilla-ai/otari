"""Making an upstream provider message safe to hand back.

Lives in the service layer rather than beside its first caller in
``api/routes/_platform.py`` because it has a second one that cannot reach the
API layer: ``services/model_discovery_service`` renders the same class of text
(an any-llm exception from a provider call) into a provider health or test
response. ``scripts/check_architecture.py`` forbids a service importing
``gateway.api``, and a second copy of these patterns would be a copy that goes
stale, so the patterns live here and ``_platform`` re-exports them for the
callers that already name it.
"""

import re

_REDACTION_PLACEHOLDER = "[redacted]"
# Cap on an exposed upstream message. Providers occasionally echo the offending
# request back in the error body, and a response detail is not the place for a
# few hundred KB of it.
MAX_EXPOSED_DETAIL_CHARS = 400
# Applied in order, so a URL is masked before the trailing catch-all can pick
# apart its path. Each targets a shape that carries secrets rather than meaning:
# nothing here removes text a caller needs to understand what it got wrong.
_SECRET_SHAPES: tuple[re.Pattern[str], ...] = (
    # Authorization-header credentials, whatever scheme.
    re.compile(r"(?:Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    # Provider key formats: a known prefix plus the key body. Covers OpenAI
    # (sk-, sk-proj-), Anthropic (sk-ant-), Groq (gsk_), xAI (xai-), and
    # otari's own tk_ / gw_ tokens.
    re.compile(r"\b(?:sk|pk|rk|gsk|xai|tk|gw)[-_][A-Za-z0-9._-]{8,}", re.IGNORECASE),
    # Google API keys, which carry a fixed prefix and no separator.
    re.compile(r"\bAIza[A-Za-z0-9._-]{10,}"),
    # An explicitly named credential, as it appears in a query string or an
    # echoed request body.
    re.compile(r"\b(?:api[_-]?key|access[_-]?token|token|secret|password)\s*[=:]\s*\S+", re.IGNORECASE),
    # Upstream account identifiers. A managed-model request runs on the
    # platform's own provider account, so an error naming that account would
    # tell a workspace user whose credentials served them.
    re.compile(r"\b(?:org|proj|acct|account)[-_][A-Za-z0-9]{6,}", re.IGNORECASE),
    # Any absolute URL. A self-hosted or proxied ``api_base`` is gateway
    # topology the caller has no business learning, and a credential is
    # sometimes embedded in one.
    re.compile(r"\bhttps?://\S+", re.IGNORECASE),
    # Azure OpenAI and Mistral issue prefixless 32-character API keys.
    re.compile(r"\b[A-Za-z0-9]{32}\b"),
    # Catch-all for key material with no recognizable prefix. Long unbroken
    # alphanumeric runs are tokens, not prose.
    re.compile(r"[A-Za-z0-9_-]{40,}"),
)

# Upstream APIs sometimes reflect the request that failed. Such an echo can
# contain prompt text, tool arguments, or gateway-generated context, none of
# which belongs in a client-facing error. Parameter paths such as
# ``messages.0.content`` stay useful, while field/value pairs, including common
# validation-error spellings, and JSON payloads are rejected as a whole and make
# the caller-fault classifier use its fallback.
_PAYLOAD_ECHO = re.compile(
    r"(?:[\"']?(?:messages|input|prompt|tools?|tool_calls|response|request(?:_body)?|body|content|input_value)[\"']?\s*[:=]|\b(?:messages|input|tools?|tool_calls)(?:\.\d+|\[[^]]+\])+\.(?:content|input_value)\s*:\s*(?:(?:input_)?value\s*=|[\"'{[]))",
    re.IGNORECASE,
)


def redact_upstream_message(message: str) -> str:
    """Make an upstream provider message safe to return to the caller.

    The gateway calls providers with the *operator's* credentials, so an
    upstream message is not automatically the caller's to read: it can carry the
    gateway's own key, a self-hosted ``api_base``, or other topology. This masks
    those shapes and caps the length, leaving the part that says what was
    actually wrong with the request.

    Redaction is the second line rather than the only one. Statuses where
    secrets concentrate (a rejected credential, a 5xx) never reach here at all,
    because :func:`gateway.api.routes._pipeline.classify_provider_error` keeps a
    fixed detail for them.
    """
    redacted = message.strip()
    if _PAYLOAD_ECHO.search(redacted):
        return ""
    for pattern in _SECRET_SHAPES:
        redacted = pattern.sub(_REDACTION_PLACEHOLDER, redacted)
    redacted = " ".join(redacted.split())
    if len(redacted) > MAX_EXPOSED_DETAIL_CHARS:
        redacted = redacted[: MAX_EXPOSED_DETAIL_CHARS - len("...")].rstrip() + "..."
    return redacted
