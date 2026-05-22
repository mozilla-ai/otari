import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Annotated, Any, NamedTuple

import httpx
from any_llm import AnyLLM, LLMProvider, acompletion
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionUsage,
)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.auth.vertex_auth import setup_vertex_environment
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.metrics import record_cost, record_tokens
from gateway.models.entities import APIKey, UsageLog
from gateway.models.mcp import McpServerConfig
from gateway.rate_limit import RateLimitInfo, check_rate_limit
from gateway.services.budget_service import validate_user_budget
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_client import MCPClientPool
from gateway.services.mcp_loop import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
    MaxToolIterationsExceeded,
    inject_purpose_hints,
    mcp_tool_loop,
    mcp_tool_loop_stream,
)
from gateway.services.pricing_service import find_model_pricing
from gateway.services.sandbox_backend import SandboxBackend, SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchBackend, WebSearchNotReachableError
from gateway.streaming import (
    OPENAI_STREAM_FORMAT,
    StreamingAttemptFailure,
    iterate_streaming_attempts,
    streaming_generator,
)

router = APIRouter(prefix="/v1/chat", tags=["chat"])

_USAGE_NON_RETRYABLE_STATUS_CODES = {401, 404, 409, 422}


def rate_limit_headers(info: RateLimitInfo) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(info.limit),
        "X-RateLimit-Remaining": str(info.remaining),
        "X-RateLimit-Reset": str(int(info.reset)),
    }


def _is_web_search_tool_type(type_value: Any) -> bool:
    """Recognise the tool-array shapes that map to the web_search backend.

    Accepts:
      * ``"web_search"`` — gateway-native short form (matches OpenAI's
        ``{"type": "web_search"}`` server-managed tool).
      * ``"web_search_*"`` — Anthropic versioned types (e.g.
        ``"web_search_20250305"``, future ``"web_search_20260209"``).

    Matching Anthropic by prefix means new versions keep working without a
    code change; the backend's search semantics are version-agnostic.
    """
    if not isinstance(type_value, str):
        return False
    return type_value == "web_search" or type_value.startswith("web_search_")


def _is_code_execution_tool_type(type_value: Any) -> bool:
    """Recognise the tool-array shapes Anthropic and OpenAI use for code execution.

    Accepts:
      * ``"code_execution"`` — gateway-native short form
      * ``"code_interpreter"`` — OpenAI Responses/Assistants API
      * ``"code_execution_*"`` — Anthropic versioned types
        (e.g. ``"code_execution_20250825"``)

    Matching Anthropic by prefix means new versions (``code_execution_20260101``,
    etc.) keep working without a code change. Our sandbox is a generic Python
    REPL, so we don't need to track per-version semantics.
    """
    if not isinstance(type_value, str):
        return False
    return (
        type_value == "code_execution" or type_value == "code_interpreter" or type_value.startswith("code_execution_")
    )


# Gateway-internal fields the provider SDKs (any-llm, anthropic, openai, …)
# don't accept as ``acompletion`` kwargs. Strip these from the model_dump
# before forwarding to upstream — Anthropic in particular rejects unknown
# kwargs with a hard error.
_GATEWAY_INTERNAL_FIELDS = (
    "mcp_servers",
    "mcp_server_ids",
    "tools_header",
    "max_tool_iterations",
    "user",
)


def _strip_gateway_fields(
    fields: dict[str, Any],
    *,
    tools_extracted: bool = False,
    remaining_user_tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Strip gateway-internal fields from a ``request.model_dump(...)`` payload.

    Mutates ``fields`` in place and returns it for chaining. When the caller
    extracted any gateway-managed tool entry from ``tools`` (sandbox /
    web_search / future), pass ``tools_extracted=True`` and the remaining
    user-supplied tools; the original ``tools`` list is replaced (or popped
    entirely if none remain).
    """
    for k in _GATEWAY_INTERNAL_FIELDS:
        fields.pop(k, None)
    if tools_extracted:
        if remaining_user_tools:
            fields["tools"] = remaining_user_tools
        else:
            fields.pop("tools", None)
    return fields


def _resolve_sandbox_purpose_hint(sandbox_tool_entry: dict[str, Any] | None) -> str | None:
    """Resolve the per-tool ``purpose_hint`` for the sandbox.

    Priority: tool entry's ``purpose_hint`` → ``GATEWAY_SANDBOX_PURPOSE_HINT``
    env → ``None`` (SandboxBackend falls back to its built-in default).
    """
    return (
        (sandbox_tool_entry.get("purpose_hint") if sandbox_tool_entry else None)
        or os.environ.get("GATEWAY_SANDBOX_PURPOSE_HINT")
        or None
    )


def _extract_first_matching_tool(
    tools: list[dict[str, Any]] | None,
    predicate: Callable[[Any], bool],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first tool entry whose ``type`` matches ``predicate``.

    Returns ``(entry_or_None, remaining_tools_or_None)``. The extracted entry
    is thin (no function schema); the gateway-managed backend's
    ``openai_tools`` provides the full definition during tool-use-loop
    injection. Remaining user-supplied tools pass through unchanged.
    """
    if not tools:
        return None, tools
    entry: dict[str, Any] | None = None
    remaining: list[dict[str, Any]] = []
    for t in tools:
        if entry is None and isinstance(t, dict) and predicate(t.get("type")):
            entry = t
        else:
            remaining.append(t)
    return entry, (remaining or None)


def _extract_code_execution_tool(
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first code-execution-style entry out of ``tools``.

    Detects the gateway-native ``{"type": "code_execution"}`` shape plus the
    provider-native equivalents from OpenAI (``code_interpreter``) and Anthropic
    (``code_execution_20250825`` and future versions). All three map to the same
    sandbox backend so swapping ``base_url`` to the gateway keeps existing
    SDK code working unchanged.
    """
    return _extract_first_matching_tool(tools, _is_code_execution_tool_type)


def _extract_web_search_tool(
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first web-search-style entry out of ``tools``.

    Accepts both the gateway-native ``{"type": "web_search"}`` shape (matching
    OpenAI's server-managed tool) and Anthropic's dated variants
    (``web_search_20250305`` etc.). All map to the same backend.
    """
    return _extract_first_matching_tool(tools, _is_web_search_tool_type)


def _resolve_web_search_purpose_hint(tool_entry: dict[str, Any] | None) -> str | None:
    """Per-tool entry → ``GATEWAY_WEB_SEARCH_PURPOSE_HINT`` → ``None`` (backend default)."""
    return (
        (tool_entry.get("purpose_hint") if tool_entry else None)
        or os.environ.get("GATEWAY_WEB_SEARCH_PURPOSE_HINT")
        or None
    )


def _build_web_search_backend(*, base_url: str, tool_entry: dict[str, Any]) -> WebSearchBackend:
    """Construct a WebSearchBackend honouring env-level + per-tool config.

    Per-tool entry fields (``max_results``, ``allowed_domains``,
    ``blocked_domains``, ``purpose_hint``) override env-level defaults.
    Operator-level env knobs:

      * ``GATEWAY_WEB_SEARCH_ENGINES`` — comma-separated SearXNG engine list
      * ``GATEWAY_WEB_SEARCH_MAX_RESULTS`` — default cap on returned hits
      * ``GATEWAY_WEB_SEARCH_EXTRACT`` — "0"/"false" to disable in-process
        content extraction (snippet-only mode).
      * ``GATEWAY_WEB_SEARCH_PURPOSE_HINT`` — per-deployment hint override.
    """
    kwargs: dict[str, Any] = {"base_url": base_url}

    engines_str = os.environ.get("GATEWAY_WEB_SEARCH_ENGINES")
    if engines_str:
        engines = tuple(e.strip() for e in engines_str.split(",") if e.strip())
        if engines:
            kwargs["engines"] = engines

    max_env = os.environ.get("GATEWAY_WEB_SEARCH_MAX_RESULTS")
    if max_env:
        try:
            parsed_max = int(max_env)
        except ValueError:
            logger.warning("GATEWAY_WEB_SEARCH_MAX_RESULTS=%r is not an int; ignoring", max_env)
        else:
            if parsed_max >= 1:
                kwargs["max_results"] = parsed_max
            else:
                logger.warning("GATEWAY_WEB_SEARCH_MAX_RESULTS=%r is not >= 1; ignoring", max_env)
    req_max = tool_entry.get("max_results")
    if isinstance(req_max, int) and req_max > 0:
        kwargs["max_results"] = req_max

    extract_env = os.environ.get("GATEWAY_WEB_SEARCH_EXTRACT")
    if extract_env is not None:
        kwargs["extract_content"] = extract_env.lower() not in {"0", "false", "no", "off"}

    allowed = tool_entry.get("allowed_domains")
    if isinstance(allowed, list) and allowed:
        kwargs["allowed_domains"] = tuple(str(d) for d in allowed)
    blocked = tool_entry.get("blocked_domains")
    if isinstance(blocked, list) and blocked:
        kwargs["blocked_domains"] = tuple(str(d) for d in blocked)

    purpose_hint = _resolve_web_search_purpose_hint(tool_entry)
    if purpose_hint:
        kwargs["purpose_hint"] = purpose_hint

    return WebSearchBackend(**kwargs)


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)

    @field_validator("messages")
    @classmethod
    def validate_message_structure(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, message in enumerate(v):
            if "role" not in message:
                msg = f"messages[{i}]: 'role' is required"
                raise ValueError(msg)
        return v

    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    mcp_servers: list[McpServerConfig] | None = None
    mcp_server_ids: list[uuid.UUID] | None = None
    tools_header: str | None = Field(
        default=None,
        max_length=4000,
        description=(
            "Optional override for the lead-in that the gateway prepends before the "
            "per-tool hint block in the system message. Useful for expressing "
            "global tool-selection policy (e.g. 'prefer MCP tools over code_execution'). "
            "Falls back to GATEWAY_TOOLS_HEADER env, then to the built-in default."
        ),
    )
    max_tool_iterations: int | None = Field(default=None, ge=1, le=MAX_TOOL_ITERATIONS_CAP)


def get_provider_kwargs(
    config: GatewayConfig,
    provider: LLMProvider,
) -> dict[str, Any]:
    """Get provider kwargs from config for acompletion calls.

    Args:
        config: Gateway configuration
        provider: Provider name

    Returns:
        Dictionary of provider kwargs (credentials, client_args, etc.)

    """
    kwargs: dict[str, Any] = {}
    if provider.value in config.providers:
        provider_config = config.providers[provider.value]

        if provider == LLMProvider.VERTEXAI:
            vertex_creds = provider_config.get("credentials")
            vertex_project = provider_config.get("project")
            vertex_location = provider_config.get("location")

            kwargs.update(
                setup_vertex_environment(
                    credentials=vertex_creds,
                    project=vertex_project,
                    location=vertex_location,
                )
            )
            if "client_args" in provider_config:
                kwargs["client_args"] = provider_config["client_args"]
        else:
            kwargs = {k: v for k, v in provider_config.items() if k != "client_args"}
            if "client_args" in provider_config:
                kwargs["client_args"] = provider_config["client_args"]

    return kwargs


async def log_usage(
    db: AsyncSession,
    log_writer: LogWriter,
    api_key_id: str | None,
    model: str,
    provider: str | None,
    endpoint: str,
    user_id: str | None = None,
    response: ChatCompletion | AsyncIterator[ChatCompletionChunk] | None = None,
    usage_override: CompletionUsage | None = None,
    error: str | None = None,
) -> None:
    """Log API usage to database and update user spend.

    Args:
        db: Database session
        api_key_id: API key identifier (None if using master key)
        model: Model name
        provider: Provider name
        endpoint: Endpoint path
        user_id: User identifier for tracking
        response: Response object (if successful)
        usage_override: Usage data for streaming requests
        error: Error message (if failed)

    """
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
        api_key_id=api_key_id,
        user_id=user_id,
        timestamp=datetime.now(UTC),
        model=model,
        provider=provider,
        endpoint=endpoint,
        status="success" if error is None else "error",
        error_message=error,
    )

    usage_data = usage_override
    if not usage_data and response and isinstance(response, ChatCompletion) and response.usage:
        usage_data = response.usage

    if usage_data:
        usage_log.prompt_tokens = usage_data.prompt_tokens
        usage_log.completion_tokens = usage_data.completion_tokens
        usage_log.total_tokens = usage_data.total_tokens

        record_tokens(
            str(provider or ""),
            model,
            usage_data.prompt_tokens,
            usage_data.completion_tokens,
        )

        pricing = await find_model_pricing(db, provider, model, as_of=usage_log.timestamp)
        if pricing:
            cost = (usage_data.prompt_tokens / 1_000_000) * pricing.input_price_per_million + (
                usage_data.completion_tokens / 1_000_000
            ) * pricing.output_price_per_million
            usage_log.cost = cost
            record_cost(str(provider or ""), model, cost)
        else:
            model_ref = f"{provider}:{model}" if provider else model
            logger.warning(f"No pricing configured for '{model_ref}'. Usage will be tracked without cost.")

    await log_writer.put(usage_log)


def _extract_platform_user_token(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    token = auth_header[7:].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    return token


def _split_model_selector(model_selector: str) -> tuple[str | None, str]:
    if ":" in model_selector:
        provider, model_name = model_selector.split(":", 1)
        return provider or None, model_name
    if "/" in model_selector:
        provider, model_name = model_selector.split("/", 1)
        return provider or None, model_name
    return None, model_selector


def _platform_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _safe_detail_from_platform(response: httpx.Response, fallback: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return fallback

    detail = payload.get("detail") if isinstance(payload, dict) else None
    return detail if isinstance(detail, str) else fallback


async def _post_platform(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_seconds: float,
) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        return await client.post(url, headers=headers, json=body)


class ResolvedAttempt(BaseModel):
    """A single resolution attempt returned by the platform."""

    attempt_id: str
    position: int
    provider: str
    model: str
    api_base: str | None = None
    api_key: str
    managed: bool


class ResolvedRoute(BaseModel):
    """The full resolution plan returned by the platform."""

    request_id: str
    fallback_enabled: bool
    attempts: list[ResolvedAttempt]


async def _resolve_platform_credentials(
    config: GatewayConfig,
    user_token: str,
    model_selector: str,
) -> ResolvedRoute:
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Platform mode is misconfigured",
        )

    provider, model_name = _split_model_selector(model_selector)
    timeout_ms = int(config.platform.get("resolve_timeout_ms", 5000))
    resolve_url = _platform_url(platform_base_url, "/gateway/provider-keys/resolve")
    resolve_headers = {
        "X-Gateway-Token": config.platform_token or "",
        "X-User-Token": user_token,
    }
    resolve_body: dict[str, Any] = {"model": model_name}
    if provider:
        resolve_body["provider"] = provider

    try:
        response = await _post_platform(
            url=resolve_url,
            headers=resolve_headers,
            body=resolve_body,
            timeout_seconds=timeout_ms / 1000,
        )
    except (httpx.TimeoutException, httpx.NetworkError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        ) from None

    if response.status_code == 200:
        payload = response.json()
        return _parse_resolve_payload(payload)

    if response.status_code in {401, 402, 403, 404, 429}:
        detail = _safe_detail_from_platform(response, "Authorization request rejected")
        headers: dict[str, str] | None = None
        if response.status_code == 429 and response.headers.get("Retry-After"):
            headers = {"Retry-After": response.headers["Retry-After"]}
        raise HTTPException(status_code=response.status_code, detail=detail, headers=headers)

    if response.status_code == 422 or response.status_code >= 500:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        )

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Authorization service unavailable",
    )


def _parse_resolve_payload(payload: dict[str, Any]) -> ResolvedRoute:
    """Build a ResolvedRoute from either the new attempts-list shape or the
    legacy single-attempt shape.

    The legacy shape lacks ``attempts``/``request_id`` and instead has the
    primary attempt's fields at the top level (``provider``, ``model``,
    ``api_key``, ``api_base``, ``managed``, ``correlation_id``). Older otari
    deployments still respond this way; we map them onto a single-attempt route
    so the rest of the gateway code never has to know.
    """
    attempts_payload = payload.get("attempts")
    if attempts_payload is not None:
        attempts = [
            ResolvedAttempt(
                attempt_id=str(att["attempt_id"]),
                position=int(att["position"]),
                provider=str(att["provider"]),
                model=str(att["model"]),
                api_base=att.get("api_base"),
                api_key=str(att["api_key"]),
                managed=bool(att.get("managed", False)),
            )
            for att in attempts_payload
        ]
        return ResolvedRoute(
            request_id=str(payload["request_id"]),
            fallback_enabled=bool(payload.get("fallback_enabled", False)),
            attempts=attempts,
        )

    correlation_id = str(payload["correlation_id"])
    return ResolvedRoute(
        request_id=correlation_id,
        fallback_enabled=False,
        attempts=[
            ResolvedAttempt(
                attempt_id=correlation_id,
                position=0,
                provider=str(payload["provider"]),
                model=str(payload["model"]),
                api_base=payload.get("api_base"),
                api_key=str(payload["api_key"]),
                managed=bool(payload.get("managed", False)),
            )
        ],
    )


# Status codes that cause the gateway to move on to the next attempt in a
# multi-attempt route. 401/403 are included because users configure multi-attempt
# routing policies on the platform precisely to handle credential outages — when
# they've opted in, an auth failure on one provider should fall through to the
# next, not surface to the client. Single-attempt requests still see auth errors
# directly because there's nothing to fall back to.
_FALLBACK_RETRYABLE_STATUS_CODES = {401, 403, 408, 429, 500, 502, 503, 504}
_FALLBACK_NON_RETRYABLE_STATUS_CODES = {400, 422}


def _classify_upstream_error(exc: BaseException) -> tuple[bool, str]:
    """Classify an upstream provider error.

    Returns ``(retryable, error_class)``. ``error_class`` is a short string used
    for logging and reporting back to the platform. Streaming-only failures still
    pass through this classifier; the caller decides whether to actually retry.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return True, "timeout"
    if isinstance(exc, httpx.NetworkError):
        return True, "conn_err"

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            status_code = getattr(resp, "status_code", None)

    if isinstance(status_code, int):
        if status_code in _FALLBACK_NON_RETRYABLE_STATUS_CODES:
            return False, f"http_{status_code}"
        if status_code in _FALLBACK_RETRYABLE_STATUS_CODES or 500 <= status_code <= 599:
            return True, f"http_{status_code}"
        return False, f"http_{status_code}"

    return False, "unknown"


async def _resolve_platform_mcp_servers(
    config: GatewayConfig,
    user_token: str,
    mcp_server_ids: list[uuid.UUID],
) -> list[McpServerConfig]:
    """Swap workspace-scoped MCP server ids for inline configs by calling the platform."""
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Platform mode is misconfigured",
        )

    timeout_ms = int(config.platform.get("resolve_timeout_ms", 5000))
    resolve_url = _platform_url(platform_base_url, "/gateway/mcp-servers/resolve")
    headers = {
        "X-Gateway-Token": config.platform_token or "",
        "X-User-Token": user_token,
    }
    body: dict[str, Any] = {"mcp_server_ids": [str(uid) for uid in mcp_server_ids]}

    try:
        response = await _post_platform(
            url=resolve_url, headers=headers, body=body, timeout_seconds=timeout_ms / 1000
        )
    except (httpx.TimeoutException, httpx.NetworkError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        ) from None

    if response.status_code == 200:
        payload = response.json()
        return [
            McpServerConfig(
                name=s["name"],
                url=s["url"],
                authorization_token=s.get("authorization_token"),
                purpose_hint=s.get("purpose_hint"),
                allowed_tools=s.get("allowed_tools"),
            )
            for s in payload.get("servers", [])
        ]

    # Mirror the status-code semantics of `_resolve_platform_credentials`:
    # client errors (auth/quota/not-found/rate-limit) are forwarded so the
    # caller sees the real status (and can honour Retry-After on 429), while
    # the platform's server-side or unexpected responses collapse to 502.
    if response.status_code in {401, 402, 403, 404, 429}:
        detail = _safe_detail_from_platform(response, "MCP server resolution failed")
        response_headers: dict[str, str] | None = None
        if response.status_code == 429 and response.headers.get("Retry-After"):
            response_headers = {"Retry-After": response.headers["Retry-After"]}
        raise HTTPException(status_code=response.status_code, detail=detail, headers=response_headers)

    if response.status_code == 422 or response.status_code >= 500:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        )

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Authorization service unavailable",
    )


async def _report_platform_usage(
    config: GatewayConfig,
    correlation_id: str,
    outcome: str,
    usage: CompletionUsage | None,
    error_class: str | None = None,
) -> None:
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        return

    timeout_ms = int(config.platform.get("usage_timeout_ms", 5000))
    max_retries = int(config.platform.get("usage_max_retries", 3))
    usage_url = _platform_url(platform_base_url, "/gateway/usage")
    headers = {"X-Gateway-Token": config.platform_token or ""}

    payload: dict[str, Any] = {"correlation_id": correlation_id, "status": outcome}
    if outcome == "success":
        token_usage = usage or CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        payload["usage"] = {
            "prompt_tokens": token_usage.prompt_tokens,
            "completion_tokens": token_usage.completion_tokens,
            "total_tokens": token_usage.total_tokens,
        }
    elif error_class is not None:
        payload["error_class"] = error_class

    delay_seconds = 0.25
    for attempt in range(1, max_retries + 1):
        should_retry = False
        try:
            response = await _post_platform(
                url=usage_url,
                headers=headers,
                body=payload,
                timeout_seconds=timeout_ms / 1000,
            )
            if response.status_code == 204:
                return
            if response.status_code in _USAGE_NON_RETRYABLE_STATUS_CODES:
                return
            should_retry = response.status_code >= 500
        except (httpx.TimeoutException, httpx.NetworkError):
            should_retry = True

        if not should_retry or attempt == max_retries:
            return

        await asyncio.sleep(delay_seconds)
        delay_seconds *= 2


@router.post("/completions", response_model=None)
async def chat_completions(
    raw_request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    request: ChatCompletionRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> ChatCompletion | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming responses.
    Handles reasoning content from otari providers.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use virtual user created with API key
    """
    if not request.model.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request: model is required",
        )

    api_key: APIKey | None = None
    api_key_id: str | None = None
    user_id: str | None = None
    rate_limit_info: RateLimitInfo | None = None
    platform_mode = config.is_platform_mode
    route: ResolvedRoute | None = None
    user_token: str | None = None  # set inside the platform_mode branch; referenced again later

    if platform_mode:
        user_token = _extract_platform_user_token(raw_request)
        start_time = time.perf_counter()
        route = await _resolve_platform_credentials(
            config=config,
            user_token=user_token,
            model_selector=request.model,
        )
        resolve_latency_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Otari-Request-ID"] = route.request_id
        logger.info(
            "Platform resolve succeeded request_id=%s attempts=%d fallback_enabled=%s resolve_latency_ms=%.2f",
            route.request_id,
            len(route.attempts),
            route.fallback_enabled,
            resolve_latency_ms,
        )
    else:
        if db is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session unavailable",
            )

        api_key, is_master_key = await verify_api_key_or_master_key(raw_request, db, config)
        api_key_id = api_key.id if api_key else None
        user_id = resolve_user_id(
            user_id_from_request=request.user,
            api_key=api_key,
            is_master_key=is_master_key,
            master_key_error=HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="When using master key, 'user' field is required in request body",
            ),
            no_api_key_error=HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API key validation failed",
            ),
            no_user_error=HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API key has no associated user",
            ),
        )

        rate_limit_info = check_rate_limit(raw_request, user_id)
        _ = await validate_user_budget(db, user_id, request.model, strategy=config.budget_strategy)
        if config.budget_strategy == "for_update":
            await db.rollback()

    # Workspace-scoped MCP server references (platform mode only). Callers
    # pass `mcp_server_ids: [uuid, ...]` instead of inlining each config; we
    # resolve them against the platform's `/gateway/mcp-servers/resolve`
    # endpoint and merge with any inline `mcp_servers` so the downstream
    # MCP loop sees a single list. In standalone mode there's no platform
    # to consult, so we reject the field with a 400 rather than silently
    # ignoring it.
    if request.mcp_server_ids and not platform_mode:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="mcp_server_ids is only available in platform mode",
        )
    if platform_mode and request.mcp_server_ids:
        assert user_token is not None  # guaranteed by the platform_mode branch above
        resolved_mcp_servers = await _resolve_platform_mcp_servers(
            config=config,
            user_token=user_token,
            mcp_server_ids=request.mcp_server_ids,
        )
        request.mcp_servers = (request.mcp_servers or []) + resolved_mcp_servers

    # Per-request opt-in for sandboxed code execution. Matches Anthropic /
    # OpenAI's wire shape: caller adds {"type": "code_execution"} to their
    # `tools` array. The sandbox endpoint is operator-controlled — set
    # GATEWAY_SANDBOX_URL in the gateway's environment. We deliberately do
    # NOT honour a per-request URL override (e.g. `sandbox_url` on the tool
    # entry) because that would let an untrusted caller use the gateway as
    # an open HTTP client (SSRF / arbitrary-POST surface). Operators that
    # want per-tenant sandbox isolation should stand up multiple gateway
    # instances or proxy at a layer they control.
    # Mutually exclusive with `mcp_servers` for now (multi-backend dispatch
    # is the next iteration); the multi-attempt routing-policy fallback is
    # also bypassed when sandbox is in use.
    sandbox_tool_entry, tools_after_sandbox = _extract_code_execution_tool(request.tools)
    sandbox_url: str | None = os.environ.get("GATEWAY_SANDBOX_URL") or None
    use_sandbox = False
    if sandbox_tool_entry is not None:
        if sandbox_url is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "code_execution tool requested but no sandbox is configured on this gateway. "
                    "Set GATEWAY_SANDBOX_URL on the gateway, or remove code_execution from `tools`."
                ),
            )
        if request.mcp_servers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "code_execution and mcp_servers cannot be combined in the same request yet; "
                    "pick one. Multi-backend dispatch is a planned refinement."
                ),
            )
        use_sandbox = True

    # web_search opt-in mirrors the sandbox path; see comment above for the
    # threat-model rationale. GATEWAY_WEB_SEARCH_URL is operator-controlled —
    # see web_search_backend.py for the wire protocol the service must expose.
    web_search_tool_entry, remaining_user_tools = _extract_web_search_tool(tools_after_sandbox)
    web_search_url: str | None = os.environ.get("GATEWAY_WEB_SEARCH_URL") or None
    use_web_search = False
    if web_search_tool_entry is not None:
        if web_search_url is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "web_search tool requested but no search backend is configured on this gateway. "
                    "Set GATEWAY_WEB_SEARCH_URL on the gateway, or remove web_search from `tools`."
                ),
            )
        if use_sandbox or request.mcp_servers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "web_search cannot be combined with code_execution or mcp_servers in the same "
                    "request yet; pick one."
                ),
            )
        use_web_search = True

    # ------------------------------------------------------------------
    # Streaming path: iterate `route.attempts` before any bytes are flushed,
    # then commit to the first attempt that yields a chunk. Implemented in
    # `_run_streaming_with_fallback` via `iterate_streaming_attempts`.
    #
    # Mid-stream failover (after first chunk) is out of scope: recovering
    # would require either silently buffering the prefix (delays first byte)
    # or a client-aware "restart" event (breaks OpenAI-SDK compatibility).
    # Errors after first chunk propagate to the client.
    # ------------------------------------------------------------------
    if request.stream:
        # Sandbox / web_search paths collapse to the standalone streaming block
        # below so the tool loop sees a single set of credentials; multi-attempt
        # fallback is intentionally not layered around them (see comment above).
        if platform_mode and not use_sandbox and not use_web_search:
            if route is None or not route.attempts:
                if route is not None:
                    logger.error(
                        "Platform returned empty attempts list request_id=%s",
                        route.request_id,
                    )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Authorization service returned no resolvable provider",
                )
            try:
                return await _run_streaming_with_fallback(
                    route=route,
                    request=request,
                    response=response,
                    config=config,
                    background_tasks=background_tasks,
                    rate_limit_info=rate_limit_info,
                )
            except HTTPException:
                raise
            except Exception as exc:
                # Every attempt failed before any bytes were flushed.
                logger.error(
                    "All streaming attempts failed request_id=%s: %s",
                    route.request_id,
                    exc,
                )
                if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=(
                            "LLM provider timeout" if len(route.attempts) <= 1 else "All upstream providers timed out"
                        ),
                    ) from exc
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=("LLM provider error" if len(route.attempts) <= 1 else "All upstream providers failed"),
                ) from exc

        # Standalone path: single attempt, no fallback (unchanged from v1.0).
        provider, model = AnyLLM.split_model_provider(request.model)
        provider_kwargs = get_provider_kwargs(config, provider)

        # Standalone streaming path: single attempt, optionally wrapped in
        # the MCP tool-use loop. When `mcp_servers` is set the tool loop owns
        # the stream and re-emits chunks; the multi-attempt fallback
        # (`_run_streaming_with_fallback`) is intentionally not used because
        # an MCP-driven request shouldn't silently swap providers mid-loop.
        mcp_server_configs = request.mcp_servers
        max_tool_iterations = min(
            request.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
            MAX_TOOL_ITERATIONS_CAP,
        )

        request_fields = _strip_gateway_fields(
            request.model_dump(exclude_unset=True),
            tools_extracted=sandbox_tool_entry is not None or web_search_tool_entry is not None,
            remaining_user_tools=remaining_user_tools,
        )
        completion_kwargs = {**provider_kwargs, **request_fields}
        if completion_kwargs.get("stream_options") is None:
            completion_kwargs["stream_options"] = {"include_usage": True}

        try:
            if mcp_server_configs:
                # Bind the truthy value to a non-Optional local so mypy can
                # narrow inside the nested `_mcp_stream` closure.
                pool_configs = mcp_server_configs

                async def _mcp_stream() -> AsyncIterator[ChatCompletionChunk]:
                    async with MCPClientPool(pool_configs) as pool:
                        kwargs = {
                            **completion_kwargs,
                            "messages": inject_purpose_hints(
                                completion_kwargs["messages"],
                                pool.purpose_hints(),
                                header=request.tools_header,
                            ),
                        }
                        async for chunk in mcp_tool_loop_stream(
                            completion_kwargs=kwargs,
                            pool=pool,
                            max_iterations=max_tool_iterations,
                        ):
                            yield chunk

                stream: AsyncIterator[ChatCompletionChunk] = _mcp_stream()
            elif use_sandbox:
                # SandboxBackend duck-types as MCPClientPool — same tool-loop helper.
                # Eagerly open the backend *before* constructing the
                # StreamingResponse so that a failure to reach the sandbox
                # surfaces synchronously as an HTTP error. A lazy `async with`
                # inside the generator would only run after the response
                # was committed, and SandboxNotReachableError would land in
                # the SSE channel after a 200 OK header — confusing for
                # clients that expected a normal HTTP failure.
                assert sandbox_url is not None
                sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
                sandbox_backend = SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint)
                await sandbox_backend.__aenter__()  # may raise SandboxNotReachableError

                async def _sandbox_stream() -> AsyncIterator[ChatCompletionChunk]:
                    try:
                        kwargs = {
                            **completion_kwargs,
                            "messages": inject_purpose_hints(
                                completion_kwargs["messages"],
                                sandbox_backend.purpose_hints(),
                                header=request.tools_header,
                            ),
                        }
                        async for chunk in mcp_tool_loop_stream(
                            completion_kwargs=kwargs,
                            pool=sandbox_backend,  # type: ignore[arg-type]
                            max_iterations=max_tool_iterations,
                        ):
                            yield chunk
                    finally:
                        await sandbox_backend.__aexit__(None, None, None)

                stream = _sandbox_stream()
            elif use_web_search:
                # Same eager-open rationale as the sandbox path above.
                assert web_search_url is not None
                assert web_search_tool_entry is not None
                web_search_backend = _build_web_search_backend(
                    base_url=web_search_url,
                    tool_entry=web_search_tool_entry,
                )
                await web_search_backend.__aenter__()

                async def _web_search_stream() -> AsyncIterator[ChatCompletionChunk]:
                    try:
                        kwargs = {
                            **completion_kwargs,
                            "messages": inject_purpose_hints(
                                completion_kwargs["messages"],
                                web_search_backend.purpose_hints(),
                                header=request.tools_header,
                            ),
                        }
                        async for chunk in mcp_tool_loop_stream(
                            completion_kwargs=kwargs,
                            pool=web_search_backend,  # type: ignore[arg-type]
                            max_iterations=max_tool_iterations,
                        ):
                            yield chunk
                    finally:
                        await web_search_backend.__aexit__(None, None, None)

                stream = _web_search_stream()
            else:
                stream = await acompletion(**completion_kwargs)  # type: ignore[assignment]
        except HTTPException:
            raise
        except SandboxNotReachableError as exc:
            # The sandbox is part of the gateway's own infra, not the LLM
            # provider — surface a clearer status so operators don't chase
            # a "provider outage" that's actually the sandbox container
            # being down. 502 keeps "upstream dependency failed" semantics.
            logger.error("Sandbox unreachable for %s:%s: %s", provider, model, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
            ) from exc
        except WebSearchNotReachableError as exc:
            logger.error("Web search backend unreachable for %s:%s: %s", provider, model, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
            ) from exc
        except Exception as exc:
            if db is not None:
                await log_usage(
                    db=db,
                    log_writer=log_writer,
                    api_key_id=api_key_id,
                    model=model,
                    provider=provider,
                    endpoint="/v1/chat/completions",
                    user_id=user_id,
                    error=str(exc),
                )
            logger.error("Stream creation failed for %s:%s: %s", provider, model, exc)
            if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="LLM provider timeout",
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM provider error",
            ) from exc

        return _build_streaming_response(
            stream=stream,
            provider=provider,
            model=model,
            platform_mode=False,
            correlation_id=None,
            request_id=None,
            config=config,
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            user_id=user_id,
            rate_limit_info=rate_limit_info,
        )

    # ------------------------------------------------------------------
    # Non-streaming path. Routing-policy fallback (`route.attempts`) and MCP
    # tool-use loops are mutually exclusive: when `mcp_servers` is set we run
    # a single MCP loop against the primary attempt's credentials and skip
    # the per-attempt retry walk (see PR design note — an MCP-driven request
    # shouldn't silently swap providers between tool-use rounds).
    # ------------------------------------------------------------------
    mcp_server_configs = request.mcp_servers
    max_tool_iterations = min(
        request.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
        MAX_TOOL_ITERATIONS_CAP,
    )

    if platform_mode:
        if route is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error: missing route context",
            )
        platform_route = route
        attempts_to_try = platform_route.attempts
        if not attempts_to_try:
            logger.error(
                "Platform returned empty attempts list request_id=%s",
                platform_route.request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Authorization service returned no resolvable provider",
            )
        # MCP / sandbox / web_search modes: collapse to the primary attempt,
        # no per-attempt fallback (a tool-use loop shouldn't silently swap
        # providers between rounds).
        if mcp_server_configs or use_sandbox or use_web_search:
            attempts_to_try = attempts_to_try[:1]
    else:
        provider, model = AnyLLM.split_model_provider(request.model)
        provider_kwargs = get_provider_kwargs(config, provider)
        attempts_to_try = []  # standalone path doesn't use the attempts list

    class _AttemptFailure(NamedTuple):
        position: int
        provider: str
        model: str
        error_class: str

    failures: list[_AttemptFailure] = []
    last_exc: BaseException | None = None

    if platform_mode:
        base_request_fields = _strip_gateway_fields(
            request.model_dump(exclude_unset=True),
            tools_extracted=sandbox_tool_entry is not None or web_search_tool_entry is not None,
            remaining_user_tools=remaining_user_tools,
        )
        for attempt in attempts_to_try:
            attempt_provider = LLMProvider(attempt.provider)
            attempt_model = attempt.model
            attempt_kwargs: dict[str, Any] = {"api_key": attempt.api_key}
            if attempt.api_base:
                attempt_kwargs["api_base"] = attempt.api_base

            completion_kwargs = {
                **attempt_kwargs,
                **base_request_fields,
                "model": f"{attempt_provider.value}:{attempt_model}",
            }

            try:
                if mcp_server_configs:
                    async with MCPClientPool(mcp_server_configs) as pool:
                        mcp_kwargs = {
                            **completion_kwargs,
                            "messages": inject_purpose_hints(
                                completion_kwargs["messages"],
                                pool.purpose_hints(),
                                header=request.tools_header,
                            ),
                        }
                        completion: ChatCompletion = await mcp_tool_loop(
                            completion_kwargs=mcp_kwargs,
                            pool=pool,
                            max_iterations=max_tool_iterations,
                        )
                elif use_sandbox:
                    assert sandbox_url is not None
                    sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
                    async with SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint) as backend:
                        sandbox_kwargs = {
                            **completion_kwargs,
                            "messages": inject_purpose_hints(
                                completion_kwargs["messages"],
                                backend.purpose_hints(),
                                header=request.tools_header,
                            ),
                        }
                        completion = await mcp_tool_loop(
                            completion_kwargs=sandbox_kwargs,
                            pool=backend,  # type: ignore[arg-type]
                            max_iterations=max_tool_iterations,
                        )
                elif use_web_search:
                    assert web_search_url is not None
                    assert web_search_tool_entry is not None
                    async with _build_web_search_backend(
                        base_url=web_search_url,
                        tool_entry=web_search_tool_entry,
                    ) as web_backend:
                        web_kwargs = {
                            **completion_kwargs,
                            "messages": inject_purpose_hints(
                                completion_kwargs["messages"],
                                web_backend.purpose_hints(),
                                header=request.tools_header,
                            ),
                        }
                        completion = await mcp_tool_loop(
                            completion_kwargs=web_kwargs,
                            pool=web_backend,  # type: ignore[arg-type]
                            max_iterations=max_tool_iterations,
                        )
                else:
                    completion = await acompletion(**completion_kwargs)  # type: ignore[assignment]
            except HTTPException:
                raise
            except MaxToolIterationsExceeded as exc:
                # The gateway's own iteration cap was hit, not an upstream
                # failure. Surface a distinct 422 so callers can tell a
                # runaway tool loop apart from a provider outage.
                logger.warning(
                    "Tool loop iteration cap hit request_id=%s position=%d cap=%d",
                    platform_route.request_id,
                    attempt.position,
                    max_tool_iterations,
                )
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc
            except BaseException as exc:
                retryable, error_class = _classify_upstream_error(exc)
                background_tasks.add_task(
                    _report_platform_usage,
                    config,
                    attempt.attempt_id,
                    "error",
                    None,
                    error_class,
                )
                logger.warning(
                    "Provider call failed request_id=%s position=%d provider=%s model=%s error=%s retryable=%s",
                    platform_route.request_id,
                    attempt.position,
                    attempt.provider,
                    attempt.model,
                    error_class,
                    retryable,
                )
                last_exc = exc
                if not retryable:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="LLM provider error",
                    ) from exc
                failures.append(_AttemptFailure(attempt.position, attempt.provider, attempt.model, error_class))
                continue

            # Success on this attempt.
            background_tasks.add_task(
                _report_platform_usage,
                config,
                attempt.attempt_id,
                "success",
                completion.usage,
                None,
            )
            response.headers["X-Correlation-ID"] = attempt.attempt_id
            if rate_limit_info:
                for key, value in rate_limit_headers(rate_limit_info).items():
                    response.headers[key] = value
            return completion

        # All attempts exhausted with retryable errors.
        logger.error(
            "All upstream attempts failed request_id=%s failures=%s",
            platform_route.request_id,
            failures,
        )
        is_single_attempt = len(attempts_to_try) <= 1
        if last_exc is not None and isinstance(last_exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
            detail = "LLM provider timeout" if is_single_attempt else "All upstream providers timed out"
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=detail,
            ) from last_exc
        detail = "LLM provider error" if is_single_attempt else "All upstream providers failed"
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        ) from last_exc

    # Standalone path (no platform / no fallback). Same MCP / sandbox /
    # web_search semantics as platform mode above: if `mcp_servers` is set,
    # the request goes through the MCP tool-use loop; if `tools` includes a
    # code_execution entry, the request goes through the SandboxBackend
    # tool-use loop; if `tools` includes a web_search entry, the request
    # goes through the WebSearchBackend tool-use loop; otherwise a single
    # ``acompletion`` call.
    request_fields = _strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=sandbox_tool_entry is not None or web_search_tool_entry is not None,
        remaining_user_tools=remaining_user_tools,
    )
    completion_kwargs = {**provider_kwargs, **request_fields}

    try:
        if mcp_server_configs:
            async with MCPClientPool(mcp_server_configs) as pool:
                mcp_kwargs = {
                    **completion_kwargs,
                    "messages": inject_purpose_hints(
                        completion_kwargs["messages"],
                        pool.purpose_hints(),
                        header=request.tools_header,
                    ),
                }
                completion = await mcp_tool_loop(
                    completion_kwargs=mcp_kwargs,
                    pool=pool,
                    max_iterations=max_tool_iterations,
                )
        elif use_sandbox:
            assert sandbox_url is not None
            sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
            async with SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint) as backend:
                sandbox_kwargs = {
                    **completion_kwargs,
                    "messages": inject_purpose_hints(
                        completion_kwargs["messages"],
                        backend.purpose_hints(),
                        header=request.tools_header,
                    ),
                }
                completion = await mcp_tool_loop(
                    completion_kwargs=sandbox_kwargs,
                    pool=backend,  # type: ignore[arg-type]
                    max_iterations=max_tool_iterations,
                )
        elif use_web_search:
            assert web_search_url is not None
            assert web_search_tool_entry is not None
            async with _build_web_search_backend(
                base_url=web_search_url,
                tool_entry=web_search_tool_entry,
            ) as web_backend:
                web_kwargs = {
                    **completion_kwargs,
                    "messages": inject_purpose_hints(
                        completion_kwargs["messages"],
                        web_backend.purpose_hints(),
                        header=request.tools_header,
                    ),
                }
                completion = await mcp_tool_loop(
                    completion_kwargs=web_kwargs,
                    pool=web_backend,  # type: ignore[arg-type]
                    max_iterations=max_tool_iterations,
                )
        else:
            completion = await acompletion(**completion_kwargs)  # type: ignore[assignment]
        if db is not None:
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/chat/completions",
                user_id=user_id,
                response=completion,
            )
    except HTTPException:
        raise
    except SandboxNotReachableError as exc:
        # Sandbox is gateway-side infra, not an LLM provider. Clearer detail
        # so operators don't chase a provider outage that's really the
        # sandbox container being down.
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
        ) from exc
    except WebSearchNotReachableError as exc:
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
        ) from exc
    except MaxToolIterationsExceeded as e:
        # Gateway-owned cap, not an upstream provider failure. 422 lets
        # callers distinguish a runaway tool loop from a real outage.
        logger.warning("Tool loop iteration cap hit (standalone): cap=%d", max_tool_iterations)
        if db is not None:
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/chat/completions",
                user_id=user_id,
                error=str(e),
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        if db is not None:
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/chat/completions",
                user_id=user_id,
                error=str(e),
            )

        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        if isinstance(e, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM provider timeout",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return completion


def _build_streaming_response(
    *,
    stream: AsyncIterator[ChatCompletionChunk],
    provider: LLMProvider,
    model: str,
    platform_mode: bool,
    correlation_id: str | None,
    request_id: str | None,
    config: GatewayConfig,
    db: AsyncSession | None,
    log_writer: LogWriter | None,
    api_key_id: str | None,
    user_id: str | None,
    rate_limit_info: RateLimitInfo | None,
) -> StreamingResponse:
    """Wrap an already-opened upstream stream in an SSE response."""

    def _format_chunk(chunk: ChatCompletionChunk) -> str:
        return f"data: {chunk.model_dump_json()}\n\n"

    def _extract_usage(chunk: ChatCompletionChunk) -> CompletionUsage | None:
        if not chunk.usage:
            return None
        return CompletionUsage(
            prompt_tokens=chunk.usage.prompt_tokens or 0,
            completion_tokens=chunk.usage.completion_tokens or 0,
            total_tokens=chunk.usage.total_tokens or 0,
        )

    async def _on_complete(usage_data: CompletionUsage) -> None:
        if platform_mode and correlation_id:
            asyncio.create_task(
                _report_platform_usage(
                    config=config,
                    correlation_id=correlation_id,
                    outcome="success",
                    usage=usage_data,
                )
            )
            return
        if db is None or log_writer is None:
            return
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
            user_id=user_id,
            usage_override=usage_data,
        )

    async def _on_error(error: str) -> None:
        if platform_mode and correlation_id:
            asyncio.create_task(
                _report_platform_usage(
                    config=config,
                    correlation_id=correlation_id,
                    outcome="error",
                    usage=None,
                )
            )
            return
        if db is None or log_writer is None:
            return
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
            user_id=user_id,
            error=error,
        )

    rl_headers = rate_limit_headers(rate_limit_info) if rate_limit_info else {}
    # StreamingResponse builds its own response object, so headers we want on
    # the wire have to be passed in here — assigning to the dependency-injected
    # `Response` object doesn't propagate to streaming responses.
    headers = dict(rl_headers)
    if platform_mode and correlation_id:
        headers["X-Correlation-ID"] = correlation_id
    if platform_mode and request_id:
        headers["X-Otari-Request-ID"] = request_id
    return StreamingResponse(
        streaming_generator(
            stream=stream,
            format_chunk=_format_chunk,
            extract_usage=_extract_usage,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=_on_complete,
            on_error=_on_error,
            label=f"{provider}:{model}",
        ),
        media_type="text/event-stream",
        headers=headers,
    )


async def _run_streaming_with_fallback(
    *,
    route: ResolvedRoute,
    request: ChatCompletionRequest,
    response: Response,
    config: GatewayConfig,
    background_tasks: BackgroundTasks,
    rate_limit_info: RateLimitInfo | None,
) -> StreamingResponse:
    """Iterate route.attempts for a streaming request, falling through on any
    attempt that fails before its first chunk arrives.

    Once an attempt yields its first chunk, we commit and start flushing to the
    client — errors past that point propagate to the SSE channel as today.
    """
    # Strip gateway-internal fields the upstream SDKs don't accept (Anthropic
    # in particular rejects unknown kwargs). This path runs in platform mode
    # with neither MCP nor sandbox, so we don't pass remaining_user_tools.
    base_request_fields = _strip_gateway_fields(request.model_dump(exclude_unset=True))
    first_chunk_timeout_seconds = int(config.platform.get("streaming_first_chunk_timeout_ms", 2000)) / 1000

    async def _build_for_attempt(
        attempt: ResolvedAttempt,
    ) -> AsyncIterator[ChatCompletionChunk]:
        attempt_provider = LLMProvider(attempt.provider)
        provider_kwargs: dict[str, Any] = {"api_key": attempt.api_key}
        if attempt.api_base:
            provider_kwargs["api_base"] = attempt.api_base
        completion_kwargs = {
            **provider_kwargs,
            **base_request_fields,
            "model": f"{attempt_provider.value}:{attempt.model}",
        }
        if completion_kwargs.get("stream_options") is None:
            completion_kwargs["stream_options"] = {"include_usage": True}
        return await acompletion(**completion_kwargs)  # type: ignore[return-value]

    async def _on_attempt_failed(attempt: ResolvedAttempt, failure: StreamingAttemptFailure) -> None:
        background_tasks.add_task(
            _report_platform_usage,
            config,
            attempt.attempt_id,
            "error",
            None,
            failure.error_class,
        )
        logger.warning(
            "Streaming attempt failed request_id=%s position=%d provider=%s model=%s error=%s",
            route.request_id,
            attempt.position,
            attempt.provider,
            attempt.model,
            failure.error_class,
        )

    chosen, stream = await iterate_streaming_attempts(
        attempts=route.attempts,
        build_stream=_build_for_attempt,
        classify_error=_classify_upstream_error,
        on_attempt_failed=_on_attempt_failed,
        first_chunk_timeout_seconds=first_chunk_timeout_seconds,
    )

    return _build_streaming_response(
        stream=stream,
        provider=LLMProvider(chosen.provider),
        model=chosen.model,
        platform_mode=True,
        correlation_id=chosen.attempt_id,
        request_id=route.request_id,
        config=config,
        db=None,  # platform mode doesn't use the local DB
        log_writer=None,  # unused when db is None
        api_key_id=None,
        user_id=None,
        rate_limit_info=rate_limit_info,
    )
