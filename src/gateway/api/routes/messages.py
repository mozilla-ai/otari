import math
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any, Literal

from any_llm import AnyLLM, LLMProvider, amessages
from any_llm.types.completion import CompletionUsage
from any_llm.types.messages import (
    MessageDeltaEvent,
    MessageResponse,
    MessagesParams,
    MessageStartEvent,
    MessageStreamEvent,
)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    CodeExecutionPortDep,
    ModelProviderPortDep,
    build_sandbox_container_registry,
    build_sandbox_file_bridge,
    extract_credential_token,
    get_config,
    get_db_if_needed,
    get_log_writer,
    get_unit_of_work_if_needed,
    verify_api_key_or_master_key,
)
from gateway.api.routes._helpers import latest_user_text, routing_signal_from_messages
from gateway.api.routes._normalize import normalize_request_messages, sandbox_requested
from gateway.api.routes._pipeline import (
    DB_UNAVAILABLE_DETAIL,
    NO_RESOLVABLE_PROVIDER_DETAIL,
    ErrorKind,
    RequestContext,
    classify_provider_error,
    default_attempt_kwargs,
    prepare_gateway_tools,
    provider_error_headers,
    raise_all_streaming_attempts_failed,
    release_reservation,
    resolve_dispatch_provider,
    resolve_request_context,
    run_platform_non_stream,
    run_single_attempt_stream,
    run_standalone_non_stream,
    run_streaming_with_fallback,
    scope_prompt_cache_key,
)
from gateway.api.routes._platform import (
    ResolvedAttempt,
    SettledCost,
    _resolve_platform_credentials,
)
from gateway.api.routes._schema_derive import SESSION_LABEL_DESC, SESSION_LABEL_MAX_LENGTH, derive_request_base
from gateway.api.routes._tools import CODE_EXECUTION_HEADER, _strip_gateway_fields
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.core.usage import GatewayUsage
from gateway.log_config import logger
from gateway.models.guardrails import GuardrailConfig
from gateway.models.mcp import MAX_MCP_SERVER_IDS, McpServerConfig
from gateway.models.tools import CodeExecutor
from gateway.services.code_execution import ContainerLease
from gateway.services.file_service import StagedFile
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_loop import ToolBackend
from gateway.services.mcp_loop_messages import (
    MAX_TOOL_ITERATIONS_CAP,
    MCP_ACTIVITY_ID_PREFIX,
    MCP_CLIENT_BETA,
    SERVER_TOOL_USE_ID_PREFIX,
    WEB_SEARCH_TOOL_USE_ID_PREFIX,
    anthropic_tool_loop,
    anthropic_tool_loop_stream,
)
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME
from gateway.services.tool_format import inject_purpose_hints_anthropic, openai_to_anthropic_tools
from gateway.services.web_search_budget import WebSearchBudget
from gateway.streaming import ANTHROPIC_STREAM_FORMAT, StreamFormat
from gateway.types.attempt import Attempt

router = APIRouter(tags=["messages"])

# See chat.USAGE_ENDPOINT.
USAGE_ENDPOINT = "/v1/messages"


def _merge_anthropic_betas(body_betas: list[str] | None, raw_request: Request) -> list[str] | None:
    """Combine legacy body betas with Anthropic's standard beta header."""
    betas = list(body_betas or [])
    for header_value in raw_request.headers.getlist("anthropic-beta"):
        betas.extend(beta.strip() for beta in header_value.split(",") if beta.strip())
    return list(dict.fromkeys(betas)) or None


def _serves_messages_natively(dispatch_model: Any) -> bool:
    """Whether the dispatched provider has an Anthropic Messages API of its own.

    The same question any-llm asks before refusing ``betas``, asked the same
    way: a provider that serves Messages natively overrides ``_amessages``,
    while a bridged one inherits the base implementation that converts
    Messages to Completions (and refuses what cannot survive the conversion).
    ``SUPPORTS_MESSAGES`` does not answer it, being true for every provider the
    bridge covers.

    An unknown or unparseable selector answers yes, so nothing is stripped on a
    guess; any-llm then refuses the beta itself, as it did before.

    This reads a private any-llm attribute, which nothing public answers today
    (``SUPPORTS_MESSAGES`` is true for every bridged provider). It is a shim in
    the sense of CONTRIBUTING's "when the fix is upstream": the durable answer
    is a public capability flag on the any-llm provider class, asked for in
    https://github.com/mozilla-ai/any-llm/issues/1418, and this goes when that
    lands. ``getattr`` keeps a renamed attribute from breaking a request; it
    degrades to forwarding the betas, the pre-shim behavior.
    """
    if not isinstance(dispatch_model, str) or not dispatch_model:
        return True
    try:
        provider, _ = AnyLLM.split_model_provider(dispatch_model)
        native = getattr(AnyLLM.get_provider_class(provider), "_amessages", None)
        return native is not getattr(AnyLLM, "_amessages", None)
    except Exception:  # noqa: BLE001 - any-llm raises its own types for an unknown provider
        return True


def _split_client_betas(kwargs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Take out the betas no provider should see, and say whether MCP's was among them.

    Two never travel. The MCP client capability is the gateway's own, consumed
    here. And every beta is dropped for a provider with no Messages API of its
    own, where any-llm refuses the request outright: a beta names an Anthropic
    feature that provider was never going to serve, so forwarding it would make
    a request stop working purely because its model changed, which is the swap
    `auto` exists to keep invisible.
    """
    betas = kwargs.get("betas")
    if not isinstance(betas, list) or not betas:
        return kwargs, False

    saw_mcp_beta = MCP_CLIENT_BETA in betas
    provider_betas = (
        [beta for beta in betas if beta != MCP_CLIENT_BETA] if _serves_messages_natively(kwargs.get("model")) else []
    )
    if provider_betas == betas:
        return kwargs, False

    provider_kwargs = {**kwargs}
    if provider_betas:
        provider_kwargs["betas"] = provider_betas
    else:
        provider_kwargs.pop("betas")
    return provider_kwargs, saw_mcp_beta


class MessagesRequest(derive_request_base(MessagesParams)):  # type: ignore[misc]
    """Anthropic Messages API-compatible request.

    The wire fields are derived from any-llm's ``MessagesParams`` (see
    ``_schema_derive``) so the schema cannot silently drop a param any-llm
    forwards. Gateway-internal fields (``mcp_servers``, ``mcp_server_ids``,
    ``guardrails``, ``tools_header``, ``max_tool_iterations``) opt the request
    into gateway-managed MCP / sandbox / web_search / guardrails without
    changing the upstream wire shape. They're stripped before the request is
    forwarded.
    """

    messages: list[dict[str, Any]] = Field(min_length=1)
    # any-llm types ``stream`` as ``bool | None``; keep the Anthropic wire
    # contract (a non-nullable boolean defaulting to false) for stable SDK
    # generation.
    stream: bool = False

    # Gateway-internal: identical semantics to ChatCompletionRequest.
    mcp_servers: list[McpServerConfig] | None = None
    # Bounded on the list arm, not the union, so the ceiling caps the number of
    # ids rather than the length of any one value (see `core/sql.MAX_FILTER_VALUES`).
    mcp_server_ids: Annotated[list[uuid.UUID], Field(max_length=MAX_MCP_SERVER_IDS)] | None = None
    guardrails: list[GuardrailConfig] | None = Field(default=None, max_length=8)
    tools_header: str | None = None
    max_tool_iterations: int | None = Field(default=None, ge=1, le=MAX_TOOL_ITERATIONS_CAP)
    session_label: str | None = Field(default=None, max_length=SESSION_LABEL_MAX_LENGTH, description=SESSION_LABEL_DESC)


class CountTokensRequest(BaseModel):
    """Anthropic ``/v1/messages/count_tokens`` request.

    A subset of :class:`MessagesRequest`: the input fields that affect the token
    count, minus ``max_tokens`` and the streaming/sampling controls, since the
    endpoint only counts input tokens. ``context_management`` and ``betas`` are
    accepted for wire compatibility, but the local estimate does not apply
    provider-side context edits. Clients such as Claude Code call this on every
    turn to keep their prompt within the model's context window.
    """

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    cache_control: dict[str, Any] | None = None
    context_management: dict[str, Any] | None = None
    betas: list[str] | None = None


class CountTokensResponse(BaseModel):
    """Anthropic ``/v1/messages/count_tokens`` response."""

    input_tokens: int


def _is_gateway_minted_result(block: Any) -> bool:
    """Whether a ``web_search_tool_result`` block was minted by this gateway.

    Provenance is the reserved id prefix the gateway mints its ``server_tool_use``
    with (``mcp_loop_messages.WEB_SEARCH_TOOL_USE_ID_PREFIX``), matched here on the
    ``tool_use_id`` the result carries back. Anthropic issues ``srvtoolu_`` ids of its
    own and cannot produce that prefix, so a provider's blocks survive untouched
    whatever they contain, including a ``max_uses_exceeded`` error from its own capped
    search. Same rule as :func:`_is_gateway_minted_mcp_block`.

    The empty ``encrypted_content`` below is the older signal, kept for transcripts
    minted before the prefix existed: Anthropic always populates that field with a
    signed blob and the gateway cannot, so hits that all carry an empty value are
    ours. It only recognizes the success shape, which is why the prefix replaced it.
    An empty ``content`` list counts as ours too: that is what a gateway search with
    no usable hits produces, and a provider reporting no results uses the error shape.
    """
    if not isinstance(block, dict) or block.get("type") != "web_search_tool_result":
        return False
    if str(block.get("tool_use_id") or "").startswith(WEB_SEARCH_TOOL_USE_ID_PREFIX):
        return True
    hits = block.get("content")
    if not isinstance(hits, list):
        return False
    return all(isinstance(hit, dict) and not hit.get("encrypted_content") for hit in hits)


def _is_gateway_minted_code_execution_result(block: Any) -> bool:
    """Whether a ``code_execution_tool_result`` block was minted by this gateway.

    Provenance is the reserved ``server_tool_use`` id prefix, as for web search.
    Anthropic's own results, which a caller echoes when the provider ran the
    code, carry ``srvtoolu_`` ids and survive untouched.
    """
    if not isinstance(block, dict) or block.get("type") != "code_execution_tool_result":
        return False
    return str(block.get("tool_use_id") or "").startswith(SERVER_TOOL_USE_ID_PREFIX)


def _code_execution_pair_as_text(use: dict[str, Any] | None, result: dict[str, Any]) -> dict[str, Any]:
    """Fold a gateway-minted code-execution pair into one assistant ``text`` block.

    Unlike a web-search pair, whose hits are already in the transcript, an
    execution's output exists nowhere else, so dropping the pair would make the
    model forget what its code printed on the previous turn. A ``tool_use`` /
    ``tool_result`` rewrite is not available either: ``tool_result`` must open a
    user turn, which would mean splitting the echoed assistant message. A text
    block keeps the code and its output in the model's view in a shape every
    provider accepts.
    """
    code = str(((use or {}).get("input") or {}).get("code") or "")
    raw_content = result.get("content")
    content: dict[str, Any] = raw_content if isinstance(raw_content, dict) else {}
    parts = [f"[code executed]\n```\n{code}\n```"] if code else ["[code executed]"]
    if content.get("type") == "code_execution_tool_result_error":
        parts.append(f"error: {content.get('error_code') or 'unavailable'}")
    else:
        for label in ("stdout", "stderr"):
            value = content.get(label)
            if isinstance(value, str) and value:
                parts.append(f"{label}:\n{value}")
        return_code = content.get("return_code")
        if isinstance(return_code, int) and return_code != 0:
            parts.append(f"return_code: {return_code}")
    return {"type": "text", "text": "\n".join(parts)}


def _is_gateway_minted_mcp_block(block: Any) -> bool:
    """Whether ``block`` carries this gateway's reserved MCP activity prefix."""
    if not isinstance(block, dict):
        return False
    block_type = block.get("type")
    if block_type == "mcp_tool_use":
        activity_id = block.get("id")
    elif block_type == "mcp_tool_result":
        activity_id = block.get("tool_use_id")
    else:
        return False
    return str(activity_id or "").startswith(MCP_ACTIVITY_ID_PREFIX)


def _strip_gateway_minted_blocks(messages: Any) -> Any:
    """Drop this gateway's own server-tool blocks from inbound ``messages``.

    Continuing an Anthropic conversation means echoing the previous assistant turn.
    A gateway-minted ``web_search_tool_result`` carries an ``encrypted_content`` the
    gateway cannot sign, while a gateway-minted MCP pair describes execution the
    internal loop already consumed. Neither should be shipped to a provider on the
    next request. Mirrors ``responses._strip_gateway_minted_items``, but where
    Responses has no way to tell its own minted items from a provider's, here it can:
    both web search and MCP mint an Otari-prefixed call id a provider cannot produce.
    Genuine provider-run pairs therefore round-trip untouched. Each use is removed
    only alongside the result that answers it, matched by ``tool_use_id``, so a
    provider's pair is never split.

    A gateway-minted code-execution pair is not dropped but folded into a text
    block (:func:`_code_execution_pair_as_text`), because its output lives nowhere
    else in the transcript.

    A message left with no content is dropped: an empty ``content`` array is rejected
    by the API, and a turn that held nothing but our pair has nothing left to say.
    """
    if not isinstance(messages, list):
        return messages
    kept_messages: list[Any] = []
    dropped = 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            kept_messages.append(message)
            continue
        # Two passes: identify our web-search results and our provenance-prefixed
        # MCP uses, then drop each complete pair. A provider's pair matches neither.
        minted_web_ids = {
            block.get("tool_use_id") for block in content if _is_gateway_minted_result(block)
        }
        minted_mcp_ids = {
            block.get("id") if block.get("type") == "mcp_tool_use" else block.get("tool_use_id")
            for block in content
            if _is_gateway_minted_mcp_block(block)
        }
        minted_code_uses = {
            block.get("id"): block
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "server_tool_use"
            and block.get("name") == CODE_EXECUTION_TOOL_NAME
            and str(block.get("id") or "").startswith(SERVER_TOOL_USE_ID_PREFIX)
        }
        kept_blocks: list[Any] = []
        for block in content:
            if _is_minted_pair_member(block, minted_web_ids, minted_mcp_ids):
                continue
            if _is_gateway_minted_code_execution_result(block):
                kept_blocks.append(_code_execution_pair_as_text(minted_code_uses.get(block.get("tool_use_id")), block))
                continue
            if isinstance(block, dict) and block.get("id") in minted_code_uses:
                continue
            kept_blocks.append(block)
        if kept_blocks == content:
            kept_messages.append(message)
            continue
        dropped += len(content) - len(kept_blocks)
        if kept_blocks:
            kept_messages.append({**message, "content": kept_blocks})
    if dropped:
        logger.debug("Stripped %d gateway-minted content block(s) from the inbound messages", dropped)
    return kept_messages


def _is_minted_pair_member(
    block: Any,
    minted_web_ids: set[Any],
    minted_mcp_ids: set[Any],
) -> bool:
    """Whether ``block`` is one half of a gateway-minted server-tool pair."""
    if not isinstance(block, dict):
        return False
    if _is_gateway_minted_result(block):
        return block.get("tool_use_id") in minted_web_ids
    block_type = block.get("type")
    if block_type == "server_tool_use":
        return block.get("id") in minted_web_ids
    if block_type == "mcp_tool_use":
        return block.get("id") in minted_mcp_ids
    return block_type == "mcp_tool_result" and block.get("tool_use_id") in minted_mcp_ids


def _anthropic_error(
    error_type: str,
    message: str,
    status_code: int,
    headers: dict[str, str] | None = None,
) -> HTTPException:
    """Create an HTTPException with Anthropic-style error body."""
    return HTTPException(
        status_code=status_code,
        detail={"type": "error", "error": {"type": error_type, "message": message}},
        headers=headers,
    )


_ERR_INVALID_REQUEST = "invalid_request_error"
_ERR_API = "api_error"
_ERR_PERMISSION = "permission_error"
_ERR_AUTHENTICATION = "authentication_error"
_ERR_NOT_FOUND = "not_found_error"
_ERR_RATE_LIMIT = "rate_limit_error"

# Anthropic error.type keyed by HTTP status, used when re-wrapping a plain-string
# HTTPException into the Anthropic envelope: classified provider failures plus
# preamble auth/permission/resolve rejections. Unlisted statuses (e.g. the 502
# used for a credentials fault, or a 500) fall back to api_error.
_STATUS_TO_ANTHROPIC_TYPE = {
    400: _ERR_INVALID_REQUEST,
    401: _ERR_AUTHENTICATION,
    403: _ERR_PERMISSION,
    404: _ERR_NOT_FOUND,
    429: _ERR_RATE_LIMIT,
}


def _ensure_anthropic_error(exc: HTTPException) -> HTTPException:
    """Re-wrap a plain-string ``HTTPException`` in the Anthropic error envelope,
    preserving the status code and headers (e.g. a 429's ``Retry-After``).

    HTTPExceptions already carrying the Anthropic ``detail`` dict (raised via
    ``_anthropic_error``) pass through unchanged, so this is safe to apply to any
    HTTPException on the ``/api/v1/messages`` path, including format-agnostic ones
    raised by the hybrid preamble (platform resolve/auth) and the shared
    execution runners.
    """
    if not isinstance(exc.detail, str):
        return exc
    error_type = _STATUS_TO_ANTHROPIC_TYPE.get(exc.status_code, _ERR_API)
    return HTTPException(
        status_code=exc.status_code,
        detail={"type": "error", "error": {"type": error_type, "message": exc.detail}},
        headers=exc.headers,
    )


_MASTER_KEY_USER_REQUIRED = "When using master key, 'metadata.user_id' is required in request body"
_USER_FORBIDDEN = "'metadata.user_id' does not match the authenticated API key's user"
_PROVIDER_ERROR = "The request could not be completed by the provider"

_ERROR_KIND_TO_ANTHROPIC_TYPE = {
    ErrorKind.INVALID_REQUEST: _ERR_INVALID_REQUEST,
    ErrorKind.API: _ERR_API,
    ErrorKind.PERMISSION: _ERR_PERMISSION,
    ErrorKind.RATE_LIMIT: _ERR_RATE_LIMIT,
}


def _billable_messages_usage(usage: Any) -> GatewayUsage:
    """Use per-iteration totals when Anthropic reports compaction sampling."""
    billable_parts = list(getattr(usage, "iterations", None) or []) or [usage]
    input_tokens = sum((getattr(part, "input_tokens", None) or 0) for part in billable_parts)
    output_tokens = sum((getattr(part, "output_tokens", None) or 0) for part in billable_parts)
    return GatewayUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_tokens=sum(
            (getattr(part, "cache_read_input_tokens", None) or 0) for part in billable_parts
        ),
        cache_write_tokens=sum(
            (getattr(part, "cache_creation_input_tokens", None) or 0) for part in billable_parts
        ),
        cache_write_1h_tokens=sum(_cache_write_1h_tokens(part) for part in billable_parts),
        cache_tokens_in_prompt=False,
    )


def _messages_stream_usage(event: MessageStreamEvent) -> CompletionUsage | None:
    if isinstance(event, MessageDeltaEvent):
        return _billable_messages_usage(event.usage)
    if isinstance(event, MessageStartEvent):
        usage = event.message.usage
        input_tokens = usage.input_tokens or 0
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        if input_tokens or cache_read or cache_write:
            return GatewayUsage(
                prompt_tokens=input_tokens,
                completion_tokens=0,
                total_tokens=input_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                cache_write_1h_tokens=_cache_write_1h_tokens(usage),
                cache_tokens_in_prompt=False,
            )
    return None


def _cache_write_1h_tokens(usage: Any) -> int:
    """Read Anthropic's optional 1-hour cache-creation breakdown."""
    cache_creation = getattr(usage, "cache_creation", None)
    return getattr(cache_creation, "ephemeral_1h_input_tokens", 0) or 0


def _requested_cache_write_ttl(*values: Any) -> Literal["5m", "1h"] | None:
    """Return the longest explicitly requested Anthropic cache-write TTL."""
    found_cache_write = False

    def visit(value: Any) -> bool:
        nonlocal found_cache_write
        if isinstance(value, dict):
            cache_control = value.get("cache_control")
            if value.get("type") == "ephemeral":
                found_cache_write = True
                if value.get("ttl") == "1h":
                    return True
            if isinstance(cache_control, dict) and cache_control.get("type") == "ephemeral":
                found_cache_write = True
                if cache_control.get("ttl") == "1h":
                    return True
            return any(visit(child) for child in value.values())
        if isinstance(value, list):
            return any(visit(item) for item in value)
        return False

    if any(visit(value) for value in values):
        return "1h"
    return "5m" if found_cache_write else None


class _MessagesAdapter:
    """Anthropic Messages edges of the shared pipeline.

    Provider-call and tool-loop functions are resolved as module globals at
    call time so tests can monkeypatch ``gateway.api.routes.messages.amessages``
    and friends.
    """

    name = "messages"
    endpoint = USAGE_ENDPOINT
    stream_format: StreamFormat = ANTHROPIC_STREAM_FORMAT
    # A successful non-streaming call without provider usage data skips the
    # usage-log row (only the reservation is settled), matching the wire
    # behavior this endpoint has always had.
    log_success_without_usage = False

    def error(
        self,
        status_code: int,
        message: str,
        kind: ErrorKind = ErrorKind.API,
        headers: dict[str, str] | None = None,
    ) -> HTTPException:
        return _anthropic_error(_ERROR_KIND_TO_ANTHROPIC_TYPE[kind], message, status_code, headers)

    def provider_error(self, exc: BaseException) -> HTTPException:
        mapping = classify_provider_error(exc)
        if mapping is not None:
            error_type = _STATUS_TO_ANTHROPIC_TYPE.get(mapping.status_code, _ERR_API)
            return _anthropic_error(
                error_type,
                mapping.detail,
                mapping.status_code,
                provider_error_headers(exc, mapping.status_code),
            )
        return _anthropic_error(_ERR_API, _PROVIDER_ERROR, status.HTTP_500_INTERNAL_SERVER_ERROR)

    def format_chunk(self, chunk: MessageStreamEvent) -> str:
        return f"event: {chunk.type}\ndata: {chunk.model_dump_json(exclude_none=True)}\n\n"

    def extract_stream_usage(self, chunk: MessageStreamEvent) -> CompletionUsage | None:
        return _messages_stream_usage(chunk)

    def extract_usage(self, result: MessageResponse) -> CompletionUsage | None:
        if not result.usage:
            return None
        return _billable_messages_usage(result.usage)

    def attach_cost(
        self,
        value: MessageResponse | MessageStreamEvent,
        settlement: SettledCost,
    ) -> bool:
        usage: Any
        if isinstance(value, MessageResponse):
            usage = value.usage
        elif isinstance(value, MessageDeltaEvent):
            usage = value.usage
        else:
            return False
        if usage is None:
            return False
        updated = usage.model_copy(
            update={
                "cost_usd": settlement.cost_usd,
                "pricing_source": settlement.pricing_source,
            }
        )
        value.usage = updated
        return True

    def is_stream_cost_carrier(self, chunk: MessageStreamEvent) -> bool:
        return isinstance(chunk, MessageDeltaEvent)

    async def call_provider(self, kwargs: dict[str, Any]) -> MessageResponse:
        provider_kwargs, _ = _split_client_betas(kwargs)
        return await amessages(**provider_kwargs)  # type: ignore[return-value]

    async def open_provider_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[MessageStreamEvent]:
        provider_kwargs, _ = _split_client_betas(kwargs)
        return await amessages(**provider_kwargs)  # type: ignore[return-value]

    def prepare_stream_kwargs(
        self,
        kwargs: dict[str, Any],
        *,
        require_usage: bool = False,
    ) -> dict[str, Any]:
        del require_usage
        kwargs["stream"] = True
        return kwargs

    async def run_tool_loop(
        self,
        kwargs: dict[str, Any],
        pool: ToolBackend,
        max_iterations: int,
        on_first_response: Callable[[], None] | None = None,
        *,
        emit_native_web_search: bool = False,
        emit_native_code_execution: bool = False,
        web_search_budget: WebSearchBudget | None = None,
        container: ContainerLease | None = None,
    ) -> MessageResponse:
        # Standalone dispatch has no lock-in callback; only pass the kwarg on
        # the platform-attempt path so test fakes can mirror each call shape.
        extra: dict[str, Any] = {}
        if on_first_response is not None:
            extra["on_first_response"] = on_first_response
        if web_search_budget is not None:
            extra["web_search_budget"] = web_search_budget
        if emit_native_code_execution:
            extra["emit_native_code_execution"] = True
        if container is not None:
            extra["container"] = container
        provider_kwargs, _ = _split_client_betas(kwargs)
        return await anthropic_tool_loop(
            completion_kwargs=provider_kwargs,
            pool=pool,
            max_iterations=max_iterations,
            emit_native_web_search=emit_native_web_search,
            **extra,
        )

    def open_tool_loop_stream(
        self,
        kwargs: dict[str, Any],
        pool: ToolBackend,
        max_iterations: int,
        *,
        emit_native_web_search: bool = False,
        emit_native_code_execution: bool = False,
        web_search_budget: WebSearchBudget | None = None,
        container: ContainerLease | None = None,
    ) -> AsyncIterator[MessageStreamEvent]:
        provider_kwargs, emit_native_mcp = _split_client_betas(kwargs)
        extra: dict[str, Any] = {}
        if emit_native_mcp:
            extra["emit_native_mcp"] = True
        if web_search_budget is not None:
            extra["web_search_budget"] = web_search_budget
        if emit_native_code_execution:
            extra["emit_native_code_execution"] = True
        if container is not None:
            extra["container"] = container
        return anthropic_tool_loop_stream(
            completion_kwargs=provider_kwargs,
            pool=pool,
            max_iterations=max_iterations,
            emit_native_web_search=emit_native_web_search,
            **extra,
        )

    def inject_hints(
        self,
        kwargs: dict[str, Any],
        hints: list[tuple[str, str]],
        *,
        header: str | None,
    ) -> dict[str, Any]:
        return inject_purpose_hints_anthropic({**kwargs}, hints, header=header)

    def attempt_kwargs(
        self,
        attempt: ResolvedAttempt,
        base_request_fields: dict[str, Any],
    ) -> dict[str, Any]:
        return default_attempt_kwargs(attempt, base_request_fields)

    def local_attempt_kwargs(
        self,
        attempt: Attempt,
        base_request_fields: dict[str, Any],
    ) -> dict[str, Any]:
        return attempt.call_kwargs(base_request_fields)

    def prepare_platform_call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs


CONTAINER_ON_MANAGED_CREDENTIAL_DETAIL = (
    "container cannot be used on this route: it resolves to a provider account this gateway "
    "manages on behalf of many workspaces, and a container id addresses state on that account "
    "rather than on your workspace. Use a model served by your own provider key."
)


def _reject_container_on_managed_credential(ctx: RequestContext) -> None:
    """Refuse a caller-chosen container id when the upstream account is not the caller's.

    A container id names an execution environment and the files uploaded into it,
    scoped to the *provider account* that minted it, not to an otari tenant. A
    managed attempt runs on a credential the platform owns and many workspaces
    share, so forwarding an id the caller picked would let anyone holding one
    resume another tenant's container and read its workspace.

    This is the container-shaped case of what :func:`scope_prompt_cache_key`
    solves two calls later by namespacing the key. A container id is minted by
    the provider and carries the caller's claim on it, so it cannot be
    namespaced: forward or refuse are the only answers, and on a shared account
    the answer is refuse.

    Refused when *any* attempt on the route is managed, not only the first: which
    attempt serves the request is decided during fallback, past this point. A
    BYO-only route keeps the field, because there the account, and so the
    container, is already the caller's own. Standalone never reaches this: its
    credentials are the deployment operator's own, and the managed rung
    (``_serve_from_hosted_credential``) answers ``None`` in every build that
    mounts this route.
    """
    route = ctx.route
    if route is None or not any(attempt.managed for attempt in route.attempts):
        return
    raise _anthropic_error(
        _ERR_INVALID_REQUEST,
        CONTAINER_ON_MANAGED_CREDENTIAL_DETAIL,
        status.HTTP_400_BAD_REQUEST,
    )


_ADAPTER = _MessagesAdapter()


@router.post("/messages", response_model=None)
async def create_message(
    raw_request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    request: MessagesRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    uow: Annotated[UnitOfWork | None, Depends(get_unit_of_work_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    model_provider: ModelProviderPortDep,
    code_execution_port: CodeExecutionPortDep,
) -> dict[str, Any] | StreamingResponse:
    """Anthropic Messages API-compatible endpoint.

    Supports MCP tool-use loops, sandboxed code execution, and SearXNG
    web_search in both standalone mode and hybrid mode. Hybrid-mode requests
    resolve credentials via the platform service and get multi-attempt
    fallback across the resolved route, tool-loop requests included (fallback
    applies up to the pre-lock-in point, same as chat).
    """
    user_from_metadata = request.metadata.get("user_id") if request.metadata else None
    merged_betas = _merge_anthropic_betas(request.betas, raw_request)
    if merged_betas is not None:
        request.betas = merged_betas

    # Remove replayed gateway-owned activity before admission derives prompt
    # size. Waiting until request_fields are built below would reserve against
    # result content that never reaches the provider and can falsely reject or
    # overcharge the request. Provenance comes from each block, so this is
    # independent of whether the current request enables the same tool again.
    request.messages = _strip_gateway_minted_blocks(request.messages)

    # Uploads the normalizer found for the code-execution sandbox, handed to the
    # sandbox session once the billed user and workspace are resolved.
    sandbox_inputs: list[StagedFile] = []

    async def _normalize(
        user_id: str,
        provider: LLMProvider | None,
        model: str,
        instance: str | None,
        workspace_id: uuid.UUID | None,
        workspace_executor: CodeExecutor | None,
    ) -> tuple[int, CompletionUsage | None]:
        # Resolve uploaded file/image blocks into the Anthropic wire payload
        # before the cost estimate. Standalone only; no-op when the files
        # feature is off or the request has no attachments.
        request.messages, stats = await normalize_request_messages(
            request.messages,
            fmt="anthropic",
            config=config,
            provider=provider,
            model=model,
            db=db,
            raw_request=raw_request,
            user_id=user_id,
            instance=instance,
            workspace_id=workspace_id,
            sandbox_requested=sandbox_requested(
                request.tools,
                config=config,
                provider=provider,
                dialect=_ADAPTER.name,
                code_execution_header=raw_request.headers.get(CODE_EXECUTION_HEADER),
                workspace_executor=workspace_executor,
            ),
        )
        sandbox_inputs.extend(stats.sandbox_inputs)
        return len(str(request.messages)) + len(str(request.system or "")), stats.vision_usage()

    try:
        ctx = await resolve_request_context(
            adapter=_ADAPTER,
            raw_request=raw_request,
            response=response,
            db=db,
            uow=uow,
            config=config,
            log_writer=log_writer,
            model=request.model,
            user_id_from_request=str(user_from_metadata) if user_from_metadata else None,
            estimate_prompt_chars=len(str(request.messages)) + len(str(request.system or "")),
            estimate_max_output_tokens=request.max_tokens,
            estimate_cache_write_ttl=_requested_cache_write_ttl(
                request.cache_control,
                request.system,
                request.messages,
                request.tools,
            ),
            master_key_user_required_detail=_MASTER_KEY_USER_REQUIRED,
            user_forbidden_detail=_USER_FORBIDDEN,
            routing_signal=lambda: routing_signal_from_messages(
                request.messages, raw_request, has_tools=bool(request.tools)
            ),
            normalize_messages=_normalize,
            tools=request.tools,
        )
    except HTTPException as exc:
        # The hybrid preamble (platform resolve / auth) raises format-agnostic
        # plain-string HTTPExceptions (some with a Retry-After header); re-wrap
        # them in the Anthropic envelope so /api/v1/messages errors stay structured.
        raise _ensure_anthropic_error(exc) from exc

    if request.container is not None and ctx.hybrid_mode:
        try:
            _reject_container_on_managed_credential(ctx)
        except HTTPException:
            # A no-op in hybrid, the only mode that reaches this gate, since
            # hybrid reserves nothing locally. Kept so this exit already settles
            # if the gate ever covers a mode that does pre-debit the estimate.
            await release_reservation(ctx)
            raise

    tool_ctx = await prepare_gateway_tools(
        adapter=_ADAPTER,
        ctx=ctx,
        response=response,
        guardrails=request.guardrails,
        guardrail_text=latest_user_text(request.messages),
        tools=request.tools,
        mcp_servers=request.mcp_servers,
        mcp_server_ids=request.mcp_server_ids,
        max_tool_iterations=request.max_tool_iterations,
        tools_header=request.tools_header,
        code_execution_header=raw_request.headers.get(CODE_EXECUTION_HEADER),
        code_execution_port=code_execution_port,
        # Anthropic's own field, which is where an Anthropic SDK puts the id it
        # read off the last response. Resolved at admission against this caller's
        # leases; the provider never sees it when the sandbox runs the code.
        container_id=request.container,
        sandbox_containers=build_sandbox_container_registry(
            config=config,
            uow=ctx.uow,
            user_id=ctx.user_id,
            workspace_id=ctx.workspace_id,
            port=code_execution_port,
        ),
        sandbox_files=build_sandbox_file_bridge(
            raw_request=raw_request,
            config=config,
            uow=ctx.uow,
            user_id=ctx.user_id,
            workspace_id=ctx.workspace_id,
            inputs=sandbox_inputs,
        ),
    )

    # Strip gateway-internal fields, convert any caller-supplied OpenAI-shaped
    # tools to Anthropic shape so a mixed list works.
    request_fields = _strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=tool_ctx.tools_extracted,
        remaining_user_tools=tool_ctx.remaining_user_tools,
        web_search_declared_name=tool_ctx.web_search_declared_name,
    )
    scope_prompt_cache_key(request_fields, ctx)
    if request_fields.get("tools"):
        request_fields["tools"] = openai_to_anthropic_tools(request_fields["tools"])
    if tool_ctx.use_sandbox:
        # ``container`` is the gateway's to honor when the sandbox runs the code:
        # admission has already resolved it against this caller's leases, or
        # refused the request. Forwarding it would ask the provider to attach a
        # container no tool call will reach. ``prepare_gateway_tools`` either
        # claimed the provider-native declaration or refused the request, so no
        # provider tool survives alongside the sandbox and dropping it here cannot
        # strand a container the provider would have used.
        request_fields.pop("container", None)

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------
    if request.stream:
        if ctx.hybrid_mode:
            route = ctx.route
            assert route is not None  # guaranteed by the hybrid-mode preamble
            if not route.attempts:
                logger.error("Platform returned empty attempts list request_id=%s", route.request_id)
                raise _anthropic_error(
                    _ERR_API,
                    NO_RESOLVABLE_PROVIDER_DETAIL,
                    status.HTTP_502_BAD_GATEWAY,
                )
            try:
                return await run_streaming_with_fallback(
                    adapter=_ADAPTER,
                    route=route,
                    base_request_fields=request_fields,
                    config=config,
                    background_tasks=background_tasks,
                    rate_limit_info=ctx.rate_limit_info,
                    tool_ctx=tool_ctx,
                    session_label=request.session_label,
                    started_at=ctx.started_at,
                )
            except HTTPException as exc:
                # Hybrid terminal failures arrive as format-agnostic plain-string
                # HTTPExceptions; ensure the Anthropic envelope (dict details pass
                # through unchanged).
                converted = _ensure_anthropic_error(exc)
                if converted is exc:
                    raise
                raise converted from exc
            except Exception as exc:
                raise_all_streaming_attempts_failed(_ADAPTER, exc, route)

        # Standalone: single attempt streaming.
        resolved = await resolve_dispatch_provider(
            ctx, config, request.model, adapter=_ADAPTER, model_provider=model_provider
        )
        call_kwargs = {**resolved.kwargs, **request_fields, "model": resolved.dispatch_model}
        return await run_single_attempt_stream(
            adapter=_ADAPTER,
            ctx=ctx,
            tool_ctx=tool_ctx,
            call_kwargs=call_kwargs,
            provider=resolved.instance,
            model=resolved.model,
            session_label=request.session_label,
            display_model=resolved.alias,
            base_request_fields=request_fields,
        )

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------
    if ctx.hybrid_mode:
        route = ctx.route
        assert route is not None  # guaranteed by the hybrid-mode preamble
        try:
            result = await run_platform_non_stream(
                adapter=_ADAPTER,
                route=route,
                base_request_fields=request_fields,
                tool_ctx=tool_ctx,
                response=response,
                background_tasks=background_tasks,
                config=config,
                rate_limit_info=ctx.rate_limit_info,
                session_label=request.session_label,
            )
        except HTTPException as exc:
            # Hybrid terminal failures arrive as format-agnostic plain-string
            # HTTPExceptions; ensure the Anthropic envelope (dict details pass
            # through unchanged, including the sandbox / web_search 502s that
            # run_platform_non_stream raises via the adapter).
            converted = _ensure_anthropic_error(exc)
            if converted is exc:
                raise
            raise converted from exc
        return result.model_dump(exclude_none=True)

    # Standalone non-stream path
    resolved = await resolve_dispatch_provider(
        ctx, config, request.model, adapter=_ADAPTER, model_provider=model_provider
    )
    call_kwargs = {**resolved.kwargs, **request_fields, "model": resolved.dispatch_model}
    result = await run_standalone_non_stream(
        adapter=_ADAPTER,
        ctx=ctx,
        tool_ctx=tool_ctx,
        call_kwargs=call_kwargs,
        response=response,
        provider=resolved.instance,
        model=resolved.model,
        display_model=resolved.alias,
        base_request_fields=request_fields,
    )

    return result.model_dump(exclude_none=True)


# Input tokens are approximated as ``chars / 4``, because the gateway has no tokenizer.
# The count rounds up, so a caller that gauges context-window headroom stays inside the limit.
_CHARS_PER_TOKEN = 4


def _estimate_input_tokens(request: CountTokensRequest) -> int:
    """Approximate the prompt's input-token count from its serialized length."""
    chars = len(str(request.messages))
    if request.system:
        chars += len(str(request.system))
    if request.tools:
        chars += len(str(request.tools))
    return max(1, math.ceil(chars / _CHARS_PER_TOKEN))


@router.post("/messages/count_tokens")
async def count_message_tokens(
    raw_request: Request,
    request: CountTokensRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> CountTokensResponse:
    """Anthropic ``/v1/messages/count_tokens``-compatible endpoint.

    Returns ``{"input_tokens": N}`` without contacting an upstream provider:
    counting is local, so there is no budget reservation, pricing, or usage
    logging. Authentication mirrors :func:`create_message` — hybrid mode
    resolves the caller's token against the platform, standalone mode validates
    the API key — so the endpoint is not an open token-counting oracle.
    """
    try:
        if config.is_hybrid_mode:
            # Resolve against the platform purely to authenticate the caller (same
            # as create_message); the routing plan is discarded since counting is
            # local. Without this, any non-empty bearer string would be accepted.
            user_token = extract_credential_token(raw_request)
            await _resolve_platform_credentials(
                config=config,
                user_token=user_token,
                model_selector=request.model,
            )
        else:
            if db is None:
                raise _anthropic_error(_ERR_API, DB_UNAVAILABLE_DETAIL, status.HTTP_500_INTERNAL_SERVER_ERROR)
            # No session cookie here either, for the reason the completions path
            # gives (`_pipeline.resolve_request_context`): this counts tokens for
            # the plane a cookie may not reach, so it must not report on one.
            await verify_api_key_or_master_key(raw_request, db, config)
    except HTTPException as exc:
        # Keep count_tokens auth errors in the Anthropic envelope too.
        raise _ensure_anthropic_error(exc) from exc

    return CountTokensResponse(input_tokens=_estimate_input_tokens(request))
