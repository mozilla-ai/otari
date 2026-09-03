import math
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any, Literal

from any_llm import LLMProvider, amessages
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
    ModelProviderPortDep,
    get_config,
    get_db_if_needed,
    get_log_writer,
    verify_api_key_or_master_key,
)
from gateway.api.routes._helpers import latest_user_text, routing_signal_from_messages
from gateway.api.routes._normalize import normalize_request_messages
from gateway.api.routes._pipeline import (
    DB_UNAVAILABLE_DETAIL,
    NO_RESOLVABLE_PROVIDER_DETAIL,
    ErrorKind,
    RequestContext,
    classify_provider_error,
    default_attempt_kwargs,
    prepare_gateway_tools,
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
    _extract_platform_user_token,
    _resolve_platform_credentials,
)
from gateway.api.routes._schema_derive import SESSION_LABEL_DESC, SESSION_LABEL_MAX_LENGTH, derive_request_base
from gateway.api.routes._tools import _strip_gateway_fields
from gateway.core.config import GatewayConfig
from gateway.core.usage import GatewayUsage
from gateway.log_config import logger
from gateway.models.guardrails import GuardrailConfig
from gateway.models.mcp import MAX_MCP_SERVER_IDS, McpServerConfig
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_loop import ToolBackend
from gateway.services.mcp_loop_messages import (
    MAX_TOOL_ITERATIONS_CAP,
    anthropic_tool_loop,
    anthropic_tool_loop_stream,
)
from gateway.services.tool_format import inject_purpose_hints_anthropic, openai_to_anthropic_tools
from gateway.streaming import ANTHROPIC_STREAM_FORMAT, StreamFormat
from gateway.types.attempt import Attempt

router = APIRouter(prefix="/v1", tags=["messages"])


class MessagesRequest(derive_request_base(MessagesParams)):  # type: ignore[misc]
    """Anthropic Messages API-compatible request.

    The wire fields are derived from any-llm's ``MessagesParams`` (see
    ``_schema_derive``) so the schema cannot silently drop a param any-llm
    forwards. ``container`` is an Anthropic wire param ``MessagesParams`` does
    not model, declared here and forwarded as an any-llm ``**kwargs`` param.
    Gateway-internal fields (``mcp_servers``, ``mcp_server_ids``,
    ``guardrails``, ``tools_header``, ``max_tool_iterations``) opt the request
    into gateway-managed MCP / sandbox / web_search / guardrails without
    changing the upstream wire shape. They're stripped before the request is
    forwarded.
    """

    messages: list[dict[str, Any]] = Field(min_length=1)
    # Anthropic's top-level container id, for continuing a code-execution
    # container across turns. ``MessagesParams`` does not model it, so the
    # derived base would drop a caller's value before the provider call. It
    # rides any-llm's ``**kwargs``, which is why it is also registered in
    # ``_pipeline._FORWARDED_PARAMS``: without that, a bridged (non-Anthropic)
    # provider's rejection reads as an upstream outage instead of a 400.
    #
    # Stopgap: remove this declaration once the SDK pin carries the param
    # (mozilla-ai/any-llm#1329, merged after 1.26.0; tracked in #924). Until
    # then it also shadows whatever annotation any-llm picks for it.
    container: str | None = None
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

    Provenance is the empty ``encrypted_content``: Anthropic always populates that
    field with a signed blob, and the gateway cannot, so it sends the field empty
    (see ``mcp_loop_messages._native_web_search_blocks``). A result block whose hits
    all carry an empty value is therefore ours; one carrying real signed content came
    from a provider that ran the search itself and must survive untouched.

    An empty ``content`` list counts as ours: that is what a gateway search with no
    usable hits produces, and a provider reporting no results uses the error shape
    instead.
    """
    if not isinstance(block, dict) or block.get("type") != "web_search_tool_result":
        return False
    hits = block.get("content")
    if not isinstance(hits, list):
        # The error shape (``web_search_tool_result_error``) is a dict, and only a
        # provider produces it. Never ours.
        return False
    return all(isinstance(hit, dict) and not hit.get("encrypted_content") for hit in hits)


def _strip_gateway_minted_blocks(messages: Any) -> Any:
    """Drop this gateway's own server-tool blocks from inbound ``messages``.

    Continuing an Anthropic conversation means echoing the previous assistant turn,
    and a gateway-minted ``web_search_tool_result`` carries an ``encrypted_content``
    the gateway cannot sign, so an echoed turn would ship an unsignable block to a
    provider. Mirrors ``responses._strip_gateway_minted_items``, but where Responses
    has no way to tell its own minted items from a provider's, here it can: only
    blocks with gateway provenance are removed (see
    :func:`_is_gateway_minted_result`), so a genuine provider-run search's signed
    blocks round-trip untouched even with interception on. A `server_tool_use` is
    removed only alongside the gateway-minted result that answers it, matched by
    ``tool_use_id``, so a provider's pair is never split.

    Only called when interception is active (opted in, with a backend configured),
    which is the only way one of our blocks can be in a transcript at all.

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
        # Two passes: identify our result blocks, then drop them along with the
        # server_tool_use each one answers. A provider's pair matches neither.
        minted_ids = {
            block.get("tool_use_id") for block in content if _is_gateway_minted_result(block)
        }
        kept_blocks = [block for block in content if not _is_minted_pair_member(block, minted_ids)]
        if len(kept_blocks) == len(content):
            kept_messages.append(message)
            continue
        dropped += len(content) - len(kept_blocks)
        if kept_blocks:
            kept_messages.append({**message, "content": kept_blocks})
    if dropped:
        logger.debug("Stripped %d gateway-minted content block(s) from the inbound messages", dropped)
    return kept_messages


def _is_minted_pair_member(block: Any, minted_ids: set[Any]) -> bool:
    """Whether ``block`` is one half of a gateway-minted server-tool pair."""
    if not isinstance(block, dict):
        return False
    if _is_gateway_minted_result(block):
        return True
    return block.get("type") == "server_tool_use" and block.get("id") in minted_ids


def _anthropic_error(error_type: str, message: str, status_code: int) -> HTTPException:
    """Create an HTTPException with Anthropic-style error body."""
    return HTTPException(
        status_code=status_code,
        detail={"type": "error", "error": {"type": error_type, "message": message}},
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
    HTTPException on the ``/v1/messages`` path, including format-agnostic ones
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
    endpoint = "/v1/messages"
    stream_format: StreamFormat = ANTHROPIC_STREAM_FORMAT
    # A successful non-streaming call without provider usage data skips the
    # usage-log row (only the reservation is settled), matching the wire
    # behavior this endpoint has always had.
    log_success_without_usage = False

    def error(self, status_code: int, message: str, kind: ErrorKind = ErrorKind.API) -> HTTPException:
        return _anthropic_error(_ERROR_KIND_TO_ANTHROPIC_TYPE[kind], message, status_code)

    def provider_error(self, exc: BaseException) -> HTTPException:
        mapping = classify_provider_error(exc)
        if mapping is not None:
            error_type = _STATUS_TO_ANTHROPIC_TYPE.get(mapping.status_code, _ERR_API)
            return _anthropic_error(error_type, mapping.detail, mapping.status_code)
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
        return await amessages(**kwargs)  # type: ignore[return-value]

    async def open_provider_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[MessageStreamEvent]:
        return await amessages(**kwargs)  # type: ignore[return-value]

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
    ) -> MessageResponse:
        # Standalone dispatch has no lock-in callback; only pass the kwarg on
        # the platform-attempt path so test fakes can mirror each call shape.
        extra: dict[str, Any] = {}
        if on_first_response is not None:
            extra["on_first_response"] = on_first_response
        return await anthropic_tool_loop(
            completion_kwargs=kwargs,
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
    ) -> AsyncIterator[MessageStreamEvent]:
        return anthropic_tool_loop_stream(
            completion_kwargs=kwargs,
            pool=pool,
            max_iterations=max_iterations,
            emit_native_web_search=emit_native_web_search,
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
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    model_provider: ModelProviderPortDep,
) -> dict[str, Any] | StreamingResponse:
    """Anthropic Messages API-compatible endpoint.

    Supports MCP tool-use loops, sandboxed code execution, and SearXNG
    web_search in both standalone mode and hybrid mode. Hybrid-mode requests
    resolve credentials via the platform service and get multi-attempt
    fallback across the resolved route, tool-loop requests included (fallback
    applies up to the pre-lock-in point, same as chat).
    """
    user_from_metadata = request.metadata.get("user_id") if request.metadata else None

    async def _normalize(
        user_id: str,
        provider: LLMProvider | None,
        model: str,
        instance: str | None,
        workspace_id: uuid.UUID | None,
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
        )
        return len(str(request.messages)) + len(str(request.system or "")), stats.vision_usage()

    try:
        ctx = await resolve_request_context(
            adapter=_ADAPTER,
            raw_request=raw_request,
            response=response,
            db=db,
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
        )
    except HTTPException as exc:
        # The hybrid preamble (platform resolve / auth) raises format-agnostic
        # plain-string HTTPExceptions (some with a Retry-After header); re-wrap
        # them in the Anthropic envelope so /v1/messages errors stay structured.
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
    if tool_ctx.intercepts_web_search and request_fields.get("messages"):
        request_fields["messages"] = _strip_gateway_minted_blocks(request_fields["messages"])
    if tool_ctx.use_sandbox:
        # ``container`` addresses Anthropic's own code-execution container, and
        # the gateway sandbox owns execution for this request, so the provider
        # would be asked to attach a container no tool call will reach.
        # ``prepare_gateway_tools`` has already refused the one shape where a
        # provider-native code-execution tool survives alongside the sandbox, so
        # dropping it here cannot strand a container the provider would have used.
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


# The gateway has no tokenizer (see budget_service.estimate_cost), so input
# tokens are approximated as ``chars / 4`` — the same heuristic used for budget
# pre-debit. count_tokens callers (e.g. Claude Code) use the result only to
# gauge headroom against the context window, so an approximate count is fine.
# Round up: an over-count keeps callers safely inside the context window, while
# an under-count could let a prompt slip over the limit.
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
            user_token = _extract_platform_user_token(raw_request)
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
        # Keep /v1/messages/count_tokens auth errors in the Anthropic envelope too.
        raise _ensure_anthropic_error(exc) from exc

    return CountTokensResponse(input_tokens=_estimate_input_tokens(request))
