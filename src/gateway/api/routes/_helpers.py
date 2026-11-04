from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col
from starlette.datastructures import FormData, UploadFile
from starlette.types import Message

from gateway.core.config import CONVERSATION_HEADER, ROUTER_HEADER
from gateway.core.database import DATA_ERRORS, DATABASE_ERRORS, describe_database_error
from gateway.core.env import otari_env
from gateway.core.error_codes import COUNTRY_NOT_SUPPORTED, error_headers
from gateway.log_config import logger
from gateway.models.guardrails import GuardrailConfig
from gateway.models.tenancy import Workspace
from gateway.ports.telemetry_storage_port import TelemetryStoragePort
from gateway.services.guardrails import (
    GuardrailResult,
    GuardrailsNotReachableError,
    InProcessGuardrail,
    run_input_guardrails,
)
from gateway.services.inference import is_retry_futile, request_country_gate
from gateway.services.routing import RoutingSignal
from gateway.services.traces import RequestTrace, TraceService, TraceWriter
from gateway.services.url_safety import UnsafeURLError
from gateway.services.workspace_scope import default_workspace_id

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig
    from gateway.db import APIKey
    from gateway.services.provider_kwargs import MissingCredential
    from gateway.services.providers import CountryGate


GUARDRAILS_RESULT_HEADER = "Otari-Guardrails"
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
       tracking). Naming a *different* user is rejected by default. That refusal
       is a misuse signal, not a budget control: spend binds to the key's own user
       whether the name is rejected or ignored.

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


def bounded_body(request: Request, max_bytes: int, too_large: str) -> Request:
    """``request``, refusing its body with 413 once more than ``max_bytes`` arrive, whatever ``Content-Length`` says.

    For a route that reads its own multipart form after its auth check: FastAPI
    parses ``File`` parameters before any dependency runs.
    """
    received = 0

    async def receive() -> Message:
        nonlocal received
        message = await request.receive()
        received += len(message.get("body", b""))
        if received > max_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, too_large)
        return message

    return Request(request.scope, receive)


# Room for the multipart framing and the text fields around a route's files.
FORM_FRAMING_BYTES = 1024 * 1024
# Text fields a client may send past the ones a route reads, before its form is refused.
FORM_FIELD_SLACK = 16
FILE_SCHEMA = {"type": "string", "contentMediaType": "application/octet-stream"}
_UPLOAD_CHUNK_BYTES = 1024 * 1024


def form_field_error(field: str, msg: str, error_type: str = "value_error") -> RequestValidationError:
    """The 422 FastAPI gives a body it cannot validate, for one field of a form a route reads itself."""
    return RequestValidationError([{"type": error_type, "loc": ("body", field), "msg": msg}])


def form_fields[M: BaseModel](form: FormData, model: type[M]) -> M:
    """The text fields of ``form`` as ``model``, refused the way FastAPI refuses a body it cannot validate."""
    values = {name: form.get(name) for name in model.model_fields if name in form}
    for name, value in values.items():
        if not isinstance(value, str):
            raise form_field_error(name, "Expected a text field, not a file")
    try:
        return model.model_validate(values)
    except ValidationError as exc:
        raise RequestValidationError(
            [{"type": e["type"], "loc": ("body", *e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        ) from exc


def form_file(form: FormData, field: str) -> UploadFile:
    """The one file sent in ``field``, refused when missing or sent as text."""
    value = form.get(field)
    if value is None:
        raise form_field_error(field, "Field required", "missing")
    if not isinstance(value, UploadFile):
        raise form_field_error(field, "Expected a file, not a text field")
    return value


def multipart_request_body(
    fields: type[BaseModel] | None, files: Mapping[str, str], *, required_files: Collection[str] = ()
) -> dict[str, Any]:
    """The ``openapi_extra`` of a route that reads its own multipart form.

    The body is the text fields of ``fields`` and one file per entry of ``files``
    (name to description), which FastAPI cannot derive because the route reads
    the form itself.
    """
    schema = fields.model_json_schema() if fields is not None else {}
    return {
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            **schema.get("properties", {}),
                            **{name: {**FILE_SCHEMA, "description": about} for name, about in files.items()},
                        },
                        "required": [*schema.get("required", []), *required_files],
                    }
                }
            },
        }
    }


async def upload_chunks(file: UploadFile) -> AsyncIterator[bytes]:
    """Read the upload off the request a chunk at a time, so it is never held whole."""
    while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
        yield chunk


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


def routing_signal(raw_request: Request) -> RoutingSignal:
    """Build the router's view of a request from its routing headers."""
    return RoutingSignal(
        conversation_id=_header_value(raw_request, CONVERSATION_HEADER),
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


async def apply_input_guardrails(
    guardrails: list[GuardrailConfig] | None,
    input_text: str,
    *,
    response: Response,
    config: GatewayConfig | None = None,
    credentials: Mapping[str, str] | None = None,
    mandated: Collection[str] | None = None,
    in_process: Mapping[str, InProcessGuardrail | None] | None = None,
    trace: RequestTrace | None = None,
) -> None:
    """Enforce the input guardrails for a request before the provider call.

    ``guardrails`` is the effective list: the caller's own, merged with any the
    caller's organization mandates and any a routing policy mandates (see
    :func:`gateway.api.routes._pipeline.merge_guardrail_layers`, which is where
    the merge happens so every completion endpoint enforces a mandate alike).
    ``credentials`` carries the bearer credential an organization entry stores
    for the endpoint it names, keyed by profile; it never comes from the request
    body. ``mandated`` names the profiles a layer above the caller supplied,
    which is what decides whether a URL that fails its safety check is the
    caller's malformed request (a 400 naming their own URL) or a stored entry
    being unevaluable (governed by its ``mode`` / ``on_unavailable``, with the
    endpoint kept out of the response). ``in_process`` carries the guardrails
    this worker holds for the organization's own definitions, keyed by profile;
    one of those is answered here rather than sent to any service, and a profile
    whose guardrail this worker does not hold arrives as ``None`` and is
    unevaluable.

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
        HTTPException: ``400`` when a *caller-supplied* guardrail's ``url``
            fails the SSRF/scheme safety check (a mandated entry's failure is a
            502 or a recorded inconclusive instead, per its own settings);
            ``403`` when a ``block`` guardrail flags the input; ``502`` when a
            ``block`` guardrail that fails closed can't be evaluated. The 502
            body names the profile and not the endpoint, which goes to the log
            instead.
    """
    if not guardrails:
        return

    # Effective guardrails URL: dashboard override / config / env, falling back to
    # the env var when no config is threaded in (e.g. unit tests). A dashboard
    # override mutates config, so it hot-applies on the next request.
    default_url = (config.guardrails_url if config is not None else None) or otari_env("GUARDRAILS_URL") or None
    started = datetime.now(UTC)
    try:
        verdict = await run_input_guardrails(
            guardrails,
            input_text,
            default_url=default_url,
            credentials=credentials,
            mandated=mandated,
            in_process=in_process,
        )
    except UnsafeURLError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GuardrailsNotReachableError as exc:
        if trace is not None:
            ended = datetime.now(UTC)
            for guardrail in guardrails:
                trace.record_guardrail(
                    profile=guardrail.profile, mode=guardrail.mode, valid=None, started=started, ended=ended
                )
        # The full reason, endpoint included, goes to the log; the caller gets
        # the error's `public_detail`, which names the profile and nothing else.
        # An organization's guardrail endpoint is not the caller's to see
        # (otari#654), and it is not theirs to fix either.
        logger.warning("guardrail check could not be evaluated: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=exc.public_detail) from exc

    if trace is not None:
        # The checks run concurrently, so each span covers the whole evaluation.
        ended = datetime.now(UTC)
        for result in verdict.results:
            trace.record_guardrail(
                profile=result.profile, mode=result.mode, valid=result.valid, started=started, ended=ended
            )

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
        # `status` and `reason` are fixed words, so a `valid: null` still says
        # whether the check ran.
        response.headers[GUARDRAILS_RESULT_HEADER] = json.dumps(
            [_guardrail_summary(r) for r in verdict.results], separators=(",", ":")
        )


def _guardrail_summary(result: GuardrailResult) -> dict[str, object]:
    """One guardrail's entry in the :data:`GUARDRAILS_RESULT_HEADER` value."""
    summary: dict[str, object] = {
        "profile": result.profile,
        "mode": result.mode,
        "valid": result.valid,
        "score": result.score,
        "status": result.status,
    }
    if result.unavailable is not None:
        summary["reason"] = result.unavailable
    return summary


async def resolve_managed_workspace_id(db: AsyncSession, workspace_id: uuid.UUID | None) -> uuid.UUID:
    """The workspace a master-key management write lands in, validated.

    ``None`` means the deployment's default workspace, which is what an operator
    acting deployment-wide gets everywhere else (``services/workspace_scope``),
    and is where every row predating workspace scoping was backfilled.

    A named workspace is checked rather than left to the foreign key, matching
    ``POST /api/v1/keys``: an id naming no workspace is a bad request, and letting it
    reach the constraint answers 500 "Database error" for a value the caller
    supplied and can fix.
    """
    if workspace_id is None:
        return await default_workspace_id(db)
    named = (await db.execute(select(col(Workspace.id)).where(col(Workspace.id) == workspace_id))).scalar_one_or_none()
    if named is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found",
        )
    return named


@contextmanager
def retryable_on_database_outage(what: str) -> Iterator[None]:
    """Answer a database outage while storing an idempotent batch with a 503 to retry.

    For ingest only: every row it writes is keyed, so a sender that retries a
    batch this refused stores each event once. An OTLP exporter retries a 503 and
    drops the batch on a 500. A row the database refuses for what it holds is not
    an outage, and is raised unchanged.
    """
    try:
        yield
    except DATA_ERRORS:
        raise
    except DATABASE_ERRORS as exc:
        logger.warning("%s: database unavailable, asking the sender to retry: %s", what, describe_database_error(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not store the batch right now. Retry it.",
            headers={"Retry-After": "2"},
        ) from exc


def database_error(operation: str, exc: BaseException) -> HTTPException:
    """Log a failed database operation safely and return the generic 500 for the client.

    Logs the error's class, SQLSTATE and constraint only, never its message or a
    traceback, which carry the statement's values (see ``describe_database_error``).
    """
    logger.error("%s failed: %s", operation, describe_database_error(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error")


def missing_credential_error(missing: MissingCredential) -> HTTPException:
    """The answer for a call refused because no credential resolved for its provider."""
    return HTTPException(status_code=missing.status_code, detail=missing.detail, headers=error_headers(missing.code))


# The OpenAI and Anthropic SDKs read this header before their own retry rule
# (``_should_retry`` in each SDK's ``_base_client.py``), so ``false`` stops them
# retrying a 5xx they would otherwise retry. The name is theirs, not Otari's.
SHOULD_RETRY_HEADER = "x-should-retry"


def with_retry_hint(exc: BaseException, status_code: int, headers: dict[str, str] | None) -> dict[str, str] | None:
    """``headers`` for a provider failure answered with ``status_code``, telling the client not to retry a futile 5xx.

    Only a 5xx carries it: an SDK does not retry the 4xx answers anyway.
    """
    if status_code < status.HTTP_500_INTERNAL_SERVER_ERROR or not is_retry_futile(exc):
        return headers
    return {**(headers or {}), SHOULD_RETRY_HEADER: "false"}


async def erase_user_records(
    request: Request, storage: TelemetryStoragePort, traces: TraceService, user_id: str
) -> None:
    """Erase a user's telemetry and agent traces, ahead of soft-deleting the user.

    Raises a 500 when either erasure fails, before anything else is written, so
    the caller deletes nothing and the request can be retried.
    """
    # First, and outside the soft-delete's transaction. Telemetry storage settles
    # its own writes and a deployment can be storing them out of process, where
    # no transaction of ours reaches, so the two cannot be made atomic and the
    # order is a choice about which way to fail. This way a failed erasure
    # commits nothing else and answers 5xx, leaving the user active and the whole
    # request retryable. The other order strands it: the repeat request would
    # find the user already deleted and answer 404, so nothing would ever erase
    # what was left behind. The cost is the narrow window where a
    # later failure leaves an active user whose telemetry is gone, which is
    # analytics lost rather than an erasure silently not performed, and the
    # operator asked for that user's deletion either way.
    #
    # Broad on purpose: what this raises depends on the store this build bound,
    # so an out-of-process adapter's transport error must not escape as a 500
    # with a stack trace.
    try:
        await storage.purge_user(user_id=user_id)
    except Exception:
        logger.exception("Telemetry erasure failed for user %s; user was not deleted", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not erase this user's telemetry; the user was not deleted",
        ) from None
    # Agent traces follow the same rule, for the same reason. The trace writer
    # stops writing the user's traces first, or it would write some after the purge.
    trace_writer: TraceWriter | None = getattr(request.app.state, "trace_writer", None)
    if trace_writer is not None:
        await trace_writer.erase_user(user_id)
    try:
        await traces.purge_user(user_id)
    except Exception:
        logger.exception("Trace erasure failed for user %s; user was not deleted", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not erase this user's agent traces; the user was not deleted",
        ) from None


def country_gate_for(config: GatewayConfig, raw_request: Request) -> CountryGate:
    """The country gate for ``raw_request``."""
    client = raw_request.client
    return request_country_gate(config, raw_request.headers, client.host if client is not None else None)


def country_not_supported(gate: CountryGate, selector: str, *, noun: str = "Model") -> HTTPException:
    """The 403 for a ``selector`` whose provider ``gate`` closes to the request's country."""
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=gate.refusal_detail(selector, noun=noun),
        headers=error_headers(COUNTRY_NOT_SUPPORTED),
    )
