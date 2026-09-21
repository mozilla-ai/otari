"""Shared glue for running the content normalizer from the request routes.

Resolves the target model's multimodal capabilities and rewrites file/image
content blocks in place — passthrough for natively-capable providers, extract to
text for text-only models. Used by the Chat-Completions, Anthropic Messages, and
OpenAI Responses endpoints so all three get identical handling.

Only the standalone path calls this: hybrid mode routes to frontier providers
that natively understand documents/images (passthrough is already correct) and
has no local DB / file store to resolve ``file_id`` references against.
"""

from __future__ import annotations

import uuid
from typing import Any

from any_llm import LLMProvider
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._tools import (
    _extract_code_execution_tool,
    decide_code_executor,
    first_provider_code_execution_tool,
    parse_code_execution_header,
    provider_runs_code_natively,
    resolve_code_executor_preference,
)
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.content_normalizer import NormalizationStats, WireFormat, normalize_messages
from gateway.services.model_capabilities import resolve_capabilities
from gateway.types.code_execution import CodeExecutor


def sandbox_requested(
    tools: list[dict[str, Any]] | None,
    *,
    config: GatewayConfig,
    provider: LLMProvider | None,
    dialect: str,
    code_execution_header: str | None,
    workspace_executor: CodeExecutor | None = None,
) -> bool:
    """Whether this request's code will run on the gateway's sandbox.

    True for the explicit ``otari_code_execution`` type, and for a provider's own
    declaration the executor would bring here (see ``_tools.decide_code_executor``):
    a file the request attaches is then staged for the sandbox rather than only
    shown to the model. Decided from the same three layers admission uses, the
    workspace pin (``workspace_executor``, read once in the preamble), the header
    and the deployment default, against the dispatched provider. A header outside
    the vocabulary answers false here and is refused at admission.
    """
    explicit, remaining = _extract_code_execution_tool(tools)
    if explicit is not None:
        return True
    keyword = first_provider_code_execution_tool(remaining)
    if keyword is None or not config.sandbox_configured():
        return False
    try:
        requested = parse_code_execution_header(code_execution_header)
    except ValueError:
        return False
    preference, _ = resolve_code_executor_preference(
        requested=requested, workspace=workspace_executor, deployment=config.effective_code_executor()
    )
    native = provider_runs_code_natively(
        keyword, provider=provider.value if provider is not None else None, dialect=dialect
    )
    return decide_code_executor(preference, sandbox_configured=True, native_available=native) is CodeExecutor.OTARI


async def normalize_request_messages(
    messages: list[dict[str, Any]],
    *,
    fmt: WireFormat,
    config: GatewayConfig,
    provider: LLMProvider | None,
    model: str,
    db: AsyncSession | None,
    raw_request: Request,
    user_id: str | None,
    instance: str | None = None,
    workspace_id: uuid.UUID | None = None,
    sandbox_requested: bool = False,
) -> tuple[list[dict[str, Any]], NormalizationStats]:
    """Normalize ``messages`` for the resolved ``provider/model``.

    ``sandbox_requested`` is whether the request declared the gateway's
    code-execution tool; the normalizer then records referenced uploads on the
    stats for the sandbox backend to seed (see ``NormalizationStats.sandbox_inputs``).

    No-ops (returns the input untouched) when file understanding is disabled or
    the provider couldn't be parsed — the downstream provider call surfaces an
    unknown model with its own status code.

    This is called after the budget reservation, so it must never raise: an
    unexpected failure would otherwise leak the in-flight reservation. The
    normalizer is already defensive per-block; this is a belt-and-suspenders
    guard that forwards the original messages unchanged on any error.
    """
    if provider is None or not config.file_understanding_enabled:
        return messages, NormalizationStats()
    try:
        caps = resolve_capabilities(config, provider, model, instance=instance)
        file_store = getattr(raw_request.app.state, "file_store", None)
        return await normalize_messages(
            messages,
            config=config,
            caps=caps,
            fmt=fmt,
            db=db,
            file_store=file_store,
            user_id=user_id,
            workspace_id=workspace_id,
            sandbox_requested=sandbox_requested,
        )
    except Exception as exc:  # noqa: BLE001 — never fail the request / leak the reservation
        logger.warning("content normalization failed; forwarding messages unchanged: %s", exc)
        return messages, NormalizationStats()
