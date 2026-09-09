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

from gateway.api.routes._tools import _extract_code_execution_tool
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.content_normalizer import NormalizationStats, WireFormat, normalize_messages
from gateway.services.file_service import SandboxFileBridge, StagedFile
from gateway.services.model_capabilities import resolve_capabilities


def sandbox_requested(tools: list[dict[str, Any]] | None) -> bool:
    """Whether the request declared the gateway's own code-execution tool."""
    entry, _remaining = _extract_code_execution_tool(tools)
    return entry is not None


def build_sandbox_file_bridge(
    *,
    config: GatewayConfig,
    raw_request: Request,
    hybrid_mode: bool,
    user_id: str | None,
    workspace_id: uuid.UUID | None,
    inputs: list[StagedFile],
) -> SandboxFileBridge | None:
    """The file bridge a sandbox session gets, or ``None`` where files are unavailable.

    Hybrid mode has no local file store or database to hold what a run produces,
    so its sandbox runs without one, exactly as before.
    """
    file_store = getattr(raw_request.app.state, "file_store", None)
    if hybrid_mode or not config.files_enabled or file_store is None or user_id is None or workspace_id is None:
        return None
    return SandboxFileBridge(
        file_store=file_store, config=config, user_id=user_id, workspace_id=workspace_id, inputs=inputs
    )


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
