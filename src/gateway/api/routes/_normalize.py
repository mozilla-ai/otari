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

from typing import Any

from any_llm import LLMProvider
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.content_normalizer import NormalizationStats, WireFormat, normalize_messages
from gateway.services.model_capabilities import resolve_capabilities


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
) -> tuple[list[dict[str, Any]], NormalizationStats]:
    """Normalize ``messages`` for the resolved ``provider/model``.

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
        )
    except Exception as exc:  # noqa: BLE001 — never fail the request / leak the reservation
        logger.warning("content normalization failed; forwarding messages unchanged: %s", exc)
        return messages, NormalizationStats()
