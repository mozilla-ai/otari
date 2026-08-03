"""Startup pricing warnings: require_pricing, and unpriced search tools."""

import logging
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.models.entities import ModelPricing
from gateway.services.pricing_init_service import (
    warn_if_require_pricing_without_pricing,
    warn_if_search_tools_lack_flat_pricing,
)

_WARNING_MARKER = "ALL billable requests"


def _strict_config(*, default_pricing: bool) -> GatewayConfig:
    return GatewayConfig(master_key="k", require_pricing=True, default_pricing=default_pricing)


def _capture_gateway_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Route the ``gateway`` logger (which does not propagate) into caplog."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")


@pytest.mark.asyncio
async def test_soft_note_when_default_pricing_enabled(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """Default pricing on: no dire warning, but a softer coverage note is logged."""
    _capture_gateway_logs(caplog)
    try:
        await warn_if_require_pricing_without_pricing(_strict_config(default_pricing=True), async_db)
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert _WARNING_MARKER not in caplog.text
    assert "relying on default_pricing" in caplog.text


@pytest.mark.asyncio
async def test_warning_when_default_pricing_disabled_and_no_rows(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """With defaults off and an empty pricing table, the fail-closed warning fires."""
    _capture_gateway_logs(caplog)
    try:
        await warn_if_require_pricing_without_pricing(_strict_config(default_pricing=False), async_db)
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert _WARNING_MARKER in caplog.text


_SEARCH_WARNING_MARKER = "No flat per-request rate is configured for search tool(s)"


def _search_config() -> GatewayConfig:
    return GatewayConfig(
        master_key="k",
        require_pricing=False,
        search_tools={"exa-search": {"provider": "exa", "api_key": "k"}},
    )


async def _warn_for_search_tools(db: AsyncSession, caplog: pytest.LogCaptureFixture) -> None:
    _capture_gateway_logs(caplog)
    try:
        await warn_if_search_tools_lack_flat_pricing(_search_config(), db)
    finally:
        gateway_logger.removeHandler(caplog.handler)


@pytest.mark.asyncio
async def test_warning_for_a_search_tool_with_no_flat_rate(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """An unpriced tool reserves nothing, so the operator hears about it at startup."""
    await _warn_for_search_tools(async_db, caplog)
    assert _SEARCH_WARNING_MARKER in caplog.text
    assert "exa:exa-search" in caplog.text


@pytest.mark.asyncio
async def test_no_warning_once_a_flat_rate_exists(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """An explicit rate silences it, including a deliberate 0 for a free tool."""
    async_db.add(
        ModelPricing(
            model_key="exa:exa-search",
            effective_at=datetime(2020, 1, 1, tzinfo=UTC),
            input_price_per_million=0.0,
            output_price_per_million=0.0,
        )
    )
    await async_db.commit()

    await _warn_for_search_tools(async_db, caplog)
    assert _SEARCH_WARNING_MARKER not in caplog.text
