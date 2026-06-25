"""Startup warning for require_pricing interacts with default pricing."""

import logging

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.services.pricing_init_service import warn_if_require_pricing_without_pricing

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
