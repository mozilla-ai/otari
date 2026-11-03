"""Startup pricing warnings: require_pricing, and unpriced search and fetch tools."""

import logging
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.models.pricing import ModelPricing
from gateway.services.pricing_init_service import (
    warn_about_legacy_tool_prices,
    warn_about_unpriced_web_tools,
    warn_if_gateway_tools_lack_pricing,
    warn_if_require_pricing_without_pricing,
)

_WARNING_MARKER = "ALL billable requests"


def _strict_config(*, default_pricing: bool) -> GatewayConfig:
    return GatewayConfig(master_key="k", require_pricing=True, default_pricing=default_pricing)


def _capture_gateway_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Route the ``gateway`` logger (which does not propagate) into caplog."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")


@pytest.mark.asyncio
async def test_soft_note_when_default_pricing_enabled(async_db: AsyncSession, caplog: pytest.LogCaptureFixture) -> None:
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


_SEARCH_WARNING_MARKER = "No price is set for search or fetch tool(s)"


def _search_config(*, require_pricing: bool) -> GatewayConfig:
    return GatewayConfig(
        master_key="k",
        require_pricing=require_pricing,
        search_tools={"exa-search": {"provider": "exa", "api_key": "k"}},
    )


async def _warn_for_web_tools(db: AsyncSession, caplog: pytest.LogCaptureFixture, config: GatewayConfig) -> None:
    _capture_gateway_logs(caplog)
    try:
        await warn_about_unpriced_web_tools(config, db)
    finally:
        gateway_logger.removeHandler(caplog.handler)


async def _warn_for_search_tools(
    db: AsyncSession, caplog: pytest.LogCaptureFixture, *, require_pricing: bool = False
) -> None:
    await _warn_for_web_tools(db, caplog, _search_config(require_pricing=require_pricing))


@pytest.mark.asyncio
async def test_warning_for_a_search_tool_with_no_flat_rate(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """An unpriced tool reserves nothing, so the operator hears about it at startup."""
    await _warn_for_search_tools(async_db, caplog)
    assert _SEARCH_WARNING_MARKER in caplog.text
    assert "exa-search" in caplog.text
    assert "price_per_call" in caplog.text
    assert "nothing is reserved" in caplog.text


@pytest.mark.asyncio
async def test_the_search_tool_warning_names_the_402_under_require_pricing(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """With require_pricing on, an unpriced tool's calls are refused, not merely unreserved."""
    await _warn_for_search_tools(async_db, caplog, require_pricing=True)
    assert _SEARCH_WARNING_MARKER in caplog.text
    assert "HTTP 402" in caplog.text
    assert "nothing is reserved" not in caplog.text


@pytest.mark.asyncio
async def test_no_warning_once_a_flat_rate_exists(async_db: AsyncSession, caplog: pytest.LogCaptureFixture) -> None:
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
    assert "exa-search" not in caplog.text


@pytest.mark.asyncio
async def test_the_tool_warning_names_builtin_fetch_with_no_price(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture, unpriced_builtin_fetch: None
) -> None:
    """The built-in fetcher is a fetch tool priced by its own row, whatever serves search."""
    await _warn_for_web_tools(async_db, caplog, GatewayConfig(require_pricing=False, web_fetch_enabled=True))

    assert _SEARCH_WARNING_MARKER in caplog.text
    assert "builtin_fetch" in caplog.text


@pytest.mark.asyncio
async def test_the_tool_warning_names_the_fetch_tool_that_reads_a_searchs_pages_with_the_fetch_tool_off(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """The pages a search reads take that fetch tool's price, so it is checked whether or not the fetch tool is on."""
    config = GatewayConfig(
        require_pricing=True,
        search_tools={"fake-search": {"provider": "fake", "fetch_tool": "fake-fetch"}},
        fetch_tools={"fake-fetch": {"provider": "fake"}, "other-fetch": {"provider": "fake"}},
    )
    async_db.add(_tool_row("fake-search", kind="search"))
    await async_db.commit()

    await _warn_for_web_tools(async_db, caplog, config)

    assert _SEARCH_WARNING_MARKER in caplog.text
    assert "fake-fetch" in caplog.text
    assert "other-fetch" not in caplog.text


@pytest.mark.asyncio
async def test_the_tool_warning_omits_the_fetch_tool_of_a_search_that_reads_no_page(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    config = GatewayConfig(
        require_pricing=True,
        web_search_extract=False,
        search_tools={"fake-search": {"provider": "fake", "fetch_tool": "fake-fetch"}},
        fetch_tools={"fake-fetch": {"provider": "fake"}},
    )
    async_db.add(_tool_row("fake-search", kind="search"))
    await async_db.commit()

    await _warn_for_web_tools(async_db, caplog, config)

    assert "fake-fetch" not in caplog.text


@pytest.mark.asyncio
async def test_gateway_tool_warning_omits_disabled_fetch(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    _capture_gateway_logs(caplog)
    try:
        await warn_if_gateway_tools_lack_pricing(GatewayConfig(require_pricing=False), async_db)
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert "web_fetch" not in caplog.text
    assert "otari:web_fetch" not in caplog.text


def _tool_row(key: str, *, kind: str = "model") -> ModelPricing:
    """A per-request rate of $0.005 stored under ``key``, a model row unless ``kind`` says otherwise."""
    return ModelPricing(
        kind=kind,
        model_key=key,
        effective_at=datetime(2020, 1, 1, tzinfo=UTC),
        input_price_per_million=5000.0,
        output_price_per_million=0.0,
        unit="request",
    )


@pytest.mark.asyncio
async def test_a_search_tools_own_row_silences_the_unpriced_warning(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    async_db.add(_tool_row("exa-search", kind="search"))
    await async_db.commit()

    await _warn_for_search_tools(async_db, caplog)
    assert "exa-search" not in caplog.text


@pytest.mark.asyncio
async def test_tool_prices_alone_still_leave_every_model_unpriced(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """A tool's own row, and one under its old model-kind key, price no model."""
    async_db.add_all([_tool_row("exa-search", kind="search"), _tool_row("builtin:builtin_fetch")])
    await async_db.commit()

    _capture_gateway_logs(caplog)
    try:
        await warn_if_require_pricing_without_pricing(_strict_config(default_pricing=False), async_db)
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert _WARNING_MARKER in caplog.text


@pytest.mark.asyncio
async def test_a_tool_price_under_its_old_key_is_reported_deprecated(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """Named so the operator can move it; a model price under the same provider is not a tool's."""
    async_db.add_all([_tool_row("exa:exa-search"), _tool_row("exa:a-model")])
    await async_db.commit()

    _capture_gateway_logs(caplog)
    try:
        await warn_about_legacy_tool_prices(_search_config(require_pricing=False), async_db)
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert "exa:exa-search (deployment)" in caplog.text
    assert "exa:a-model" not in caplog.text
    assert "price_per_call" in caplog.text


@pytest.mark.asyncio
async def test_a_tool_price_under_its_old_slash_key_is_reported_and_prices_no_model(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """The legacy ``provider/instance`` spelling still prices the tool, so it is a tool's row everywhere."""
    async_db.add(_tool_row("exa/exa-search"))
    await async_db.commit()

    _capture_gateway_logs(caplog)
    try:
        await warn_about_legacy_tool_prices(_search_config(require_pricing=False), async_db)
        await warn_if_require_pricing_without_pricing(
            GatewayConfig(
                master_key="k",
                require_pricing=True,
                default_pricing=False,
                search_tools={"exa-search": {"provider": "exa", "api_key": "k"}},
            ),
            async_db,
        )
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert "exa/exa-search (deployment)" in caplog.text
    assert _WARNING_MARKER in caplog.text


_TOOL_WARNING_MARKER = "Gateway tool(s)"


async def _warn_for_gateway_tools(db: AsyncSession, caplog: pytest.LogCaptureFixture, config: GatewayConfig) -> None:
    _capture_gateway_logs(caplog)
    try:
        await warn_if_gateway_tools_lack_pricing(config, db)
    finally:
        gateway_logger.removeHandler(caplog.handler)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "name", "kind"),
    [
        (
            GatewayConfig(require_pricing=True, search_tools={"fake-search": {"provider": "fake"}}),
            "fake-search",
            "search",
        ),
        (GatewayConfig(require_pricing=True, web_fetch_enabled=True), "builtin_fetch", "fetch"),
    ],
)
async def test_a_tools_own_row_leaves_it_out_of_the_tool_warning(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture, config: GatewayConfig, name: str, kind: str
) -> None:
    """The row the check at admission finds silences the warning, as it lets the calls through."""
    async_db.add(_tool_row(name, kind=kind))
    await async_db.commit()

    await _warn_for_web_tools(async_db, caplog, config)

    assert name not in caplog.text


@pytest.mark.asyncio
async def test_an_unpriced_instance_is_named_by_the_tool_warning_alone(
    async_db: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """``otari:web_search`` never prices an instance, so the gateway tool warning does not offer it."""
    config = GatewayConfig(require_pricing=True, search_tools={"fake-search": {"provider": "fake"}})

    await _warn_for_web_tools(async_db, caplog, config)
    await _warn_for_gateway_tools(async_db, caplog, config)

    assert "fake-search" in caplog.text
    assert "HTTP 402" in caplog.text
    assert _TOOL_WARNING_MARKER not in caplog.text
