"""Unit tests for fix: unparseable / unknown-provider model names return 400.

Previously ValueError / AnyLLMError from resolve_provider_selector escaped
as a bare 500.  The fix adds _raise_for_unresolvable_model (called by
resolve_dispatch_provider and every bare resolve_provider_selector site) to
map both exception types to HTTP 400 with a helpful detail string.

These tests cover:
- _raise_for_unresolvable_model maps ValueError -> 400 with model name in detail
- _raise_for_unresolvable_model maps AnyLLMError -> 400 with model name in detail
- resolve_dispatch_provider falls through to the guard when ctx.resolved_provider is None
- resolve_dispatch_provider returns cached provider when ctx.resolved_provider is set
"""

from unittest.mock import MagicMock, patch

import pytest
from any_llm.exceptions import AnyLLMError
from fastapi import HTTPException

from gateway.api.routes._pipeline import _raise_for_unresolvable_model, resolve_dispatch_provider

# ---------------------------------------------------------------------------
# _raise_for_unresolvable_model
# ---------------------------------------------------------------------------

def test_value_error_maps_to_400() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _raise_for_unresolvable_model("nosuchmodel", ValueError("Invalid model format"))
    assert exc_info.value.status_code == 400
    assert "nosuchmodel" in exc_info.value.detail


def test_any_llm_error_maps_to_400() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _raise_for_unresolvable_model("nobody:x", AnyLLMError("Unsupported provider"))
    assert exc_info.value.status_code == 400
    assert "nobody:x" in exc_info.value.detail


def test_detail_contains_model_name() -> None:
    """Detail must name the bad model so users know what to fix."""
    model = "typo:gpt-4"
    with pytest.raises(HTTPException) as exc_info:
        _raise_for_unresolvable_model(model, ValueError("bad"))
    assert model in exc_info.value.detail


# ---------------------------------------------------------------------------
# resolve_dispatch_provider
# ---------------------------------------------------------------------------

def _make_ctx(resolved_provider: object = None) -> MagicMock:
    ctx = MagicMock()
    ctx.resolved_provider = resolved_provider
    # No DB: keeps these tests on the 400 mapping alone. The rejection row
    # resolve_dispatch_provider writes on the same path (and the refund it does
    # first) are both no-ops without a session, and are covered end to end in
    # tests/integration/test_gateway_rejection_logging.py.
    ctx.db = None
    return ctx


def _make_config() -> MagicMock:
    return MagicMock()


class _NoHostedCredential:
    """A ``ModelProviderPort`` that serves nothing, which is the core adapter's answer.

    These tests are about the 400 mapping, not about the hosted-credential rung,
    so the port is pinned to the answer that leaves a resolved provider untouched.
    An ``AsyncMock`` would not: it answers every call with a truthy mock, which
    sends the caller down the re-key path and fails on a nonsense provider name.
    """

    async def resolve_hosted_credential(self, **kwargs: object) -> None:
        return None


@pytest.mark.asyncio
async def test_resolve_dispatch_provider_returns_cached() -> None:
    """When ctx.resolved_provider is set it is returned without calling resolve_provider_selector."""
    cached = MagicMock()
    ctx = _make_ctx(resolved_provider=cached)
    with patch("gateway.api.routes._pipeline.resolve_provider_selector") as mock_rps:
        result = await resolve_dispatch_provider(
            ctx, _make_config(), "openai:gpt-4o", adapter=MagicMock(), model_provider=_NoHostedCredential()
        )
    assert result is cached
    mock_rps.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_dispatch_provider_unparseable_raises_400() -> None:
    """When ctx.resolved_provider is None and selector is unparseable, returns 400."""
    ctx = _make_ctx(resolved_provider=None)
    with patch(
        "gateway.api.routes._pipeline.resolve_provider_selector",
        side_effect=ValueError("Invalid model format"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await resolve_dispatch_provider(
                ctx, _make_config(), "nosuchmodel", adapter=MagicMock(), model_provider=_NoHostedCredential()
            )
    assert exc_info.value.status_code == 400
    assert "nosuchmodel" in exc_info.value.detail


@pytest.mark.asyncio
async def test_resolve_dispatch_provider_unknown_provider_raises_400() -> None:
    """When ctx.resolved_provider is None and provider is unknown, returns 400."""
    ctx = _make_ctx(resolved_provider=None)
    with patch(
        "gateway.api.routes._pipeline.resolve_provider_selector",
        side_effect=AnyLLMError("Unsupported provider"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await resolve_dispatch_provider(
                ctx, _make_config(), "nobody:model", adapter=MagicMock(), model_provider=_NoHostedCredential()
            )
    assert exc_info.value.status_code == 400
    assert "nobody:model" in exc_info.value.detail


@pytest.mark.asyncio
async def test_resolve_dispatch_provider_fresh_resolution_succeeds() -> None:
    """When ctx.resolved_provider is None and selector is valid, returns resolved."""
    ctx = _make_ctx(resolved_provider=None)
    fresh = MagicMock()
    with patch(
        "gateway.api.routes._pipeline.resolve_provider_selector",
        return_value=fresh,
    ) as mock_rps:
        result = await resolve_dispatch_provider(
            ctx, _make_config(), "openai:gpt-4o", adapter=MagicMock(), model_provider=_NoHostedCredential()
        )
    assert result is fresh
    mock_rps.assert_called_once()
