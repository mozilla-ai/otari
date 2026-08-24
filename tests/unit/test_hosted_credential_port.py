"""Unit tests for the ``ModelProviderPort`` call on the standalone dispatch path.

``resolve_dispatch_provider`` asks the port to serve a candidate the credential
ladder could not (otari#757). What these tests pin down is the precedence and the
plain build:

- a build with no overlay (the container's own binding) leaves every candidate
  exactly as the ladder left it, so nothing changes rung for rung;
- a candidate the ladder credentialed never reaches the port at all, whether the
  key came from ``config.yml``, a stored instance, an organization-scoped key, or
  the provider's own SDK environment variable;
- an overlay-bound adapter *is* reached, and its credential re-keys the dispatch
  onto ``response_provider``;
- ``HostedAccessDeniedError`` becomes a 403 that names no adapter, and refunds;
- an adapter naming an upstream any-llm does not implement becomes a 502.
"""

import uuid
from typing import Any, cast

import pytest
from any_llm import LLMProvider
from fastapi import HTTPException

from gateway.api.routes import _pipeline as pipeline
from gateway.api.routes import chat
from gateway.container import build_container
from gateway.core.config import GatewayConfig
from gateway.ports.model_provider_port import (
    HostedAccessDeniedError,
    HostedCredential,
    ModelProviderPort,
)
from gateway.services.provider_kwargs import ResolvedProvider, resolve_provider_selector

ORGANIZATION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
WORKSPACE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")

# Every provider these tests name; cleared so a developer's own shell key cannot
# make `credential_ladder_exhausted` answer False and quietly skip the port.
_PROVIDER_ENV_VARS = ("OPENAI_API_KEY", "TOGETHER_API_KEY")


@pytest.fixture(autouse=True)
def _no_ambient_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class RecordingPort:
    """A stand-in for an overlay-bound adapter, recording what it was asked."""

    def __init__(
        self,
        *,
        credential: HostedCredential | None = None,
        error: Exception | None = None,
    ) -> None:
        self.credential = credential
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def resolve_hosted_credential(
        self,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID | None,
        provider: str,
        model: str | None,
    ) -> HostedCredential | None:
        self.calls.append(
            {
                "organization_id": organization_id,
                "workspace_id": workspace_id,
                "provider": provider,
                "model": model,
            }
        )
        if self.error is not None:
            raise self.error
        return self.credential


def _plain_build_port() -> ModelProviderPort:
    """The adapter a build with no overlay actually runs, via the composition root."""
    return build_container().resolve(ModelProviderPort, None)


def _ctx(
    *,
    resolved_provider: ResolvedProvider | None,
    organization_id: uuid.UUID | None = ORGANIZATION_ID,
    workspace_id: uuid.UUID | None = WORKSPACE_ID,
    plan: Any = None,
) -> pipeline.RequestContext:
    return pipeline.RequestContext(
        config=GatewayConfig(),
        db=None,
        log_writer=cast(Any, object()),
        hybrid_mode=False,
        route=None,
        user_token=None,
        api_key_id=None,
        user_id="user-1",
        rate_limit_info=None,
        reservation=None,
        started_at=0.0,
        workspace_id=workspace_id,
        resolved_provider=resolved_provider,
        plan=plan,
        organization_id=organization_id,
    )


def _uncredentialed() -> ResolvedProvider:
    """What the ladder produces for a candidate nothing in this gateway serves."""
    resolved = resolve_provider_selector(GatewayConfig(), "openai:gpt-4o")
    assert resolved.kwargs == {}, "guard: this candidate must reach the port uncredentialed"
    return resolved


async def _dispatch(
    ctx: pipeline.RequestContext,
    port: ModelProviderPort,
    *,
    selector: str = "openai:gpt-4o",
) -> ResolvedProvider:
    return await pipeline.resolve_dispatch_provider(
        ctx,
        GatewayConfig(),
        selector,
        adapter=chat._ADAPTER,
        model_provider=port,
    )


# ---------------------------------------------------------------------------
# The plain build behaves exactly as it did
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plain_build_leaves_an_uncredentialed_candidate_untouched() -> None:
    """The core adapter answers None, so the candidate goes to any-llm as before."""
    resolved = _uncredentialed()
    result = await _dispatch(_ctx(resolved_provider=resolved), _plain_build_port())
    assert result is resolved


@pytest.mark.asyncio
async def test_plain_build_leaves_a_credentialed_candidate_untouched() -> None:
    config = GatewayConfig(providers={"openai": {"api_key": "sk-from-config"}})
    resolved = resolve_provider_selector(config, "openai:gpt-4o")
    result = await pipeline.resolve_dispatch_provider(
        _ctx(resolved_provider=resolved),
        config,
        "openai:gpt-4o",
        adapter=chat._ADAPTER,
        model_provider=_plain_build_port(),
    )
    assert result is resolved
    assert result.kwargs == {"api_key": "sk-from-config"}


# ---------------------------------------------------------------------------
# BYO precedence: the port is the last rung, never an earlier one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_provider_key_is_not_asked_about() -> None:
    port = RecordingPort(credential=HostedCredential(api_key="hosted", api_base=None, response_provider="openai"))
    config = GatewayConfig(providers={"openai": {"api_key": "sk-from-config"}})
    resolved = resolve_provider_selector(config, "openai:gpt-4o")
    result = await pipeline.resolve_dispatch_provider(
        _ctx(resolved_provider=resolved),
        config,
        "openai:gpt-4o",
        adapter=chat._ADAPTER,
        model_provider=port,
    )
    assert result.kwargs == {"api_key": "sk-from-config"}
    assert port.calls == []


@pytest.mark.asyncio
async def test_organization_scoped_key_is_not_asked_about(monkeypatch: pytest.MonkeyPatch) -> None:
    """An organization's own key wins outright: the port is never consulted."""
    monkeypatch.setattr(
        "gateway.services.provider_kwargs.cached_org_provider_kwargs",
        lambda workspace_id, provider: {"api_key": "sk-org-owned"},
    )
    resolved = resolve_provider_selector(GatewayConfig(), "openai:gpt-4o", workspace_id=WORKSPACE_ID)
    assert resolved.kwargs == {"api_key": "sk-org-owned"}
    port = RecordingPort(credential=HostedCredential(api_key="hosted", api_base=None, response_provider="openai"))
    result = await _dispatch(_ctx(resolved_provider=resolved), port)
    assert result is resolved
    assert port.calls == []


@pytest.mark.asyncio
async def test_provider_sdk_env_var_is_not_asked_about(monkeypatch: pytest.MonkeyPatch) -> None:
    """any-llm's own env fallback is a credential in hand, so it outranks the port."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-environment")
    port = RecordingPort(credential=HostedCredential(api_key="hosted", api_base=None, response_provider="openai"))
    result = await _dispatch(_ctx(resolved_provider=_uncredentialed()), port)
    assert result.kwargs == {}
    assert port.calls == []


@pytest.mark.asyncio
async def test_request_without_an_organization_is_not_asked_about() -> None:
    """The port keys its access decision on the organization, so there is nothing to ask."""
    port = RecordingPort(credential=HostedCredential(api_key="hosted", api_base=None, response_provider="openai"))
    ctx = _ctx(resolved_provider=_uncredentialed(), organization_id=None, workspace_id=None)
    result = await _dispatch(ctx, port)
    assert result.kwargs == {}
    assert port.calls == []


@pytest.mark.asyncio
async def test_multi_candidate_plan_is_not_asked_about() -> None:
    """A routed chain dispatches from the plan's own kwargs, so asking would not serve it."""

    class _Plan:
        attempts = (object(), object())

    port = RecordingPort(credential=HostedCredential(api_key="hosted", api_base=None, response_provider="openai"))
    ctx = _ctx(resolved_provider=_uncredentialed(), plan=_Plan())
    result = await _dispatch(ctx, port)
    assert result.kwargs == {}
    assert port.calls == []


# ---------------------------------------------------------------------------
# An overlay-bound adapter is reached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overlay_adapter_serves_the_candidate() -> None:
    port = RecordingPort(
        credential=HostedCredential(
            api_key="hosted-key",
            api_base="https://fleet.example/v1",
            response_provider="together",
        )
    )
    result = await _dispatch(_ctx(resolved_provider=_uncredentialed()), port)

    assert port.calls == [
        {
            "organization_id": ORGANIZATION_ID,
            "workspace_id": WORKSPACE_ID,
            "provider": "openai",
            "model": "gpt-4o",
        }
    ]
    # Re-keyed onto the upstream that actually serves, which is what usage and
    # telemetry name; the caller's model is untouched.
    assert result.instance == "together"
    assert result.provider is LLMProvider.TOGETHER
    assert result.model == "gpt-4o"
    assert result.dispatch_model == "together:gpt-4o"
    assert result.kwargs == {"api_key": "hosted-key", "api_base": "https://fleet.example/v1"}


@pytest.mark.asyncio
async def test_overlay_adapter_without_an_api_base_omits_it() -> None:
    port = RecordingPort(
        credential=HostedCredential(api_key="hosted-key", api_base=None, response_provider="openai")
    )
    result = await _dispatch(_ctx(resolved_provider=_uncredentialed()), port)
    assert result.kwargs == {"api_key": "hosted-key"}
    assert result.instance == "openai"


@pytest.mark.asyncio
async def test_overlay_adapter_answering_none_leaves_the_candidate_alone() -> None:
    """An overlay whose own credential is absent or disabled answers as the core does."""
    port = RecordingPort(credential=None)
    resolved = _uncredentialed()
    result = await _dispatch(_ctx(resolved_provider=resolved), port)
    assert result is resolved
    assert len(port.calls) == 1


@pytest.mark.asyncio
async def test_alias_relabeling_survives_a_hosted_credential() -> None:
    """The caller's display name is the alias, not whichever upstream served it."""
    config = GatewayConfig(aliases={"fast": "openai:gpt-4o"})
    resolved = resolve_provider_selector(config, "fast")
    assert resolved.alias == "fast"
    port = RecordingPort(
        credential=HostedCredential(api_key="hosted-key", api_base=None, response_provider="together")
    )
    result = await pipeline.resolve_dispatch_provider(
        _ctx(resolved_provider=resolved),
        config,
        "fast",
        adapter=chat._ADAPTER,
        model_provider=port,
    )
    assert result.alias == "fast"
    assert result.instance == "together"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_access_denied_becomes_a_403_that_names_no_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[object] = []

    async def _record_release(ctx: pipeline.RequestContext) -> None:
        released.append(ctx)

    monkeypatch.setattr(pipeline, "release_reservation", _record_release)
    port = RecordingPort(
        error=HostedAccessDeniedError(
            "AcmeHostedAdapter: organization is not entitled to together",
            workspace_id=WORKSPACE_ID,
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await _dispatch(_ctx(resolved_provider=_uncredentialed()), port)

    assert exc_info.value.status_code == 403
    detail = str(exc_info.value.detail)
    assert "openai:gpt-4o" in detail
    assert "AcmeHostedAdapter" not in detail
    # The hold taken by the preamble is refunded before the refusal surfaces.
    assert len(released) == 1


@pytest.mark.asyncio
async def test_unknown_response_provider_becomes_a_502(monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[object] = []

    async def _record_release(ctx: pipeline.RequestContext) -> None:
        released.append(ctx)

    monkeypatch.setattr(pipeline, "release_reservation", _record_release)
    port = RecordingPort(
        credential=HostedCredential(api_key="hosted-key", api_base=None, response_provider="not-a-provider")
    )

    with pytest.raises(HTTPException) as exc_info:
        await _dispatch(_ctx(resolved_provider=_uncredentialed()), port)

    assert exc_info.value.status_code == 502
    assert pipeline.HOSTED_CREDENTIAL_UNUSABLE_DETAIL in str(exc_info.value.detail)
    assert "not-a-provider" not in str(exc_info.value.detail)
    assert len(released) == 1


# ---------------------------------------------------------------------------
# The fresh-resolution branch reaches the port too
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_resolution_also_reaches_the_port() -> None:
    """``ctx.resolved_provider`` unset (the gate could not parse the selector) still asks."""
    port = RecordingPort(
        credential=HostedCredential(api_key="hosted-key", api_base=None, response_provider="openai")
    )
    result = await _dispatch(_ctx(resolved_provider=None), port)
    assert result.kwargs == {"api_key": "hosted-key"}
    assert len(port.calls) == 1
