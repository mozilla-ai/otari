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
- an adapter naming an upstream any-llm does not implement becomes a 502;
- a candidate the port served is deployment-paid: ``BillingPort`` gates it
  before dispatch and arms the charge settlement makes, and no other candidate
  ever reaches ``BillingPort``.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any, cast

import pytest
from any_llm import LLMProvider
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes import _pipeline as pipeline
from gateway.api.routes import chat, messages, responses
from gateway.container import build_container
from gateway.core.config import GatewayConfig
from gateway.core.error_codes import ERROR_CODE_HEADER, INSUFFICIENT_FUNDS
from gateway.log_config import logger as gateway_logger
from gateway.ports.billing_port import BillingPort, InsufficientFundsError
from gateway.ports.model_provider_port import (
    HostedAccessDeniedError,
    HostedCredential,
    HostedModels,
    ModelProviderPort,
)
from gateway.services.inference import DeploymentCharge
from gateway.services.provider_kwargs import ResolvedProvider, resolve_provider_selector

ORGANIZATION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
WORKSPACE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")

# Every provider these tests name; cleared so a developer's own shell key cannot
# make `credential_ladder_exhausted` answer False and quietly skip the port.
_PROVIDER_ENV_VARS = (
    "OPENAI_API_KEY",
    "TOGETHER_API_KEY",
    "VLLM_API_KEY",
    "LM_STUDIO_API_KEY",
    "CASCADIA_API_KEY",
    "OTARI_API_KEY",
)


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
        # ``BaseException`` and not ``Exception``: one case here raises
        # ``asyncio.CancelledError``, which the call site must deliberately not
        # catch, and typing this narrowly would make that case unexpressible.
        error: BaseException | None = None,
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

    async def get_hosted_models(self, *, organization_id: uuid.UUID | None) -> HostedModels:
        raise AssertionError("dispatch must not ask for the hosted roster")


def _plain_build_port() -> ModelProviderPort:
    """The adapter a build with no overlay actually runs, via the composition root."""
    return build_container(workspace_listener=None).resolve(ModelProviderPort, None)


def _plain_build_billing() -> BillingPort:
    """The billing adapter a build with no overlay runs, via the composition root."""
    return build_container(workspace_listener=None).resolve(BillingPort, None)


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
        uow=None,
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
        billing=_plain_build_billing(),
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
        billing=_plain_build_billing(),
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
        billing=_plain_build_billing(),
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


@pytest.mark.parametrize(
    "selector",
    [
        "ollama:llama3",
        "llamacpp:local",
        "vertexai:gemini-2.5-pro",
        # These four declare a credential env var that any-llm never insists on,
        # so the declaration alone cannot tell them apart from a keyed provider.
        "vllm:my-model",
        "lmstudio:my-model",
        "cascadia:my-model",
        "otari:my-model",
        # Ambient cloud credentials this gateway cannot see: an instance profile
        # or SSO session serves these with nothing configured in otari at all.
        "bedrock:anthropic.claude-sonnet-4-20250514-v1:0",
        "sagemaker:my-endpoint",
    ],
)
@pytest.mark.asyncio
async def test_a_provider_needing_no_credential_is_not_asked_about(selector: str) -> None:
    """any-llm calls these without a key, so an empty kwargs is not a missing one.

    A bare ``ollama:llama3`` works today against a local backend, and a bare
    ``bedrock:...`` works today off an EC2 instance profile. Reporting either as
    unserved would hand a working request to a fleet that might answer it from
    somewhere else, and self-hosting is a first-class path *upstream* of this
    port (``ports/model_provider_port.py``).
    """
    port = RecordingPort(credential=HostedCredential(api_key="hosted", api_base=None, response_provider="openai"))
    resolved = resolve_provider_selector(GatewayConfig(), selector)
    assert resolved.kwargs == {}
    result = await _dispatch(_ctx(resolved_provider=resolved), port, selector=selector)
    assert result is resolved
    assert port.calls == []


@pytest.mark.parametrize(
    ("provider_config", "label"),
    [
        ({"client_args": {"timeout": 60}}, "transport tuning only"),
        ({"api_key": None}, "a key written with no value"),
        ({"api_key": ""}, "an empty key"),
        ({"client_args": {"timeout": 60}, "api_key": None}, "both"),
    ],
)
@pytest.mark.asyncio
async def test_an_instance_with_no_usable_credential_still_reaches_the_port(
    provider_config: dict[str, Any], label: str
) -> None:
    """A described instance is not a credentialed one.

    ``get_provider_kwargs`` returns a non-empty dict for each of these, so testing
    the dict for emptiness would report the ladder as having answered when it
    found a provider entry and no way to call it. any-llm would then fail its own
    missing-key check on a candidate this build could have served.
    """
    config = GatewayConfig(providers={"openai": provider_config})
    resolved = resolve_provider_selector(config, "openai:gpt-4o")
    assert resolved.kwargs, f"guard: {label} must resolve a non-empty kwargs dict"
    port = RecordingPort(credential=HostedCredential(api_key="hosted-key", api_base=None, response_provider="openai"))

    result = await pipeline.resolve_dispatch_provider(
        _ctx(resolved_provider=resolved),
        config,
        "openai:gpt-4o",
        adapter=chat._ADAPTER,
        model_provider=port,
        billing=_plain_build_billing(),
    )

    assert len(port.calls) == 1, f"the port was not asked despite {label}"
    assert result.kwargs == {"api_key": "hosted-key"}


@pytest.mark.asyncio
async def test_client_args_alone_does_not_shadow_a_real_key() -> None:
    """The converse: transport tuning beside a real key is still a credentialed rung."""
    config = GatewayConfig(providers={"openai": {"api_key": "sk-real", "client_args": {"timeout": 60}}})
    resolved = resolve_provider_selector(config, "openai:gpt-4o")
    port = RecordingPort(credential=HostedCredential(api_key="hosted", api_base=None, response_provider="openai"))

    result = await pipeline.resolve_dispatch_provider(
        _ctx(resolved_provider=resolved),
        config,
        "openai:gpt-4o",
        adapter=chat._ADAPTER,
        model_provider=port,
        billing=_plain_build_billing(),
    )

    assert port.calls == []
    assert result.kwargs["api_key"] == "sk-real"


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
    port = RecordingPort(credential=HostedCredential(api_key="hosted-key", api_base=None, response_provider="openai"))
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
    port = RecordingPort(credential=HostedCredential(api_key="hosted-key", api_base=None, response_provider="together"))
    result = await pipeline.resolve_dispatch_provider(
        _ctx(resolved_provider=resolved),
        config,
        "fast",
        adapter=chat._ADAPTER,
        model_provider=port,
        billing=_plain_build_billing(),
    )
    assert result.alias == "fast"
    assert result.instance == "together"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _capture_settlement(monkeypatch: pytest.MonkeyPatch) -> tuple[list[object], list[dict[str, Any]]]:
    """Record the refund and the activity-log row a refusal owes, without a database."""
    released: list[object] = []
    rejections: list[dict[str, Any]] = []

    async def _record_release(ctx: pipeline.RequestContext) -> None:
        released.append(ctx)

    async def _record_rejection(**kwargs: Any) -> None:
        rejections.append(kwargs)

    monkeypatch.setattr(pipeline, "release_reservation", _record_release)
    monkeypatch.setattr(pipeline, "log_gateway_rejection", _record_rejection)
    return released, rejections


@pytest.mark.asyncio
async def test_access_denied_becomes_a_403_that_names_no_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    released, rejections = _capture_settlement(monkeypatch)
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
    # The hold taken by the preamble is refunded before the refusal surfaces,
    # and the drop is countable in the activity log rather than invisible.
    assert len(released) == 1
    assert [row["status_code"] for row in rejections] == [403]
    assert rejections[0]["provider"] == "openai"
    assert rejections[0]["detail"] == detail


@pytest.mark.asyncio
async def test_a_refused_alias_does_not_spell_its_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """An alias hides its target, and a refusal is not an exception to that."""
    _capture_settlement(monkeypatch)
    config = GatewayConfig(aliases={"fast": "openai:gpt-4o"})
    resolved = resolve_provider_selector(config, "fast")
    port = RecordingPort(error=HostedAccessDeniedError("AcmeHostedAdapter: not entitled"))

    with pytest.raises(HTTPException) as exc_info:
        await pipeline.resolve_dispatch_provider(
            _ctx(resolved_provider=resolved),
            config,
            "fast",
            adapter=chat._ADAPTER,
            model_provider=port,
            billing=_plain_build_billing(),
        )

    detail = str(exc_info.value.detail)
    assert "fast" in detail
    assert "openai:gpt-4o" not in detail


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("adapter blew up"),
        TimeoutError("upstream credential service timed out"),
        ValueError("malformed adapter response"),
    ],
)
@pytest.mark.asyncio
async def test_an_adapter_failure_refunds_and_becomes_a_502(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """Any failure but the port's own refusal still owes the reservation back.

    An overlay adapter resolving credentials over the network fails the ordinary
    ways: a timeout, a connection reset, a database error. `resolve_dispatch_provider`
    is called outside any `try` that would catch one (`chat.py`, `responses.py`, and
    the non-streaming half of `messages.py`; the streaming half's `try` catches
    `HTTPException` alone), and `gateway.main` registers handlers only for
    `TenancyError` and `RequestValidationError`, so nothing above this refunds. The
    hold the preamble took would sit in `users.reserved` until the budget resets.
    """
    released, rejections = _capture_settlement(monkeypatch)
    port = RecordingPort(error=failure)

    with pytest.raises(HTTPException) as exc_info:
        await _dispatch(_ctx(resolved_provider=_uncredentialed()), port)

    assert exc_info.value.status_code == 502
    assert pipeline.HOSTED_CREDENTIAL_UNUSABLE_DETAIL in str(exc_info.value.detail)
    # Non-leaky: the adapter's own wording never reaches the caller.
    assert str(failure) not in str(exc_info.value.detail)
    assert len(released) == 1
    assert [row["status_code"] for row in rejections] == [502]


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled request stays cancelled: it is not a hosted-inference failure.

    `asyncio.CancelledError` derives from `BaseException`, so the `except Exception`
    guard above lets it through. Turning a disconnect into a 502 would both lie about
    what happened and, worse, mark the task as handled.
    """
    released, _ = _capture_settlement(monkeypatch)
    port = RecordingPort(error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await _dispatch(_ctx(resolved_provider=_uncredentialed()), port)

    assert released == []


@pytest.mark.asyncio
async def test_unknown_response_provider_becomes_a_502(monkeypatch: pytest.MonkeyPatch) -> None:
    released, rejections = _capture_settlement(monkeypatch)
    port = RecordingPort(
        credential=HostedCredential(api_key="hosted-key", api_base=None, response_provider="not-a-provider")
    )

    with pytest.raises(HTTPException) as exc_info:
        await _dispatch(_ctx(resolved_provider=_uncredentialed()), port)

    assert exc_info.value.status_code == 502
    assert pipeline.HOSTED_CREDENTIAL_UNUSABLE_DETAIL in str(exc_info.value.detail)
    assert "not-a-provider" not in str(exc_info.value.detail)
    assert len(released) == 1
    assert [row["status_code"] for row in rejections] == [502]


# ---------------------------------------------------------------------------
# The fresh-resolution branch reaches the port too
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_resolution_also_reaches_the_port() -> None:
    """``ctx.resolved_provider`` unset (the gate could not parse the selector) still asks."""
    port = RecordingPort(credential=HostedCredential(api_key="hosted-key", api_base=None, response_provider="openai"))
    result = await _dispatch(_ctx(resolved_provider=None), port)
    assert result.kwargs == {"api_key": "hosted-key"}
    assert len(port.calls) == 1


@pytest.mark.asyncio
async def test_the_plain_build_serves_no_hosted_provider() -> None:
    assert await _plain_build_port().get_hosted_models(organization_id=ORGANIZATION_ID) == {}


# ---------------------------------------------------------------------------
# Deployment-paid: the funding gate before dispatch
# ---------------------------------------------------------------------------


class RecordingBilling:
    """A stand-in for an overlay's billing adapter, recording what it was asked."""

    def __init__(self, *, refuse: BaseException | None = None, charge_error: Exception | None = None) -> None:
        self.refuse = refuse
        self.charge_error = charge_error
        self.calls: list[tuple[Any, ...]] = []

    async def apply_due_credit(self, *, organization_id: uuid.UUID) -> None:
        self.calls.append(("apply_due_credit", organization_id))

    async def require_funds_on_deposit(self, *, organization_id: uuid.UUID) -> None:
        self.calls.append(("require_funds_on_deposit", organization_id))
        if self.refuse is not None:
            raise self.refuse

    async def require_unheld_funds(self, *, organization_id: uuid.UUID) -> None:
        raise AssertionError("the completion path places no holds")

    async def hold(self, *, organization_id: uuid.UUID, amount: Decimal) -> None:
        raise AssertionError("the completion path places no holds")

    async def release_hold(self, *, organization_id: uuid.UUID, amount: Decimal) -> None:
        raise AssertionError("the completion path places no holds")

    async def charge(
        self,
        *,
        organization_id: uuid.UUID,
        amount: Decimal,
        description: str | None = None,
        actor_user_id: uuid.UUID | None = None,
        api_key_id: str | None = None,
    ) -> None:
        self.calls.append(("charge", organization_id, amount, description, actor_user_id, api_key_id))
        if self.charge_error is not None:
            raise self.charge_error


def _hosted_port() -> RecordingPort:
    return RecordingPort(credential=HostedCredential(api_key="hosted-key", api_base=None, response_provider="openai"))


async def _dispatch_billed(
    ctx: pipeline.RequestContext,
    port: ModelProviderPort,
    billing: BillingPort,
    *,
    adapter: Any = chat._ADAPTER,
    config: GatewayConfig | None = None,
) -> ResolvedProvider:
    return await pipeline.resolve_dispatch_provider(
        ctx, config or GatewayConfig(), "openai:gpt-4o", adapter=adapter, model_provider=port, billing=billing
    )


@pytest.mark.asyncio
async def test_a_hosted_credential_is_gated_on_funds_and_armed_for_a_charge() -> None:
    billing = RecordingBilling()
    ctx = _ctx(resolved_provider=_uncredentialed())

    result = await _dispatch_billed(ctx, _hosted_port(), cast(BillingPort, billing))

    assert result.deployment_paid is True
    # Any credit owed is granted before the organization's funds are judged.
    assert billing.calls == [
        ("apply_due_credit", ORGANIZATION_ID),
        ("require_funds_on_deposit", ORGANIZATION_ID),
    ]
    assert ctx.deployment_charge is not None
    assert ctx.deployment_charge.organization_id == ORGANIZATION_ID
    assert ctx.deployment_charge.billing is billing


@pytest.mark.asyncio
async def test_a_credentialed_candidate_never_reaches_billing() -> None:
    """A config, stored or organization key is paid for by its owner, so nothing is gated or charged."""
    config = GatewayConfig(providers={"openai": {"api_key": "sk-from-config"}})
    billing = RecordingBilling()
    ctx = _ctx(resolved_provider=resolve_provider_selector(config, "openai:gpt-4o"))

    result = await _dispatch_billed(ctx, _hosted_port(), cast(BillingPort, billing), config=config)

    assert result.deployment_paid is False
    assert billing.calls == []
    assert ctx.deployment_charge is None


@pytest.mark.asyncio
async def test_a_candidate_the_port_does_not_serve_never_reaches_billing() -> None:
    billing = RecordingBilling()
    ctx = _ctx(resolved_provider=_uncredentialed())

    result = await _dispatch_billed(ctx, RecordingPort(credential=None), cast(BillingPort, billing))

    assert result.deployment_paid is False
    assert billing.calls == []
    assert ctx.deployment_charge is None


@pytest.mark.parametrize(
    "adapter", [chat._ADAPTER, messages._ADAPTER, responses._ADAPTER], ids=["chat", "messages", "responses"]
)
@pytest.mark.asyncio
async def test_insufficient_funds_refunds_and_becomes_a_402(monkeypatch: pytest.MonkeyPatch, adapter: Any) -> None:
    released, rejections = _capture_settlement(monkeypatch)
    billing = RecordingBilling(refuse=InsufficientFundsError(Decimal("0")))
    ctx = _ctx(resolved_provider=_uncredentialed())

    with pytest.raises(HTTPException) as exc_info:
        await _dispatch_billed(ctx, _hosted_port(), cast(BillingPort, billing), adapter=adapter)

    refusal = exc_info.value
    detail: Any = refusal.detail
    assert refusal.status_code == 402
    assert (refusal.headers or {}).get(ERROR_CODE_HEADER) == INSUFFICIENT_FUNDS
    message = "Insufficient funds ($0.00) to serve this request."
    if adapter is messages._ADAPTER:
        assert detail == {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    else:
        assert detail == message
    assert len(released) == 1
    assert [(row["status_code"], row["provider"], row["model"]) for row in rejections] == [(402, "openai", "gpt-4o")]
    assert ctx.deployment_charge is None


@pytest.mark.asyncio
async def test_a_billing_failure_refunds_and_becomes_a_503(monkeypatch: pytest.MonkeyPatch) -> None:
    released, rejections = _capture_settlement(monkeypatch)
    billing = RecordingBilling(refuse=RuntimeError("wallet database is down"))

    with pytest.raises(HTTPException) as exc_info:
        await _dispatch_billed(_ctx(resolved_provider=_uncredentialed()), _hosted_port(), cast(BillingPort, billing))

    assert exc_info.value.status_code == 503
    assert "wallet" not in str(exc_info.value.detail)
    assert len(released) == 1
    assert [row["status_code"] for row in rejections] == [503]


@pytest.mark.asyncio
async def test_an_admission_that_does_not_commit_becomes_a_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """The commit lands any credit admission granted, so a request is not served on one that rolled back."""
    released, rejections = _capture_settlement(monkeypatch)

    async def _commit_failed(_session: object) -> bool:
        return False

    async def _no_pricing_check(*_args: object) -> None:
        return None

    monkeypatch.setattr(pipeline, "release_session", _commit_failed)
    # A session would otherwise send the unpriced-upstream warning to the database.
    monkeypatch.setattr(pipeline, "_warn_if_hosted_upstream_is_unpriced", _no_pricing_check)
    billing = RecordingBilling()
    ctx = _ctx(resolved_provider=_uncredentialed())
    ctx.db = cast(AsyncSession, object())

    with pytest.raises(HTTPException) as exc_info:
        await _dispatch_billed(ctx, _hosted_port(), cast(BillingPort, billing))

    assert exc_info.value.status_code == 503
    assert len(released) == 1
    assert [row["status_code"] for row in rejections] == [503]
    assert ctx.deployment_charge is None


@pytest.mark.asyncio
async def test_the_plain_build_gates_nothing() -> None:
    """With the core's billing adapter a deployment-paid request is admitted as it always was."""
    ctx = _ctx(resolved_provider=_uncredentialed())
    result = await _dispatch_billed(ctx, _hosted_port(), _plain_build_billing())
    assert result.deployment_paid is True
    assert result.kwargs == {"api_key": "hosted-key"}


# ---------------------------------------------------------------------------
# Deployment-paid: the charge at settlement
# ---------------------------------------------------------------------------


class _SavepointSession:
    """Just enough session for the charge: savepoints, counted by how they ended."""

    def __init__(self) -> None:
        self.released = 0
        self.rolled_back = 0

    @asynccontextmanager
    async def begin_nested(self) -> AsyncIterator[None]:
        try:
            yield
        except BaseException:
            self.rolled_back += 1
            raise
        self.released += 1


@pytest.fixture
def gateway_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """The gateway logger does not propagate, so caplog sees it only with its handler attached."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


def _charge(billing: RecordingBilling) -> DeploymentCharge:
    return DeploymentCharge(
        billing=cast(BillingPort, billing),
        organization_id=ORGANIZATION_ID,
        api_key_id="key-1",
        actor_user_id=WORKSPACE_ID,
        request_id="req-1",
    )


@pytest.mark.asyncio
async def test_a_settled_cost_is_charged_in_a_savepoint() -> None:
    billing = RecordingBilling()
    db = _SavepointSession()

    await pipeline.charge_deployment_paid(
        _charge(billing), cast(Any, db), Decimal("0.0125"), provider="openai", model="gpt-4o", expects_cost=True
    )

    assert billing.calls == [("charge", ORGANIZATION_ID, Decimal("0.0125"), "openai:gpt-4o", WORKSPACE_ID, "key-1")]
    assert db.released == 1


@pytest.mark.parametrize("cost", [Decimal(0), Decimal("-1")])
@pytest.mark.asyncio
async def test_nothing_is_charged_for_no_cost(cost: Decimal) -> None:
    billing = RecordingBilling()
    await pipeline.charge_deployment_paid(
        _charge(billing), cast(Any, _SavepointSession()), cost, provider="openai", model="gpt-4o", expects_cost=True
    )
    assert billing.calls == []


@pytest.mark.asyncio
async def test_an_unpriced_settlement_is_not_charged_and_says_so(gateway_logs: pytest.LogCaptureFixture) -> None:
    billing = RecordingBilling()

    await pipeline.charge_deployment_paid(
        _charge(billing), cast(Any, _SavepointSession()), None, provider="openai", model="gpt-4o", expects_cost=True
    )

    assert billing.calls == []
    warnings = [r.getMessage() for r in gateway_logs.records if r.levelno == logging.WARNING]
    assert any("gpt-4o" in message and "not charged" in message for message in warnings), warnings
    # The provider is left out of the log line; the request id finds the usage row that names it.
    assert all("openai" not in message for message in warnings), warnings


@pytest.mark.asyncio
async def test_a_settlement_owing_nothing_is_not_warned_about(gateway_logs: pytest.LogCaptureFixture) -> None:
    """A provider failure with no usage settles at no cost, which is not a pricing gap."""
    await pipeline.charge_deployment_paid(
        _charge(RecordingBilling()),
        cast(Any, _SavepointSession()),
        None,
        provider="openai",
        model="gpt-4o",
        expects_cost=False,
    )
    assert gateway_logs.records == []


@pytest.mark.asyncio
async def test_a_failed_charge_is_logged_and_does_not_raise(gateway_logs: pytest.LogCaptureFixture) -> None:
    billing = RecordingBilling(charge_error=RuntimeError("ledger down"))
    db = _SavepointSession()

    await pipeline.charge_deployment_paid(
        _charge(billing), cast(Any, db), Decimal("1"), provider="openai", model="gpt-4o", expects_cost=True
    )

    # Its savepoint is rolled back, so nothing it wrote reaches the settlement's commit.
    assert db.rolled_back == 1
    errors = [r.getMessage() for r in gateway_logs.records if r.levelno == logging.ERROR]
    assert any(str(ORGANIZATION_ID) in message and "req-1" in message for message in errors), errors
    assert all("ledger down" not in message for message in errors)


@pytest.mark.asyncio
async def test_a_request_that_is_not_deployment_paid_is_never_charged() -> None:
    db = _SavepointSession()
    await pipeline.charge_deployment_paid(
        None, cast(Any, db), Decimal("1"), provider="openai", model="gpt-4o", expects_cost=True
    )
    assert db.released == 0
