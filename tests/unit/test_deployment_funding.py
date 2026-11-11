"""The inference domain's funding rule for a deployment-paid request.

Admission grants owed credit before judging funds and lets the port's refusal
through; settlement charges only a positive cost, and names an unpriced one in
the log without the provider, which the usage row carries.
"""

import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any, cast

import pytest
from any_llm import LLMProvider

from gateway.log_config import logger as gateway_logger
from gateway.ports.billing_port import NO_TOOLS, BilledModel, BillingPort, InsufficientFundsError
from gateway.services.inference import (
    DeploymentCharge,
    WorkspaceFunding,
    admit_deployment_paid,
    billed_model,
    deferred_charge,
)

ORGANIZATION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
IDENTITY_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
BILLED = BilledModel(provider="openai", model="gpt-4o", open_weights=False)


class _Billing:
    def __init__(self, *, refuse: bool = False, limit: int | None = None, fail: bool = False) -> None:
        self.refuse = refuse
        self.limit = limit
        self.fail = fail
        self.calls: list[tuple[Any, ...]] = []

    async def apply_due_credit(self, *, organization_id: uuid.UUID) -> None:
        self.calls.append(("apply_due_credit", organization_id))

    async def require_funds_on_deposit(
        self,
        *,
        organization_id: uuid.UUID,
        billed: BilledModel | None,
        tools: frozenset[str],
        actor_user_id: uuid.UUID | None,
    ) -> None:
        self.calls.append(("require_funds_on_deposit", organization_id, billed, tools, actor_user_id))
        if self.refuse:
            raise InsufficientFundsError(Decimal(0))

    async def has_funds_on_deposit(
        self,
        *,
        organization_id: uuid.UUID,
        billed: BilledModel | None,
        tools: frozenset[str],
        actor_user_id: uuid.UUID | None,
    ) -> bool:
        self.calls.append(("has_funds_on_deposit", organization_id, billed, tools, actor_user_id))
        if self.fail:
            raise RuntimeError("billing down")
        return not self.refuse

    async def concurrency_limit(
        self, *, organization_id: uuid.UUID, billed: BilledModel, actor_user_id: uuid.UUID | None
    ) -> int | None:
        self.calls.append(("concurrency_limit", organization_id, billed, actor_user_id))
        return self.limit

    async def charge(self, **kwargs: Any) -> None:
        self.calls.append(("charge", kwargs))


@pytest.fixture
def gateway_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """The gateway logger does not propagate, so caplog sees it only with its handler attached."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


async def _admit(billing: _Billing, user_id: str | None = str(IDENTITY_ID)) -> DeploymentCharge:
    charge = await admit_deployment_paid(
        cast(BillingPort, billing),
        deployment_paid=True,
        organization_id=ORGANIZATION_ID,
        billed=BILLED,
        api_key_id="key-1",
        user_id=user_id,
        request_id="req-1",
    )
    assert charge is not None
    return charge


@pytest.mark.asyncio
async def test_admission_grants_credit_then_checks_funds() -> None:
    billing = _Billing()
    charge = await _admit(billing)
    assert billing.calls == [
        ("apply_due_credit", ORGANIZATION_ID),
        ("require_funds_on_deposit", ORGANIZATION_ID, BILLED, frozenset(), IDENTITY_ID),
        ("concurrency_limit", ORGANIZATION_ID, BILLED, IDENTITY_ID),
    ]
    assert (charge.organization_id, charge.api_key_id, charge.actor_user_id, charge.request_id, charge.billed) == (
        ORGANIZATION_ID,
        "key-1",
        IDENTITY_ID,
        "req-1",
        BILLED,
    )


@pytest.mark.asyncio
async def test_admission_lets_the_ports_refusal_through() -> None:
    with pytest.raises(InsufficientFundsError):
        await _admit(_Billing(refuse=True))


@pytest.mark.parametrize("user_id", [None, "a-free-form-user"])
@pytest.mark.asyncio
async def test_a_user_id_that_names_no_identity_attributes_to_no_one(user_id: str | None) -> None:
    assert (await _admit(_Billing(), user_id=user_id)).actor_user_id is None


@pytest.mark.parametrize(
    ("cost", "due"),
    [(Decimal("0.25"), Decimal("0.25")), (Decimal(0), None), (Decimal("-1"), None), (None, None)],
)
@pytest.mark.asyncio
async def test_only_a_positive_cost_is_due(cost: Decimal | None, due: Decimal | None) -> None:
    charge = await _admit(_Billing())
    assert charge.amount_due(cost, model="gpt-4o", expects_cost=False) == due


@pytest.mark.asyncio
async def test_an_unpriced_settlement_is_warned_about_without_its_provider(
    gateway_logs: pytest.LogCaptureFixture,
) -> None:
    charge = await _admit(_Billing())
    assert charge.amount_due(None, model="gpt-4o", expects_cost=True) is None
    (message,) = [record.getMessage() for record in gateway_logs.records]
    assert "gpt-4o" in message
    assert str(ORGANIZATION_ID) in message
    assert "req-1" in message


@pytest.mark.asyncio
async def test_a_charge_is_attributed_to_the_request() -> None:
    billing = _Billing()
    charge = await _admit(billing)
    await charge.charge(Decimal("0.25"), tools=NO_TOOLS, description="openai:gpt-4o")
    assert billing.calls[-1] == (
        "charge",
        {
            "organization_id": ORGANIZATION_ID,
            "amount": Decimal("0.25"),
            "billed": BILLED,
            "tools": NO_TOOLS,
            "description": "openai:gpt-4o",
            "actor_user_id": IDENTITY_ID,
            "api_key_id": "key-1",
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("deployment_paid", "organization_id"), [(False, ORGANIZATION_ID), (True, None)])
async def test_a_request_the_deployment_does_not_pay_for_is_admitted_with_nothing_to_charge(
    deployment_paid: bool, organization_id: uuid.UUID | None
) -> None:
    billing = _Billing(refuse=True)
    charge = await admit_deployment_paid(
        cast(BillingPort, billing),
        deployment_paid=deployment_paid,
        organization_id=organization_id,
        billed=BILLED,
        api_key_id="key-1",
        user_id=None,
        request_id="req-1",
    )
    assert charge is None
    assert billing.calls == []


class _Savepoint:
    def __init__(self) -> None:
        self.entered = 0
        self.rolled_back = 0

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[None]:
        self.entered += 1
        try:
            yield
        except Exception:
            self.rolled_back += 1
            raise


@pytest.mark.asyncio
async def test_settlement_charges_a_positive_cost_inside_the_savepoint() -> None:
    billing = _Billing()
    charge = await _admit(billing)
    savepoint = _Savepoint()
    await charge.settle(
        Decimal("0.25"),
        tools=NO_TOOLS,
        model="gpt-4o",
        description="openai:gpt-4o",
        expects_cost=True,
        savepoint=savepoint,
    )
    assert savepoint.entered == 1
    assert billing.calls[-1][0] == "charge"


@pytest.mark.asyncio
async def test_settlement_with_nothing_due_opens_no_savepoint() -> None:
    billing = _Billing()
    charge = await _admit(billing)
    savepoint = _Savepoint()
    await charge.settle(
        Decimal(0), tools=NO_TOOLS, model="gpt-4o", description="openai:gpt-4o", expects_cost=True, savepoint=savepoint
    )
    assert savepoint.entered == 0
    assert all(call[0] != "charge" for call in billing.calls)


class _FailingBilling(_Billing):
    async def charge(self, **kwargs: Any) -> None:
        raise RuntimeError("ledger unavailable: secret row data")


@pytest.mark.asyncio
async def test_a_failed_charge_is_rolled_back_and_logged_without_its_text(
    gateway_logs: pytest.LogCaptureFixture,
) -> None:
    charge = await _admit(_FailingBilling())
    savepoint = _Savepoint()
    await charge.settle(
        Decimal("0.25"),
        tools=NO_TOOLS,
        model="gpt-4o",
        description="openai:gpt-4o",
        expects_cost=True,
        savepoint=savepoint,
    )
    assert savepoint.rolled_back == 1
    (record,) = gateway_logs.records
    assert record.levelno == logging.ERROR
    message = record.getMessage()
    assert "RuntimeError" in message
    assert "secret row data" not in message
    assert record.exc_info is None


@pytest.mark.asyncio
async def test_a_gateway_tools_admission_bills_no_model_and_names_its_tools() -> None:
    billing = _Billing()
    tools = frozenset({"otari:web_search", "otari:web_fetch"})
    charge = await admit_deployment_paid(
        cast(BillingPort, billing),
        deployment_paid=True,
        organization_id=ORGANIZATION_ID,
        billed=None,
        api_key_id="key-1",
        user_id=None,
        request_id="req-1",
        tools=tools,
    )
    assert charge is not None
    assert charge.billed is None
    # A tool's call is counted under no model's concurrency limit.
    assert charge.concurrency_limit is None
    assert all(call[0] != "concurrency_limit" for call in billing.calls)
    await charge.charge(Decimal("0.01"), tools=NO_TOOLS, description="otari_web_search")
    assert billing.calls[1] == ("require_funds_on_deposit", ORGANIZATION_ID, None, tools, None)
    assert billing.calls[-1][1]["billed"] is None


@pytest.mark.parametrize(
    ("provider", "model", "open_weights"),
    [
        ("groq", "openai/gpt-oss-120b", True),
        (LLMProvider.GROQ, "openai/gpt-oss-120b", True),
        ("openai", "gpt-oss-120b", True),
        ("gemini", "gemma-3-27b-it", True),
        ("deepinfra", "Qwen/Qwen3-Coder", True),
        ("together", "moonshotai/Kimi-K2", True),
        ("mistral", "a-model-no-catalog-lists", True),
        ("openai", "gpt-4o", False),
        ("openai", "o3-mini", False),
        ("openai", "a-model-no-catalog-lists", False),
        ("anthropic", "claude-sonnet-4", False),
        ("vertexai", "gemini-2.5-pro", False),
        ("xai", "grok-4", False),
        ("openrouter", "anthropic/claude-sonnet-4", False),
        ("openrouter", "x-ai/grok-4", False),
        ("deepinfra", "google/gemini-2.5-flash", False),
        ("bedrock", "anthropic.claude-sonnet-4-v1:0", False),
        ("bedrock", "us.anthropic.claude-sonnet-4-v1:0", False),
        ("bedrock", "meta.llama3-70b-instruct-v1:0", True),
        ("zai", "glm-4.6", True),
    ],
)
def test_a_model_is_open_weight_unless_it_is_closed(
    provider: str | LLMProvider, model: str, open_weights: bool
) -> None:
    assert billed_model(provider, model) == BilledModel(
        provider=LLMProvider(provider).value, model=model, open_weights=open_weights
    )


def test_a_deferred_job_is_charged_to_its_creator_on_the_model_it_requested() -> None:
    """A batch or video settled later bills its creator, its requested model and its implementation."""
    billing = cast(BillingPort, _Billing())

    charge = deferred_charge(
        billing,
        organization_id=ORGANIZATION_ID,
        api_key_id="key-1",
        creator_user_id=str(IDENTITY_ID),
        provider=LLMProvider.DEEPINFRA,
        model="Wan-AI/Wan2.1",
    )

    assert charge.actor_user_id == IDENTITY_ID
    assert charge.billed == BilledModel(provider="deepinfra", model="Wan-AI/Wan2.1", open_weights=True)
    assert charge.api_key_id == "key-1"


def test_a_deferred_job_whose_creator_names_no_identity_is_charged_to_no_one() -> None:
    charge = deferred_charge(
        cast(BillingPort, _Billing()),
        organization_id=ORGANIZATION_ID,
        api_key_id=None,
        creator_user_id="service-user",
        provider="openai",
        model="gpt-4o",
    )
    assert charge.actor_user_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 3])
async def test_a_models_admission_carries_its_organizations_concurrency_limit(limit: int | None) -> None:
    charge = await _admit(_Billing(limit=limit))
    assert charge.concurrency_limit == limit


def test_the_deployments_settings_decide_which_models_are_closed() -> None:
    from gateway.core.settings.inference import (
        DEFAULT_CLOSED_MODEL_FAMILIES,
        DEFAULT_CLOSED_MODEL_PROVIDERS,
        DEFAULT_OPEN_MODEL_FAMILIES,
    )
    from gateway.services.inference import configure_closed_models

    configure_closed_models(providers=["mistral"], closed_families=["kimi"], open_families=["mistral-small"])
    try:
        assert billed_model("mistral", "mistral-large").open_weights is False
        assert billed_model("mistral", "mistral-small-3").open_weights is True
        assert billed_model("together", "moonshotai/Kimi-K2").open_weights is False
        assert billed_model("anthropic", "claude-sonnet-4").open_weights is True
    finally:
        configure_closed_models(
            providers=DEFAULT_CLOSED_MODEL_PROVIDERS,
            closed_families=DEFAULT_CLOSED_MODEL_FAMILIES,
            open_families=DEFAULT_OPEN_MODEL_FAMILIES,
        )


WORKSPACE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
SANDBOX = frozenset({"otari:code_execution"})


def _funding(billing: _Billing, organization_id: uuid.UUID | None = ORGANIZATION_ID) -> WorkspaceFunding:
    async def organization_for(workspace_id: uuid.UUID) -> uuid.UUID | None:
        assert workspace_id == WORKSPACE_ID
        return organization_id

    return WorkspaceFunding(cast(BillingPort, billing), organization_for)


@pytest.mark.asyncio
async def test_a_workspace_is_funded_on_the_read_alone() -> None:
    billing = _Billing()
    assert await _funding(billing).funded(WORKSPACE_ID, str(IDENTITY_ID), SANDBOX)
    # Neither gate method: one grants credit and the other starts what a refusal starts.
    assert billing.calls == [("has_funds_on_deposit", ORGANIZATION_ID, None, SANDBOX, IDENTITY_ID)]


@pytest.mark.asyncio
async def test_a_workspace_whose_organization_has_no_funds_is_not_funded() -> None:
    assert not await _funding(_Billing(refuse=True)).funded(WORKSPACE_ID, None, SANDBOX)


@pytest.mark.asyncio
async def test_a_workspace_with_no_organization_is_funded_without_asking_the_port() -> None:
    billing = _Billing(refuse=True)
    assert await _funding(billing, organization_id=None).funded(WORKSPACE_ID, None, SANDBOX)
    assert billing.calls == []


@pytest.mark.asyncio
async def test_a_port_that_fails_reads_as_funded_and_is_logged(gateway_logs: pytest.LogCaptureFixture) -> None:
    assert await _funding(_Billing(fail=True)).funded(WORKSPACE_ID, None, SANDBOX)
    assert "BillingPort failed to read an organization's funds" in gateway_logs.text
    assert "billing down" not in gateway_logs.text
