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

from gateway.log_config import logger as gateway_logger
from gateway.ports.billing_port import BillingPort, InsufficientFundsError
from gateway.services.inference import DeploymentCharge, admit_deployment_paid

ORGANIZATION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
IDENTITY_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")


class _Billing:
    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse
        self.calls: list[tuple[Any, ...]] = []

    async def apply_due_credit(self, *, organization_id: uuid.UUID) -> None:
        self.calls.append(("apply_due_credit", organization_id))

    async def require_funds_on_deposit(self, *, organization_id: uuid.UUID) -> None:
        self.calls.append(("require_funds_on_deposit", organization_id))
        if self.refuse:
            raise InsufficientFundsError(Decimal(0))

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
    assert billing.calls == [("apply_due_credit", ORGANIZATION_ID), ("require_funds_on_deposit", ORGANIZATION_ID)]
    assert (charge.organization_id, charge.api_key_id, charge.actor_user_id, charge.request_id) == (
        ORGANIZATION_ID,
        "key-1",
        IDENTITY_ID,
        "req-1",
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
    await charge.charge(Decimal("0.25"), description="openai:gpt-4o")
    assert billing.calls[-1] == (
        "charge",
        {
            "organization_id": ORGANIZATION_ID,
            "amount": Decimal("0.25"),
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
        Decimal("0.25"), model="gpt-4o", description="openai:gpt-4o", expects_cost=True, savepoint=savepoint
    )
    assert savepoint.entered == 1
    assert billing.calls[-1][0] == "charge"


@pytest.mark.asyncio
async def test_settlement_with_nothing_due_opens_no_savepoint() -> None:
    billing = _Billing()
    charge = await _admit(billing)
    savepoint = _Savepoint()
    await charge.settle(Decimal(0), model="gpt-4o", description="openai:gpt-4o", expects_cost=True, savepoint=savepoint)
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
        Decimal("0.25"), model="gpt-4o", description="openai:gpt-4o", expects_cost=True, savepoint=savepoint
    )
    assert savepoint.rolled_back == 1
    (record,) = gateway_logs.records
    assert record.levelno == logging.ERROR
    message = record.getMessage()
    assert "RuntimeError" in message
    assert "secret row data" not in message
    assert record.exc_info is None
