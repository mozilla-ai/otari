"""The core answer behind ``HostedGuardrailPort``: this build hosts no guardrails."""

import asyncio
import uuid
from pathlib import Path

import pytest

from gateway.adapters.hosted_guardrail_adapter import NullHostedGuardrailAdapter
from gateway.container import PortShapeError, build_container
from gateway.ports.hosted_guardrail_port import (
    HostedGuardrailPort,
    HostedGuardrailUnavailableError,
    HostedGuardrailUnfundedError,
)

ORGANIZATION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


def _adapter() -> HostedGuardrailPort:
    return build_container().resolve(HostedGuardrailPort, None)


def test_the_plain_build_binds_the_null_adapter() -> None:
    assert isinstance(_adapter(), NullHostedGuardrailAdapter)


def test_the_null_adapter_offers_nothing_to_pick() -> None:
    adapter = _adapter()

    assert asyncio.run(adapter.list_hosted_guardrails(organization_id=ORGANIZATION_ID)) == []
    assert asyncio.run(adapter.list_hosted_guardrails(organization_id=None)) == []
    got = asyncio.run(adapter.get_hosted_guardrail(organization_id=ORGANIZATION_ID, hosted_guardrail_id=uuid.uuid4()))
    assert got is None


def test_the_null_adapter_refuses_to_evaluate() -> None:
    with pytest.raises(HostedGuardrailUnavailableError):
        asyncio.run(
            _adapter().evaluate(
                organization_id=ORGANIZATION_ID,
                workspace_id=None,
                hosted_guardrail_id=uuid.uuid4(),
                text="hello",
                validate_kwargs={},
                idempotency_key="request:mandate",
            )
        )


def test_unfunded_is_a_kind_of_unavailable_carrying_no_amount_in_its_message() -> None:
    error = HostedGuardrailUnfundedError()

    assert isinstance(error, HostedGuardrailUnavailableError)
    assert "$" not in str(error)


def test_a_bootstrap_binding_a_hosted_guardrail_adapter_of_the_wrong_shape_fails_at_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "stale_guardrail_bootstrap.py").write_text(
        """
from gateway.ports.hosted_guardrail_port import HostedGuardrailPort


class StaleHostedGuardrails:
    def __init__(self, session):
        self.session = session

    async def list_hosted_guardrails(self, **kwargs):
        return []


def register(container):
    container.bind(HostedGuardrailPort, StaleHostedGuardrails)
"""
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(PortShapeError, match="StaleHostedGuardrails, bound to HostedGuardrailPort, lacks evaluate"):
        build_container("stale_guardrail_bootstrap:register")
