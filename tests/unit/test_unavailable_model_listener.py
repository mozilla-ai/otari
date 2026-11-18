"""Which refusals the health checks switch a hosted model off for."""

from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import patch

import pytest

from gateway.services.health_checks import HealthCheckService, UnavailableModelListener
from gateway.services.inference import UnavailableModel


class _Service:
    def __init__(self) -> None:
        self.switched_off: list[tuple[str, str, str]] = []

    async def switch_off_unavailable(self, provider: str, model: str, refusal: str) -> bool:
        self.switched_off.append((provider, model, refusal))
        return True


def _listener(service: _Service) -> UnavailableModelListener:
    return UnavailableModelListener(lambda _uow: cast(HealthCheckService, service))


@pytest.fixture(autouse=True)
def _no_database() -> Any:
    with patch("gateway.services.health_checks._unavailable_models.create_unit_of_work", _unit_of_work):
        yield


@asynccontextmanager
async def _unit_of_work() -> Any:
    yield object()


@pytest.mark.asyncio
async def test_a_refusal_on_the_deployments_key_switches_the_model_off() -> None:
    service = _Service()

    await _listener(service).model_unavailable(
        UnavailableModel("together", "m", "not served", uses_deployment_credential=True)
    )

    assert service.switched_off == [("together", "m", "not served")]


@pytest.mark.asyncio
async def test_a_refusal_on_an_organizations_own_key_switches_nothing() -> None:
    service = _Service()

    await _listener(service).model_unavailable(
        UnavailableModel("together", "m", "not served", uses_deployment_credential=False)
    )

    assert service.switched_off == []
