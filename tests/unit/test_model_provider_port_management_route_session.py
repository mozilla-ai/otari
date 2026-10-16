"""Regression for otari#1266: a management-plane route's model-provider port must
share the route's own database session, not open a second, independent one.
"""

from typing import Annotated, cast

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes.organization_pricing import get_organization_pricing_service
from gateway.core import database
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.services.organization_pricing_service import OrganizationPricingService


class _FakeSession:
    """A distinguishable stand-in for ``AsyncSession``; only its identity matters here."""

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _NullModelProviderPort:
    """A do-nothing port; the stub container below only records which session built it."""


class _RecordingContainer:
    """Stands in for the composition-root container, recording ``resolve``'s session and Unit of Work."""

    def __init__(self) -> None:
        self.resolved_with: list[AsyncSession | None] = []
        self.units_of_work: list[UnitOfWork | None] = []

    def resolve(
        self, port: type[ModelProviderPort], session: AsyncSession | None, *, uow: UnitOfWork | None = None
    ) -> ModelProviderPort:
        self.resolved_with.append(session)
        self.units_of_work.append(uow)
        return cast(ModelProviderPort, _NullModelProviderPort())


def test_organization_pricing_service_shares_its_own_request_session(monkeypatch: pytest.MonkeyPatch) -> None:
    # A fresh fake per call, so a route that opens two sessions is caught by identity.
    monkeypatch.setattr(database, "_SessionLocal", lambda: _FakeSession())

    container = _RecordingContainer()
    app = FastAPI()
    app.state.container = container
    app.state.config = GatewayConfig(database_url="sqlite:///./test.db")

    @app.get("/probe")
    async def probe(
        service: Annotated[OrganizationPricingService, Depends(get_organization_pricing_service)],
    ) -> dict[str, bool]:
        [uow] = container.units_of_work
        return {
            "shared": len(container.resolved_with) == 1
            and container.resolved_with[0] is service.db
            and uow is not None
            and uow._session is service.db
        }

    response = TestClient(app).get("/probe")

    assert response.status_code == 200
    assert response.json() == {"shared": True}
