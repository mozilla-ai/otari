"""Hosted providers declared in the ``hosted_providers:`` block, against a real database.

The validator is pinned in ``tests/unit/test_hosted_providers_config.py``. This
file pins the merge: a configured provider is listed, served, discovered and
given a roster like a stored one; the API refuses to change it; a stored row of
the same provider wins over it, and boot says so. Model discovery is stubbed,
and the provider call is captured at the any-llm boundary and never made.
"""

import logging
import uuid
from collections.abc import Callable, Generator, Iterator
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.providers_exceptions import HostedProviderReadOnlyError
from gateway.log_config import logger as gateway_logger
from gateway.models.pricing import ModelPricing
from gateway.models.providers import HostedProvider, HostedProviderModel
from gateway.repositories.pricing import ModelPricingRepository, OrganizationModelPricingRepository
from gateway.repositories.providers import (
    HostedProviderModelRepository,
    HostedProviderRepository,
    ProviderEndpointRepository,
)
from gateway.repositories.tenancy import OrganizationRepository, OrgProviderKeyRepository
from gateway.schemas.providers import (
    HostedModelCreateRequest,
    HostedModelUpdateRequest,
    HostedProviderCreateRequest,
    HostedProviderUpdateRequest,
)
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.pricing import DeploymentPricingService
from gateway.services.providers import HostedProviderService
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client

pytestmark = pytest.mark.asyncio

CONFIG_KEY = "sk-live-config-9876"
STORED_KEY = "sk-live-stored-1234"
PATH = f"{API_ROOT}/hosted-providers"
DIAL = "gateway.services.providers._hosted_provider_service.test_provider_credentials"


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A secret box to store under, and no ambient key that would credential the ladder first."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield


def _discovery(monkeypatch: pytest.MonkeyPatch, *models: str, unsupported: bool = False) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    async def _stub(impl_name: str, **kwargs: object) -> ProviderDiscovery:
        calls.append({"provider": impl_name, **kwargs})
        if unsupported:
            return ProviderDiscovery(provider=impl_name, models=[], error="cannot list", discovery_unsupported=True)
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id=m, object="model", created=0, owned_by=impl_name) for m in models]
        )

    monkeypatch.setattr(DIAL, _stub)
    return calls


def _configured(**entry: Any) -> GatewayConfig:
    return GatewayConfig(hosted_providers={"openai": {"api_key": CONFIG_KEY, **entry}})


def _service(db: AsyncSession, config: GatewayConfig) -> HostedProviderService:
    uow = UnitOfWork(db)
    return HostedProviderService(
        uow,
        config=config,
        providers=HostedProviderRepository(uow),
        models=HostedProviderModelRepository(uow),
        endpoints=ProviderEndpointRepository(uow),
        pricing=DeploymentPricingService(
            uow, pricing=ModelPricingRepository(uow), overrides=OrganizationModelPricingRepository(uow)
        ),
        live_byo_pairs=OrgProviderKeyRepository(uow).live_provider_pairs,
        organization_exists=OrganizationRepository(uow).exists,
    )


async def _store(service: HostedProviderService) -> None:
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=STORED_KEY))


# --------------------------------------------------------------------------- #
# The merged view
# --------------------------------------------------------------------------- #


async def test_a_configured_provider_is_listed_read_only_with_no_key(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    config = GatewayConfig(
        hosted_providers={"openai": {"api_key": CONFIG_KEY, "client_args": {"aws_secret_access_key": "shh"}}}
    )

    listed = await _service(async_db, config).list_providers()

    [public] = listed.data
    assert listed.count == 1
    assert public.source == "config"
    assert public.id is None
    assert public.api_key_last4 == "9876"
    assert public.client_args is not None and public.client_args["aws_secret_access_key"] != "shh"
    assert CONFIG_KEY not in public.model_dump_json()


async def test_a_configured_provider_is_never_written_to_the_database(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch, "gpt-4o")
    service = _service(async_db, _configured())

    await service.list_providers()
    await service.refresh_models("openai")

    assert (await async_db.execute(select(HostedProvider))).scalars().all() == []


async def test_a_configured_provider_serves_on_its_configured_credential(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    service = _service(async_db, _configured(api_base="https://hosted.test/v1", client_args={"timeout": 9}))

    resolved = await service.resolve("openai", "gpt-4o", organization_id=uuid.uuid4())

    assert resolved is not None
    assert resolved.api_key == CONFIG_KEY
    assert resolved.api_base == "https://hosted.test/v1"
    assert resolved.client_args == {"timeout": 9}
    assert "openai" in await service.serveable_models(organization_id=None)


async def test_a_configured_provider_switched_off_is_not_served(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    service = _service(async_db, _configured(enabled=False))

    assert await service.resolve("openai", organization_id=None) is None
    assert await service.serveable_models(organization_id=None) == {}
    assert await service.discovery_dials() == {}


async def test_a_configured_restriction_withholds_the_provider_from_every_other_organization(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    allowed = uuid.uuid4()
    service = _service(async_db, _configured(restricted_to_organization_id=str(allowed)))

    assert await service.resolve("openai", organization_id=allowed) is not None
    assert await service.resolve("openai", organization_id=uuid.uuid4()) is None
    assert await service.serveable_models(organization_id=uuid.uuid4()) == {}


async def test_a_configured_provider_is_discovered_and_falls_back_to_its_declared_models(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _discovery(monkeypatch, unsupported=True)
    service = _service(async_db, _configured(models=["my-model"]))

    dials = await service.discovery_dials()
    discovery = await dials["openai"]()

    assert [model.id for model in discovery.models] == ["my-model"]
    assert discovery.error is None
    assert calls[0]["api_key"] == CONFIG_KEY


# --------------------------------------------------------------------------- #
# Read-only, and the roster
# --------------------------------------------------------------------------- #


async def test_a_configured_provider_cannot_be_changed_or_deleted_through_the_api(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    service = _service(async_db, _configured())

    with pytest.raises(HostedProviderReadOnlyError):
        await service.update_provider("openai", HostedProviderUpdateRequest(enabled=False))
    with pytest.raises(HostedProviderReadOnlyError):
        await service.delete_provider("openai")


async def test_a_configured_provider_gets_a_roster_toggles_and_prices(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch, "gpt-4o")
    service = _service(async_db, _configured())

    refreshed = await service.refresh_models("openai")
    added = await service.add_model(
        "openai",
        HostedModelCreateRequest(model="o3", input_price_per_million=2, output_price_per_million=8),
    )
    toggled = await service.update_model("openai", added.id, HostedModelUpdateRequest(enabled=False))
    listed = await service.list_models("openai")

    assert refreshed.added == ["gpt-4o"]
    assert toggled.enabled is False
    assert {model.model for model in listed.data} == {"gpt-4o", "o3"}
    [rate] = (await async_db.execute(select(ModelPricing).where(ModelPricing.model_key == "openai:o3"))).scalars().all()
    assert rate.input_price_per_million == Decimal("2")
    assert await service.resolve("openai", "o3", organization_id=None) is None


# --------------------------------------------------------------------------- #
# Shadowing
# --------------------------------------------------------------------------- #


async def test_a_stored_row_wins_over_the_configured_entry(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    service = _service(async_db, _configured())

    await _store(service)
    listed = await service.list_providers()
    resolved = await service.resolve("openai", organization_id=None)
    audit = await service.audit_config()

    [public] = listed.data
    assert public.source == "database"
    assert public.shadows_config is True
    assert public.api_key_last4 == "1234"
    assert resolved is not None and resolved.api_key == STORED_KEY
    assert audit.shadowed == ["openai"]


async def test_deleting_the_shadowing_row_keeps_the_roster_and_restores_the_entry(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch, "gpt-4o")
    service = _service(async_db, _configured())
    await _store(service)

    await service.delete_provider("openai")

    resolved = await service.resolve("openai", organization_id=None)
    assert resolved is not None and resolved.api_key == CONFIG_KEY
    roster = (await async_db.execute(select(HostedProviderModel.model))).scalars().all()
    assert roster == ["gpt-4o"]


async def test_an_unknown_configured_restriction_is_audited(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    missing = uuid.uuid4()

    audit = await _service(async_db, _configured(restricted_to_organization_id=str(missing))).audit_config()

    assert audit.unknown_restrictions == {"openai": missing}


# --------------------------------------------------------------------------- #
# Over HTTP
# --------------------------------------------------------------------------- #


@pytest.fixture
def configured_client(test_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    config = test_config.model_copy(update={"hosted_providers": {"openai": {"api_key": CONFIG_KEY}}})
    yield from build_test_client(config)


@pytest.fixture
def gateway_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """The gateway logger does not propagate, so caplog sees it only with its handler attached."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


def test_the_api_lists_a_configured_provider_and_refuses_to_change_it(
    configured_client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch, "gpt-4o")

    listed = configured_client.get(PATH, headers=master_key_header)
    patched = configured_client.patch(f"{PATH}/openai", json={"enabled": False}, headers=master_key_header)
    deleted = configured_client.delete(f"{PATH}/openai", headers=master_key_header)
    models = configured_client.post(f"{PATH}/openai/models/refresh", headers=master_key_header)

    assert listed.status_code == 200, listed.text
    [row] = listed.json()["data"]
    assert row["source"] == "config"
    assert row["id"] is None
    assert CONFIG_KEY not in listed.text
    assert patched.status_code == 409
    assert "defined in config.yml" in patched.json()["detail"]
    assert deleted.status_code == 409
    assert models.status_code == 200, models.text
    assert models.json()["added"] == ["gpt-4o"]


def test_the_api_creates_a_stored_provider_over_a_configured_one(
    configured_client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)

    created = configured_client.post(
        PATH, json={"provider": "openai", "api_key": STORED_KEY}, headers=master_key_header
    )

    assert created.status_code == 201, created.text
    assert created.json()["source"] == "database"
    assert created.json()["shadows_config"] is True


def test_boot_logs_a_configured_provider_a_stored_row_shadows(
    test_config: GatewayConfig,
    clean_database: None,
    db_session_factory: Callable[[], Session],
    gateway_logs: pytest.LogCaptureFixture,
) -> None:
    with db_session_factory() as db:
        db.add(HostedProvider(provider="openai", encrypted_api_key="stored", api_key_last4="1234", enabled=True))
        db.commit()
    config = test_config.model_copy(update={"hosted_providers": {"openai": {"api_key": CONFIG_KEY}}})

    for _ in build_test_client(config):
        pass

    shadowing = [record.getMessage() for record in gateway_logs.records if "shadowed" in record.getMessage()]
    assert shadowing, "boot logged no shadowing"
    assert "hosted_providers.openai" in shadowing[0]
    assert CONFIG_KEY not in gateway_logs.text


def test_a_request_is_served_on_a_configured_provider(
    configured_client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    assert (
        configured_client.post(
            f"{API_ROOT}/users", json={"user_id": "u1", "alias": "u1"}, headers=master_key_header
        ).status_code
        == 200
    )
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> None:
        captured.update(kwargs)
        raise RuntimeError

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        configured_client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "openai:gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
            headers=master_key_header,
        )

    assert captured, "the provider call was never made"
    assert captured["api_key"] == CONFIG_KEY
    assert captured["model"] == "openai:gpt-4o"
