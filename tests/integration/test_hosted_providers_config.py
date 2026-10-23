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
from gateway.exceptions.providers_exceptions import HostedCatalogEmptyError, HostedProviderReadOnlyError
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
from gateway.services.providers import HostedProviderService, HostedSources
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client, shared_test_config


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared configuration declaring no provider, so nothing configured shadows a hosted one."""
    return shared_test_config(postgres_url, {})


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
    providers = HostedProviderRepository(uow)
    return HostedProviderService(
        uow,
        config=config,
        providers=providers,
        sources=HostedSources(config, providers),
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_a_configured_provider_is_never_written_to_the_database(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch, "gpt-4o")
    service = _service(async_db, _configured())

    await service.list_providers()
    await service.refresh_models("openai")

    assert (await async_db.execute(select(HostedProvider))).scalars().all() == []


@pytest.mark.asyncio
async def test_a_configured_provider_serves_on_its_configured_credential(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    service = _service(async_db, _configured(api_base="https://hosted.test/v1", client_args={"timeout": 9}))
    assert await service.resolve("openai", "gpt-4o", organization_id=None) is None, "served before it was offered"
    await service.add_model(
        "openai", HostedModelCreateRequest(model="gpt-4o", input_price_per_million=2.5, output_price_per_million=10)
    )

    resolved = await service.resolve("openai", "gpt-4o", organization_id=uuid.uuid4())

    assert resolved is not None
    assert resolved.api_key == CONFIG_KEY
    assert resolved.api_base == "https://hosted.test/v1"
    assert resolved.client_args == {"timeout": 9}
    assert "openai" in await service.serveable_models(organization_id=None)


@pytest.mark.asyncio
async def test_a_configured_provider_switched_off_is_not_served(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    service = _service(async_db, _configured(enabled=False))

    assert await service.resolve("openai", organization_id=None) is None
    assert await service.serveable_models(organization_id=None) == {}
    assert await service.discovery_dials() == {}


@pytest.mark.asyncio
async def test_a_configured_restriction_withholds_the_provider_from_every_other_organization(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    allowed = uuid.uuid4()
    service = _service(async_db, _configured(restricted_to_organization_id=str(allowed)))

    assert await service.resolve("openai", organization_id=allowed) is not None
    assert await service.resolve("openai", organization_id=uuid.uuid4()) is None
    assert await service.serveable_models(organization_id=uuid.uuid4()) == {}


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_declared_models_do_not_hide_a_credential_the_provider_refuses(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a provider with no listing falls back; a bad key still reads as a failure."""

    async def _refused(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(provider=impl_name, models=[], error="401 invalid api key")

    monkeypatch.setattr(DIAL, _refused)
    service = _service(async_db, _configured(models=["my-model"]))

    discovery = await (await service.discovery_dials())["openai"]()

    assert (discovery.models, discovery.error) == ([], "401 invalid api key")


# --------------------------------------------------------------------------- #
# Read-only, and the roster
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_configured_provider_cannot_be_changed_or_deleted_through_the_api(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    service = _service(async_db, _configured())

    with pytest.raises(HostedProviderReadOnlyError):
        await service.update_provider("openai", HostedProviderUpdateRequest(enabled=False))
    with pytest.raises(HostedProviderReadOnlyError):
        await service.delete_provider("openai")


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_a_roster_outliving_its_entry_serves_nothing_and_the_sweep_removes_it(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    both = GatewayConfig(
        hosted_providers={"openai": {"api_key": CONFIG_KEY}, "anthropic": {"api_key": "sk-ant-config-5555"}}
    )
    before = _service(async_db, both)
    for provider, model in (("openai", "o3"), ("anthropic", "claude-x")):
        await before.add_model(
            provider, HostedModelCreateRequest(model=model, input_price_per_million=2, output_price_per_million=8)
        )
    # The openai entry is taken out of the configuration; its roster stays behind.
    after = _service(async_db, GatewayConfig(hosted_providers={"anthropic": {"api_key": "sk-ant-config-5555"}}))

    preview = await after.refresh_catalog(apply=False)

    assert await after.resolve("openai", "o3", organization_id=None) is None
    assert "openai" not in await after.serveable_models(organization_id=None)
    assert preview.orphaned_rosters == ["openai"]
    assert [(answer.provider, answer.source) for answer in preview.providers] == [("anthropic", "config")]
    assert "openai:o3" in preview.removed
    assert "anthropic:claude-x" not in preview.removed

    applied = await after.refresh_catalog(apply=True)

    assert applied.orphaned_rosters == ["openai"]
    roster = (await async_db.execute(select(HostedProviderModel.provider))).scalars().all()
    assert roster == ["anthropic"]
    priced = (await async_db.execute(select(ModelPricing.model_key))).scalars().all()
    assert "openai:o3" not in priced
    assert "anthropic:claude-x" in priced
    assert (await after.refresh_catalog(apply=False)).orphaned_rosters == []


@pytest.mark.asyncio
@pytest.mark.parametrize("apply", [False, True])
async def test_a_sweep_drops_a_listing_whose_entry_a_stored_row_replaced_mid_dial(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch, apply: bool
) -> None:
    """The configured key answered the listing, so it is not credited to the stored key that now serves the provider."""
    service = _service(async_db, _configured())
    await service.add_model(
        "openai", HostedModelCreateRequest(model="o3", input_price_per_million=2, output_price_per_million=8)
    )

    async def _dial_and_store(impl_name: str, **_: object) -> ProviderDiscovery:
        _discovery(monkeypatch)
        await _store(service)
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id="gpt-5", object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr(DIAL, _dial_and_store)
    result = await service.refresh_catalog(apply=apply)

    [answer] = result.providers
    assert (answer.source, answer.added, answer.error is not None) == ("config", [], True)
    roster = (await async_db.execute(select(HostedProviderModel.model))).scalars().all()
    assert roster == ["o3"]


@pytest.mark.asyncio
async def test_the_sweep_removes_the_roster_of_the_only_provider_and_keeps_every_price(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is offered once the sole entry goes, which is no reason to keep a roster nothing defines."""
    _discovery(monkeypatch)
    await _service(async_db, _configured()).add_model(
        "openai", HostedModelCreateRequest(model="o3", input_price_per_million=2, output_price_per_million=8)
    )
    after = _service(async_db, GatewayConfig())

    preview = await after.refresh_catalog(apply=False)
    applied = await after.refresh_catalog(apply=True)

    assert (preview.orphaned_rosters, preview.removed) == (["openai"], [])
    assert (applied.orphaned_rosters, applied.removed_price_rows) == (["openai"], 0)
    assert (await async_db.execute(select(HostedProviderModel))).scalars().all() == []
    assert "openai:o3" in (await async_db.execute(select(ModelPricing.model_key))).scalars().all()
    with pytest.raises(HostedCatalogEmptyError):
        await after.refresh_catalog(apply=False)


@pytest.mark.asyncio
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


def _offer(client: TestClient, headers: dict[str, str], provider: str, model: str) -> None:
    """Put ``model`` on ``provider``'s roster, priced and so switched on: nothing off the roster serves."""
    offered = client.post(
        f"{PATH}/{provider}/models",
        json={"model": model, "input_price_per_million": 2.5, "output_price_per_million": 10},
        headers=headers,
    )
    assert offered.status_code == 201, offered.text


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


def test_boot_continues_when_the_audit_cannot_read_the_database(
    test_config: GatewayConfig,
    clean_database: None,
    gateway_logs: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fails(_self: HostedProviderService) -> None:
        raise RuntimeError(f"connection lost while reading {CONFIG_KEY}")

    monkeypatch.setattr(HostedProviderService, "audit_config", _fails)
    config = test_config.model_copy(update={"hosted_providers": {"openai": {"api_key": CONFIG_KEY}}})

    for client in build_test_client(config):
        assert client.get(f"{API_ROOT}/health/liveness").status_code == 200

    assert "Could not audit hosted_providers" in gateway_logs.text
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
    _offer(configured_client, master_key_header, "openai", "gpt-4o")
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


@pytest.fixture
def keyless_client(test_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    config = test_config.model_copy(
        update={"hosted_providers": {"vllm": {"api_base": "http://vllm.internal:8000/v1", "models": ["qwen3-32b"]}}}
    )
    yield from build_test_client(config)


def test_a_request_is_served_on_a_keyless_configured_provider(
    keyless_client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    _discovery(monkeypatch)
    assert (
        keyless_client.post(
            f"{API_ROOT}/users", json={"user_id": "u1", "alias": "u1"}, headers=master_key_header
        ).status_code
        == 200
    )
    _offer(keyless_client, master_key_header, "vllm", "qwen3-32b")
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> None:
        captured.update(kwargs)
        raise RuntimeError

    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        keyless_client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "vllm:qwen3-32b", "messages": [{"role": "user", "content": "Hi"}], "user": "u1"},
            headers=master_key_header,
        )

    assert captured, "the provider call was never made"
    assert captured["api_base"] == "http://vllm.internal:8000/v1"
    # The keyless placeholder, never None and never another provider's key.
    assert captured["api_key"] == "otari-no-key-required"


@pytest.mark.asyncio
async def test_a_configured_provider_lists_its_supported_countries(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)

    listed = await _service(async_db, _configured(supported_countries=["gb", "de"])).list_providers()

    [public] = listed.data
    assert public.supported_countries == ["GB", "DE"]
