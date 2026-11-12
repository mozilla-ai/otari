"""Integration tests for model metadata overrides: stored, listed over the server's word, and scoped to their owner."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.model import Model
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import NotAuthorizedError
from gateway.exceptions.providers_exceptions import OrgProviderKeyNotFoundError
from gateway.models.tenancy import Organization, User
from gateway.repositories.providers import OrgProviderKeyModelRepository
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    OrgProviderKeyRepository,
    UserRepository,
)
from gateway.schemas.providers import OrgProviderKeyCreateRequest
from gateway.services.catalog import MetadataOverride
from gateway.services.catalog._overrides import metadata_override
from gateway.services.catalog._served import refresh_served_models, reset_served_models
from gateway.services.model_discovery_service import get_model_cache
from gateway.services.organization_pricing_service import OrganizationPricingService
from gateway.services.providers import OrgProviderModelService
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy import OrgProviderKeyService
from gateway.services.tenancy.org_provider_key_service import refresh_org_provider_cache
from gateway.services.tenancy.organization_service import OrganizationService

from .conftest import build_test_client

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_OVERRIDE = f"{API_ROOT}/model-overrides/cairn:cairn-glm"


@pytest.fixture(autouse=True)
def _clean_copies() -> Iterator[None]:
    reset_served_models()
    yield
    reset_served_models()


def _config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        import_configured_providers=False,
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        model_discovery=True,
        model_cache_ttl_seconds=300,
        require_pricing=False,
        providers={"cairn": {"provider_type": "vllm", "api_base": "http://cairn.local:8000/v1"}},
    )


def _listed(config: GatewayConfig, *writes: tuple[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Start the gateway, make ``writes`` to the override, list the model, and stop it."""
    get_model_cache().clear()
    served = Model(id="cairn-glm", object="model", created=0, owned_by="tensorfold")
    client_gen = build_test_client(config)
    client = next(client_gen)
    try:
        for method, body in writes:
            response = client.request(method, _OVERRIDE, json=body, headers=_MASTER)
            assert response.status_code in (200, 204), response.text
        with (
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", AsyncMock(return_value=[served])),
        ):
            response = client.get(f"{API_ROOT}/models", headers=_MASTER)
    finally:
        client_gen.close()
    assert response.status_code == 200
    (model,) = [model for model in response.json()["data"] if "cairn-glm" in model["id"]]
    listed: dict[str, Any] = model
    return listed


def test_an_operator_s_override_is_listed_and_outlives_a_restart(postgres_url: str) -> None:
    config = _config(postgres_url)
    body = {"context_window": 1048576, "max_output_tokens": 65536, "tool_call": True, "reasoning": True}

    for writes in (("PUT", body),), ():
        model = _listed(config, *writes)
        assert model["context_window"] == 1048576
        assert model["max_output_tokens"] == 65536
        assert model["supported_parameters"] == ["tools", "tool_choice", "reasoning"]


def test_clearing_an_override_leaves_the_model_undescribed(postgres_url: str) -> None:
    model = _listed(_config(postgres_url), ("PUT", {"context_window": 8192}), ("DELETE", None))

    assert model.get("context_window") is None
    assert model.get("supported_parameters") is None


def test_a_selector_without_a_provider_is_refused(postgres_url: str) -> None:
    client_gen = build_test_client(_config(postgres_url))
    client = next(client_gen)
    try:
        response = client.put(f"{API_ROOT}/model-overrides/cairn-glm", json={}, headers=_MASTER)
    finally:
        client_gen.close()

    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# An organization's own keys
# --------------------------------------------------------------------------- #


async def _member(db: AsyncSession, organization: Organization, role: str) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=f"{organization.slug} {role}", active_organization_id=organization.id
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


def _service(db: AsyncSession) -> OrgProviderModelService:
    uow = UnitOfWork(db)
    config = GatewayConfig(require_pricing=False)
    return OrgProviderModelService(
        uow,
        config=config,
        organizations=OrganizationService(db, membership_listener=None),
        provider_keys=OrgProviderKeyService(db),
        org_pricing=OrganizationPricingService(db, config, model_provider=None),
        models=OrgProviderKeyModelRepository(uow),
        keys=OrgProviderKeyRepository(db),
        refresh_overlay=lambda: refresh_org_provider_cache(db),
    )


@pytest.mark.asyncio
async def test_an_organization_overrides_only_what_its_own_keys_serve(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    acme = await OrganizationRepository(async_db).create_organization(name="Acme", slug="acme", created_by_user_id=None)
    admin = await _member(async_db, acme, "admin")
    await OrgProviderKeyService(async_db).create_key_for_user(
        user=admin,
        request=OrgProviderKeyCreateRequest(
            provider="openai", name="TensorFold", slug="tensorfold", api_base="https://tf.example/v1", api_key="sk-1"
        ),
    )
    service = _service(async_db)

    await service.set_metadata_override(
        user=admin, selector="tensorfold:GLM-5.3-Flash", override=MetadataOverride(context_window=1048576)
    )
    with pytest.raises(OrgProviderKeyNotFoundError):
        await service.set_metadata_override(
            user=admin, selector="openai:gpt-4o", override=MetadataOverride(context_window=1)
        )
    with pytest.raises(NotAuthorizedError):
        await service.set_metadata_override(
            user=await _member(async_db, acme, "member"),
            selector="tensorfold:GLM-5.3-Flash",
            override=MetadataOverride(context_window=1),
        )

    # Read back from the database, not from what the write left in memory.
    reset_served_models()
    await refresh_served_models(UnitOfWork(async_db))
    assert metadata_override("tensorfold", "GLM-5.3-Flash", acme.id) == MetadataOverride(context_window=1048576)
    assert metadata_override("tensorfold", "GLM-5.3-Flash") is None

    await service.clear_metadata_override(user=admin, selector="tensorfold:GLM-5.3-Flash")
    reset_served_models()
    await refresh_served_models(UnitOfWork(async_db))
    assert metadata_override("tensorfold", "GLM-5.3-Flash", acme.id) is None


@pytest.mark.asyncio
async def test_an_organization_s_details_go_with_its_last_key_under_the_prefix(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    acme = await OrganizationRepository(async_db).create_organization(name="Acme", slug="acme", created_by_user_id=None)
    admin = await _member(async_db, acme, "admin")
    keys = OrgProviderKeyService(async_db)
    first, second = [
        await keys.create_key_for_user(
            user=admin, request=OrgProviderKeyCreateRequest(provider="openai", name=name, api_key="sk-1")
        )
        for name in ("Research", "Sales")
    ]
    service = _service(async_db)
    await service.set_metadata_override(user=admin, selector="openai:gpt-x", override=MetadataOverride(reasoning=True))

    for key, left in ((first, MetadataOverride(reasoning=True)), (second, None)):
        await keys.archive_key_for_user(user=admin, key_id=key.id)
        prefix = await keys.delete_key_for_user(user=admin, key_id=key.id)
        await service.forget_metadata_overrides(user=admin, prefix=prefix)
        reset_served_models()
        await refresh_served_models(UnitOfWork(async_db))
        assert metadata_override("openai", "gpt-x", acme.id) == left


def test_a_deleted_hosted_provider_takes_its_details_with_it(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    client_gen = build_test_client(_config(postgres_url))
    client = next(client_gen)
    try:
        created = client.post(
            f"{API_ROOT}/hosted-providers",
            json={"provider": "openai", "name": "tensorfold", "api_key": "sk-hosted"},
            headers=_MASTER,
        )
        assert created.status_code == 201, created.text
        put = client.put(f"{API_ROOT}/model-overrides/tensorfold:glm", json={"context_window": 8192}, headers=_MASTER)
        assert put.status_code == 200
        assert metadata_override("tensorfold", "glm") == MetadataOverride(context_window=8192)

        assert client.delete(f"{API_ROOT}/hosted-providers/tensorfold", headers=_MASTER).status_code == 200
    finally:
        client_gen.close()

    assert metadata_override("tensorfold", "glm") is None
