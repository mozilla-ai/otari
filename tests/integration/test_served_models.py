"""Integration tests for the context length a vLLM server reports, stored and listed over models.dev's."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.model import Model
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import build_model_metadata_override_service
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import User
from gateway.repositories.providers import OrgProviderKeyModelRepository
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    OrgProviderKeyRepository,
    UserRepository,
)
from gateway.schemas.providers import OrgProviderKeyCreateRequest
from gateway.services.catalog._served import refresh_served_models, reset_served_models, served_facts
from gateway.services.model_discovery_service import ProviderDiscovery, get_model_cache
from gateway.services.organization_pricing_service import OrganizationPricingService
from gateway.services.providers import OrgProviderModelService
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy import OrgProviderKeyService
from gateway.services.tenancy.org_provider_key_service import refresh_org_provider_cache
from gateway.services.tenancy.organization_service import OrganizationService

from .conftest import build_test_client

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}


def _config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        import_configured_providers=False,
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        model_discovery=True,
        model_cache_ttl_seconds=300,
        require_pricing=False,
        providers={
            "cairn": {
                "provider_type": "vllm",
                "api_base": "http://cairn.local:8000/v1",
                # Listed when the live listing fails.
                "models": ["cairn-qwen"],
            }
        },
    )


def _listed_model(config: GatewayConfig, alist: Any) -> dict[str, Any]:
    """Start the gateway, list its models with ``alist`` standing in for the server, and stop it."""
    get_model_cache().clear()
    client_gen = build_test_client(config)
    client = next(client_gen)
    try:
        with (
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", alist),
        ):
            response = client.get(f"{API_ROOT}/models", headers=_MASTER)
    finally:
        client_gen.close()
    assert response.status_code == 200
    (model,) = [model for model in response.json()["data"] if "cairn-qwen" in model["id"]]
    listed: dict[str, Any] = model
    return listed


def test_the_context_a_vllm_server_reports_is_listed_and_outlives_a_restart(postgres_url: str) -> None:
    config = _config(postgres_url)
    served = Model.model_validate(
        {
            "id": "cairn-qwen",
            "object": "model",
            "created": 0,
            "owned_by": "vllm",
            "max_model_len": 262144,
            "root": "Qwen/Qwen3-32B",
        }
    )

    assert _listed_model(config, AsyncMock(return_value=[served]))["context_window"] == 262144

    # A new start whose live listing fails falls back to the declared model, and
    # still lists the context the server reported, from the database.
    assert _listed_model(config, AsyncMock(side_effect=ConnectionError("down")))["context_window"] == 262144


def test_a_server_built_on_vllm_reports_its_context_and_reasoning_under_its_own_name(postgres_url: str) -> None:
    """TensorFold's listing, which models.dev knows nothing about."""
    config = _config(postgres_url)
    served = Model.model_validate(
        {
            "id": "cairn-qwen",
            "object": "model",
            "created": 0,
            "owned_by": "tensorfold",
            "context_length": 1048576,
            "max_model_len": 1048576,
            "reasoning": {"default_enabled": False, "default_effort": "high", "supported_efforts": ["max", "high"]},
        }
    )

    for alist in (AsyncMock(return_value=[served]), AsyncMock(side_effect=ConnectionError("down"))):
        model = _listed_model(config, alist)
        assert model["context_window"] == 1048576
        assert model["supported_parameters"] == ["reasoning"]


# --------------------------------------------------------------------------- #
# An organization's own key
# --------------------------------------------------------------------------- #


async def _owner_of_new_organization(db: AsyncSession, slug: str) -> User:
    organization = await OrganizationRepository(db).create_organization(
        name=slug.title(), slug=slug, created_by_user_id=None
    )
    user = await UserRepository(db).create_local_identity(
        full_name=f"{slug} owner", active_organization_id=organization.id
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role="owner"
    )
    return user


def _org_service(db: AsyncSession) -> OrgProviderModelService:
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
        metadata_overrides=build_model_metadata_override_service(uow),
        refresh_overlay=lambda: refresh_org_provider_cache(db),
    )


def _vllm_answers(monkeypatch: pytest.MonkeyPatch, context_by_base: dict[str, int]) -> None:
    """Make each key's dial answer as the vLLM server at its ``api_base`` would."""

    async def _stub(impl_name: str, *, api_base: str | None = None, **_: object) -> ProviderDiscovery:
        listed = {"id": "qwen", "object": "model", "created": 0, "owned_by": "vllm"}
        listed["max_model_len"] = context_by_base[api_base or ""]
        return ProviderDiscovery(provider=impl_name, models=[Model.model_validate(listed)])

    monkeypatch.setattr("gateway.services.providers._org_provider_model_service.test_provider_credentials", _stub)


@pytest.mark.asyncio
async def test_two_organizations_keep_their_own_servers_under_one_prefix(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    reset_served_models()
    acme = await _owner_of_new_organization(async_db, "acme")
    globex = await _owner_of_new_organization(async_db, "globex")
    _vllm_answers(monkeypatch, {"https://acme.example/v1": 262144, "https://globex.example/v1": 131072})
    for owner, base in ((acme, "https://acme.example/v1"), (globex, "https://globex.example/v1")):
        key = await OrgProviderKeyService(async_db).create_key_for_user(
            user=owner,
            request=OrgProviderKeyCreateRequest(
                provider="vllm", name="Cairn", slug="cairn", api_base=base, api_key="sk-live-1234"
            ),
        )
        await _org_service(async_db).refresh_models(user=owner, key_id=key.id)

    # Read back from the database, not from what each refresh left in memory.
    reset_served_models()
    await refresh_served_models(UnitOfWork(async_db))

    acme_facts = served_facts("cairn", "qwen", acme.active_organization_id)
    globex_facts = served_facts("cairn", "qwen", globex.active_organization_id)
    assert acme_facts is not None and acme_facts.context_window == 262144
    assert globex_facts is not None and globex_facts.context_window == 131072
    # Neither is the deployment's.
    assert served_facts("cairn", "qwen") is None
    reset_served_models()
