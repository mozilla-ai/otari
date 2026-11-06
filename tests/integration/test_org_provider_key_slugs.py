"""A key with a custom endpoint is named by a slug of its own, not by the provider it speaks (#2899)."""

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from any_llm import LLMProvider
from any_llm.exceptions import UnsupportedProviderError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.exceptions.providers_exceptions import (
    OrgProviderKeyInvalidSlugError,
    OrgProviderKeySlugTakenError,
)
from gateway.models.tenancy import Organization, User
from gateway.repositories.tenancy import (
    OrganizationRepository,
    WorkspaceProviderKeyOverrideRepository,
    WorkspaceRepository,
)
from gateway.schemas.providers import OrgProviderKeyCreateRequest, OrgProviderKeyUpdateRequest
from gateway.services.organization_pricing_service import OrganizationPricingService
from gateway.services.provider_kwargs import resolve_provider_selector
from gateway.services.tenancy.org_provider_key_service import (
    OrgProviderKeyService,
    refresh_org_provider_cache,
    reset_org_provider_cache,
)

from .test_organization_pricing_routes import _FakeHostedModelProvider, _identity, _rates

_SYNTHETIC = "https://api.synthetic.new/openai/v1"
_MODEL = "hf:Qwen/Qwen3.8-27B"


@pytest.fixture(autouse=True)
def _clean_overlay() -> Iterator[None]:
    reset_org_provider_cache()
    yield
    reset_org_provider_cache()


async def _organization(db: AsyncSession, slug: str) -> tuple[Organization, User]:
    organization = await OrganizationRepository(db).create_organization(name="Acme", slug=slug, created_by_user_id=None)
    admin = await _identity(db, organization, role="admin", name="admin person")
    return organization, admin


def _synthetic(name: str = "Synthetic", slug: str | None = None) -> OrgProviderKeyCreateRequest:
    return OrgProviderKeyCreateRequest(provider="openai", name=name, slug=slug, api_base=_SYNTHETIC)


@pytest.mark.asyncio
async def test_a_key_with_a_custom_endpoint_takes_a_slug_from_its_name(async_db: AsyncSession) -> None:
    _, admin = await _organization(async_db, "acme-slug-name")
    keys = OrgProviderKeyService(async_db)

    custom = await keys.create_key_for_user(user=admin, request=_synthetic())
    plain = await keys.create_key_for_user(
        user=admin, request=OrgProviderKeyCreateRequest(provider="openai", name="OpenAI")
    )

    assert custom.slug == "synthetic"
    assert plain.slug is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("slug", "api_base"),
    [
        pytest.param("openai", _SYNTHETIC, id="a-provider-name"),
        pytest.param("Has Spaces", _SYNTHETIC, id="malformed"),
        pytest.param("synthetic", None, id="no-custom-endpoint"),
    ],
)
async def test_a_slug_that_cannot_name_the_key_is_refused(
    async_db: AsyncSession, slug: str, api_base: str | None
) -> None:
    _, admin = await _organization(async_db, f"acme-bad-{abs(hash(slug)) % 10_000}")

    with pytest.raises(OrgProviderKeyInvalidSlugError):
        await OrgProviderKeyService(async_db).create_key_for_user(
            user=admin,
            request=OrgProviderKeyCreateRequest(provider="openai", name="Synthetic", slug=slug, api_base=api_base),
        )


@pytest.mark.asyncio
async def test_a_slug_is_refused_when_a_configured_instance_has_it(async_db: AsyncSession) -> None:
    _, admin = await _organization(async_db, "acme-instance-slug")
    config = GatewayConfig(providers={"synthetic": {"provider_type": "openai", "api_key": "sk-deployment"}})
    keys = OrgProviderKeyService(async_db, config)

    with pytest.raises(OrgProviderKeyInvalidSlugError):
        await keys.create_key_for_user(user=admin, request=_synthetic(slug="synthetic"))
    derived = await keys.create_key_for_user(user=admin, request=_synthetic())

    assert derived.slug == "synthetic-custom"


@pytest.mark.asyncio
async def test_two_keys_cannot_share_a_slug(async_db: AsyncSession) -> None:
    _, admin = await _organization(async_db, "acme-slug-taken")
    keys = OrgProviderKeyService(async_db)
    await keys.create_key_for_user(user=admin, request=_synthetic())

    with pytest.raises(OrgProviderKeySlugTakenError):
        await keys.create_key_for_user(
            user=admin,
            request=OrgProviderKeyCreateRequest(
                provider="anthropic", name="Other", slug="synthetic", api_base="https://other.example/v1"
            ),
        )


@pytest.mark.asyncio
async def test_a_slug_made_from_the_name_steps_around_a_provider_and_a_taken_slug(async_db: AsyncSession) -> None:
    _, admin = await _organization(async_db, "acme-slug-derived")
    keys = OrgProviderKeyService(async_db)

    first = await keys.create_key_for_user(user=admin, request=_synthetic(name="OpenAI"))
    second = await keys.create_key_for_user(user=admin, request=_synthetic(name="OpenAI EU", slug=None))
    third = await keys.create_key_for_user(
        user=admin,
        request=OrgProviderKeyCreateRequest(provider="anthropic", name="OpenAI", api_base="https://c.example/v1"),
    )

    assert (first.slug, second.slug, third.slug) == ("openai-custom", "openai-eu", "openai-custom-2")


@pytest.mark.asyncio
async def test_a_slug_survives_a_rename_and_a_new_endpoint(async_db: AsyncSession) -> None:
    _, admin = await _organization(async_db, "acme-slug-fixed")
    keys = OrgProviderKeyService(async_db)
    created = await keys.create_key_for_user(user=admin, request=_synthetic())

    updated = await keys.update_key_for_user(
        user=admin,
        key_id=created.id,
        request=OrgProviderKeyUpdateRequest(name="Synthetic EU", api_base="https://eu.synthetic.new/openai/v1"),
    )

    assert updated.slug == "synthetic"


@pytest.mark.asyncio
async def test_a_key_with_a_slug_keeps_its_endpoint(async_db: AsyncSession) -> None:
    _, admin = await _organization(async_db, "acme-slug-keeps-base")
    keys = OrgProviderKeyService(async_db)
    created = await keys.create_key_for_user(user=admin, request=_synthetic())

    with pytest.raises(OrgProviderKeyInvalidSlugError):
        await keys.update_key_for_user(
            user=admin, key_id=created.id, request=OrgProviderKeyUpdateRequest(api_base=None)
        )


@pytest.mark.asyncio
async def test_a_key_for_the_providers_own_endpoint_cannot_move_to_another(async_db: AsyncSession) -> None:
    _, admin = await _organization(async_db, "acme-slug-no-move")
    keys = OrgProviderKeyService(async_db)
    created = await keys.create_key_for_user(
        user=admin, request=OrgProviderKeyCreateRequest(provider="openai", name="OpenAI")
    )

    with pytest.raises(OrgProviderKeyInvalidSlugError):
        await keys.update_key_for_user(
            user=admin, key_id=created.id, request=OrgProviderKeyUpdateRequest(api_base=_SYNTHETIC)
        )


@pytest.mark.asyncio
async def test_requests_reach_the_key_by_its_slug_and_not_by_its_provider(async_db: AsyncSession) -> None:
    organization, admin = await _organization(async_db, "acme-slug-dispatch")
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Default", organization_id=organization.id, created_by_user_id=None
    )
    await OrgProviderKeyService(async_db).create_key_for_user(user=admin, request=_synthetic())
    config = GatewayConfig()

    by_slug = resolve_provider_selector(config, f"synthetic:{_MODEL}", workspace_id=workspace.id)
    by_provider = resolve_provider_selector(config, f"openai:{_MODEL}", workspace_id=workspace.id)

    assert (by_slug.instance, by_slug.provider, by_slug.model) == ("synthetic", LLMProvider.OPENAI, _MODEL)
    assert by_slug.kwargs["api_base"] == _SYNTHETIC
    assert by_provider.instance == "openai"
    assert by_provider.kwargs.get("api_base") != _SYNTHETIC


@pytest.mark.asyncio
async def test_a_workspace_that_disabled_the_key_cannot_reach_its_slug(async_db: AsyncSession) -> None:
    organization, admin = await _organization(async_db, "acme-slug-disabled")
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Default", organization_id=organization.id, created_by_user_id=None
    )
    created = await OrgProviderKeyService(async_db).create_key_for_user(user=admin, request=_synthetic())
    await WorkspaceProviderKeyOverrideRepository(async_db).create(
        workspace_id=workspace.id,
        organization_id=organization.id,
        org_provider_key_id=created.id,
        is_default=False,
        disabled=True,
    )
    await refresh_org_provider_cache(async_db)

    with pytest.raises(UnsupportedProviderError):
        resolve_provider_selector(GatewayConfig(), f"synthetic:{_MODEL}", workspace_id=workspace.id)


@pytest.mark.asyncio
async def test_an_organization_prices_its_custom_endpoint_while_the_deployment_hosts_its_provider(
    async_db: AsyncSession,
) -> None:
    """#2899: the deployment serving ``openai`` no longer stops a rate for a model only Synthetic serves."""
    organization, admin = await _organization(async_db, "acme-slug-pricing")
    await WorkspaceRepository(async_db).create_workspace(
        name="Default", organization_id=organization.id, created_by_user_id=None
    )
    created = await OrgProviderKeyService(async_db).create_key_for_user(user=admin, request=_synthetic())
    # A second workspace switched off the key, which used to hand openai back to the deployment.
    other = await WorkspaceRepository(async_db).create_workspace(
        name="Other", organization_id=organization.id, created_by_user_id=None
    )
    await WorkspaceProviderKeyOverrideRepository(async_db).create(
        workspace_id=other.id,
        organization_id=organization.id,
        org_provider_key_id=created.id,
        is_default=False,
        disabled=True,
    )
    pricing = OrganizationPricingService(
        async_db, GatewayConfig(), model_provider=_FakeHostedModelProvider(served="openai")
    )

    saved = await pricing.create_for_caller(admin, f"synthetic:{_MODEL}", _rates())

    effective = await pricing.rates_in_effect(organization.id, [f"synthetic:{_MODEL}"], datetime.now(UTC))
    row = effective[f"synthetic:{_MODEL}"].row
    assert row is not None
    assert row.id == saved.id
    assert await pricing.deployment_supplied_keys(organization.id, ["openai:gpt-4o"]) == {"openai:gpt-4o"}
