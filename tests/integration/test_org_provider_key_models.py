"""The models an organization offers on its provider keys, against a real DB.

Exercised at the service layer for the reason `test_org_provider_keys.py` gives:
the routes can only ever act as the one bootstrap operator, who is always an
owner, so the rules that matter are only reachable by building identities at
whatever role a case needs.

Model discovery is stubbed throughout. What is being tested is the offer rule,
the seeding, the provenance and the switch, none of which is about whether
any-llm can reach a provider; a case that dialed for real would be testing the
network.
"""

import uuid
from collections.abc import Iterator
from decimal import Decimal

import pytest
from any_llm.types.model import Model
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import NotAuthorizedError
from gateway.exceptions.providers_exceptions import (
    OrgProviderKeyNotFoundError,
    OrgProviderLastModelError,
    OrgProviderModelAlreadyOfferedError,
    OrgProviderModelNameRequiredError,
    OrgProviderModelNotFoundError,
    OrgProviderModelUnpricedError,
)
from gateway.models.pricing import SEED_ORIGIN, ModelPricing, OrganizationModelPricing
from gateway.models.provider_keys import (
    OrgProviderKeyModel,
)
from gateway.models.tenancy import Organization, User, Workspace
from gateway.repositories.providers import OrgProviderKeyModelRepository
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    OrgProviderKeyRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.schemas.providers import (
    OrgProviderKeyCreateRequest,
)
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.organization_pricing_service import (
    OrganizationPricingService,
    PricingOverrideInput,
)
from gateway.services.providers import OrgProviderModelService
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy import OrgProviderKeyService
from gateway.services.tenancy.org_provider_key_service import (
    cached_org_model_restriction,
    refresh_org_provider_cache,
    reset_org_provider_cache,
)
from gateway.services.tenancy.organization_service import OrganizationService

pytestmark = pytest.mark.asyncio


async def _organization(db: AsyncSession, *, slug: str = "acme") -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _member(db: AsyncSession, organization: Organization, *, role: str, full_name: str) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


async def _workspace(
    db: AsyncSession, organization: Organization, *, name: str = "Default", owner: User | None = None
) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name=name, organization_id=organization.id, created_by_user_id=owner.id if owner else None
    )
    if owner is not None:
        await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    return workspace


@pytest.fixture(autouse=True)
def _secret_key_and_clean_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    reset_org_provider_cache()
    yield
    reset_org_provider_cache()


async def _none(*_args: object, **_kwargs: object) -> None:
    """A repository read that answers "absent", standing in for a lost race."""
    return None


def _service(db: AsyncSession) -> OrgProviderModelService:
    uow = UnitOfWork(db)
    config = GatewayConfig()
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


def _discovery(monkeypatch: pytest.MonkeyPatch, *models: str, error: str | None = None) -> None:
    """Make every dial answer with ``models``, or with ``error`` and nothing."""

    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name,
            models=[]
            if error
            else [Model(id=model, object="model", created=0, owned_by=impl_name) for model in models],
            error=error,
        )

    monkeypatch.setattr("gateway.services.providers._org_provider_model_service.test_provider_credentials", _stub)


def _defaults(monkeypatch: pytest.MonkeyPatch, rates: dict[str, tuple[str, str]]) -> None:
    """Make the community dataset price exactly ``rates``, and nothing else."""

    def _stub(provider: str | None, model: str, as_of: object) -> ModelPricing | None:
        if model not in rates:
            return None
        input_rate, output_rate = rates[model]
        return ModelPricing(
            model_key=f"{provider}:{model}",
            input_price_per_million=Decimal(input_rate),
            output_price_per_million=Decimal(output_rate),
            pricing_tiers=[],
            unit="tokens",
        )

    monkeypatch.setattr("gateway.services.organization_pricing_service.default_model_pricing", _stub)


async def _key(db: AsyncSession, owner: User, *, provider: str = "openai", name: str = "primary") -> uuid.UUID:
    created = await OrgProviderKeyService(db).create_key_for_user(
        user=owner, request=OrgProviderKeyCreateRequest(provider=provider, name=name, api_key="sk-live-1234")
    )
    return created.id


async def _offered(db: AsyncSession, key_id: uuid.UUID) -> dict[str, bool]:
    rows = (
        (await db.execute(select(OrgProviderKeyModel).where(col(OrgProviderKeyModel.org_provider_key_id) == key_id)))
        .scalars()
        .all()
    )
    return {row.model: row.enabled for row in rows}


async def _organization_rates(db: AsyncSession) -> list[OrganizationModelPricing]:
    return list((await db.execute(select(OrganizationModelPricing))).scalars().all())


async def _deployment_rate_count(db: AsyncSession) -> int:
    return len(list((await db.execute(select(ModelPricing))).scalars().all()))


# --------------------------------------------------------------------------- #
# Offering
# --------------------------------------------------------------------------- #


async def test_adding_a_provider_offers_its_models_in_one_call(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch, "gpt-4o")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")

    key = await _service(async_db).add_provider_key(
        user=owner, request=OrgProviderKeyCreateRequest(provider="openai", name="primary", api_key="sk-live-1234")
    )

    assert await _offered(async_db, key.id) == {"gpt-4o": True}


async def test_a_key_survives_the_offer_that_follows_it_failing(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The create commits first, so an error after it would report a failure for
    something already durable, and send the admin into a retry that collides with
    the key they just made. The key comes back with nothing offered instead, which
    is the state the Refresh models button is for."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")

    async def _explode(_self: OrgProviderModelService, **_: object) -> None:
        raise OperationalError("SELECT 1", {}, Exception("the connection went away"))

    monkeypatch.setattr(OrgProviderModelService, "refresh_models", _explode)

    key = await _service(async_db).add_provider_key(
        user=owner, request=OrgProviderKeyCreateRequest(provider="openai", name="primary", api_key="sk-live-1234")
    )

    assert key.name == "primary"
    assert await _offered(async_db, key.id) == {}


async def test_an_unexpected_failure_after_the_create_is_not_swallowed(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the database may fail the offer quietly. Anything else on a key this
    call just made is a bug, and a caught one is a bug nobody sees."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")

    async def _explode(_self: OrgProviderModelService, **_: object) -> None:
        raise RuntimeError("something nothing anticipated")

    monkeypatch.setattr(OrgProviderModelService, "refresh_models", _explode)

    with pytest.raises(RuntimeError):
        await _service(async_db).add_provider_key(
            user=owner,
            request=OrgProviderKeyCreateRequest(provider="openai", name="primary", api_key="sk-live-1234"),
        )


async def test_a_refresh_offers_what_the_provider_lists_and_seeds_its_rates(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, "gpt-4o", "gpt-4o-mini")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "gpt-4o-mini": ("0.15", "0.6")})

    result = await _service(async_db).refresh_models(user=owner, key_id=key_id)

    assert result.added == ["gpt-4o", "gpt-4o-mini"]
    assert result.count == 2
    assert await _offered(async_db, key_id) == {"gpt-4o": True, "gpt-4o-mini": True}

    rates = {row.model_key: row for row in await _organization_rates(async_db)}
    assert set(rates) == {"openai:gpt-4o", "openai:gpt-4o-mini"}
    assert all(row.origin == SEED_ORIGIN for row in rates.values())
    assert rates["openai:gpt-4o"].input_price_per_million == Decimal("2.5")
    assert rates["openai:gpt-4o"].effective_to is None


async def test_seeding_never_writes_to_the_deployment_price_list(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule the whole design rests on.

    ``model_pricing`` carries no tenancy column, so a row there prices the model
    for every organization on the deployment. One organization adopting a model
    on its own credential must not move what another is billed.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, "gpt-4o")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    before = await _deployment_rate_count(async_db)

    await _service(async_db).refresh_models(user=owner, key_id=key_id)

    assert await _deployment_rate_count(async_db) == before


async def test_a_model_nothing_prices_is_offered_but_not_served(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabled until priced, so a model the pricing data has not caught up with
    cannot be billed at nothing."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, "gpt-4o", "gpt-6-unreleased")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})

    await _service(async_db).refresh_models(user=owner, key_id=key_id)

    assert await _offered(async_db, key_id) == {"gpt-4o": True, "gpt-6-unreleased": False}


async def test_a_provider_that_will_not_say_offers_nothing_and_says_why(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reported in the body, never as a status: the credential may still be right
    for dispatch, and models can be added by name."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, error="the upstream refused the credential")

    result = await _service(async_db).refresh_models(user=owner, key_id=key_id)

    assert result.error == "the upstream refused the credential"
    assert result.added == []
    assert await _offered(async_db, key_id) == {}


async def test_a_refresh_never_unlists_a_model_the_provider_dropped(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Additive only: delisting is the serving switch's job, and an upstream
    hiccup must not empty a catalog."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "gpt-4o-mini": ("0.15", "0.6")})
    _discovery(monkeypatch, "gpt-4o", "gpt-4o-mini")
    await _service(async_db).refresh_models(user=owner, key_id=key_id)

    _discovery(monkeypatch, "gpt-4o")
    result = await _service(async_db).refresh_models(user=owner, key_id=key_id)

    assert result.added == []
    assert set(await _offered(async_db, key_id)) == {"gpt-4o", "gpt-4o-mini"}


async def test_adding_a_model_by_name_offers_it(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})

    offered = await _service(async_db).add_model(user=owner, key_id=key_id, model="  gpt-4o  ")

    assert offered.model == "gpt-4o"
    assert offered.enabled is True
    assert offered.price_source == "defaults"

    with pytest.raises(OrgProviderModelAlreadyOfferedError):
        await _service(async_db).add_model(user=owner, key_id=key_id, model="gpt-4o")


async def test_the_unique_index_decides_a_racing_offer(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-check races the insert, so the constraint is the real arbiter.

    Driven by making the pre-check answer "absent" for a model that is in fact
    already offered, which is the state a concurrent request leaves behind
    between the two. Without the translation the loser leaves with a 500 rather
    than the 409 it is.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _service(async_db).add_model(user=owner, key_id=key_id, model="gpt-4o")

    monkeypatch.setattr(
        "gateway.repositories.providers.OrgProviderKeyModelRepository.get_by_model",
        _none,
    )

    with pytest.raises(OrgProviderModelAlreadyOfferedError):
        await _service(async_db).add_model(user=owner, key_id=key_id, model="gpt-4o")


async def test_a_blank_model_name_is_refused(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)

    with pytest.raises(OrgProviderModelNameRequiredError):
        await _service(async_db).add_model(user=owner, key_id=key_id, model="   ")


# --------------------------------------------------------------------------- #
# Pricing provenance
# --------------------------------------------------------------------------- #


async def test_a_rate_the_admin_set_stops_following_the_community_default(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``origin`` is the whole provenance mechanism: a seeded rate moves with the
    dataset, and a chosen one never does again."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, "gpt-4o")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _service(async_db).refresh_models(user=owner, key_id=key_id)

    # The dataset moves, and the seeded rate follows it.
    _defaults(monkeypatch, {"gpt-4o": ("3.0", "12")})
    result = await _service(async_db).refresh_pricing(user=owner, key_id=key_id)
    assert result.repriced == ["gpt-4o"]
    [stored] = await _organization_rates(async_db)
    assert stored.input_price_per_million == Decimal("3.0")
    assert stored.origin == SEED_ORIGIN

    # An admin chooses a rate, and the dataset stops reaching it.
    stored.input_price_per_million = Decimal("1.0")
    stored.origin = "api"
    await async_db.commit()
    _defaults(monkeypatch, {"gpt-4o": ("9.0", "30")})
    result = await _service(async_db).refresh_pricing(user=owner, key_id=key_id)

    assert result.repriced == []
    [stored] = await _organization_rates(async_db)
    assert stored.input_price_per_million == Decimal("1.0")


async def test_replacing_a_rate_through_the_pricing_api_stops_a_refresh_moving_it(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this caught, driven through the surface an admin actually uses.

    The test above sets ``origin`` by hand, which proves the reseed rule but not
    that anything sets it. ``PUT /organizations/me/pricing/{id}`` is what an
    admin's rate edit goes through, and until it marked the row as chosen a
    refresh moved the rate straight back to the community default: the admin set
    a price, pressed Refresh pricing, and watched it revert.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, "gpt-4o")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _service(async_db).refresh_models(user=owner, key_id=key_id)
    [seeded] = await _organization_rates(async_db)

    await OrganizationPricingService(async_db, GatewayConfig(), model_provider=None).replace_for_caller(
        owner,
        seeded.id,
        PricingOverrideInput(
            input_price_per_million=1.0,
            output_price_per_million=4.0,
            cache_read_price_per_million=None,
            cache_write_price_per_million=None,
            cache_write_1h_price_per_million=None,
            pricing_tiers=[],
            effective_from=seeded.effective_from,
            effective_to=None,
        ),
    )
    await async_db.commit()

    listed = await _service(async_db).list_models(user=owner, key_id=key_id)
    assert listed.data[0].price_source == "organization"

    result = await _service(async_db).refresh_pricing(user=owner, key_id=key_id)

    assert result.repriced == []
    [stored] = await _organization_rates(async_db)
    assert stored.input_price_per_million == Decimal("1.0")


async def test_a_default_that_left_the_dataset_leaves_the_stored_rate_alone(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, "gpt-4o")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _service(async_db).refresh_models(user=owner, key_id=key_id)

    _defaults(monkeypatch, {})
    result = await _service(async_db).refresh_pricing(user=owner, key_id=key_id)

    assert result.repriced == []
    [stored] = await _organization_rates(async_db)
    assert stored.input_price_per_million == Decimal("2.5")


async def test_refreshing_pricing_serves_a_model_the_dataset_has_caught_up_with(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason Refresh pricing exists as its own action: a model that arrived
    disabled because nothing priced it is exactly what an admin presses it for."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, "gpt-6-unreleased")
    _defaults(monkeypatch, {})
    await _service(async_db).refresh_models(user=owner, key_id=key_id)
    assert await _offered(async_db, key_id) == {"gpt-6-unreleased": False}

    _defaults(monkeypatch, {"gpt-6-unreleased": ("4.0", "16")})
    result = await _service(async_db).refresh_pricing(user=owner, key_id=key_id)

    assert result.repriced == ["gpt-6-unreleased"]
    assert await _offered(async_db, key_id) == {"gpt-6-unreleased": True}


async def test_an_unchanged_default_is_not_repriced(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compared at the rate column's own scale, so a stored value that has been
    through it and a freshly resolved one that has not still compare equal."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, "gpt-4o")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _service(async_db).refresh_models(user=owner, key_id=key_id)

    result = await _service(async_db).refresh_pricing(user=owner, key_id=key_id)

    assert result.repriced == []


async def test_a_deployment_priced_model_is_offered_without_a_seeded_rate(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deployment's list already answers, so seeding would add a tenant rate
    nobody asked for and hide the rung it came from."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    async_db.add(
        ModelPricing(
            model_key="openai:gpt-4o",
            input_price_per_million=Decimal("1.0"),
            output_price_per_million=Decimal("4.0"),
            pricing_tiers=[],
            unit="tokens",
            origin="api",
        )
    )
    await async_db.commit()
    _discovery(monkeypatch, "gpt-4o")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})

    await _service(async_db).refresh_models(user=owner, key_id=key_id)

    assert await _organization_rates(async_db) == []
    assert await _offered(async_db, key_id) == {"gpt-4o": True}
    listed = await _service(async_db).list_models(user=owner, key_id=key_id)
    assert [(row.model, row.price_source) for row in listed.data] == [("gpt-4o", "deployment")]


# --------------------------------------------------------------------------- #
# The serving switch
# --------------------------------------------------------------------------- #


async def test_the_switch_reaches_dispatch_through_the_overlay_cache(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalog and the inference gate must not disagree about one model, so
    the switch feeds the same allow-list a workspace restriction does."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    key_id = await _key(async_db, owner)
    await OrgProviderKeyService(async_db).set_org_default_for_user(user=owner, key_id=key_id)
    _discovery(monkeypatch, "gpt-4o", "gpt-4o-mini")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "gpt-4o-mini": ("0.15", "0.6")})
    await _service(async_db).refresh_models(user=owner, key_id=key_id)

    await refresh_org_provider_cache(async_db)
    assert cached_org_model_restriction(workspace.id, "openai") == ["gpt-4o", "gpt-4o-mini"]

    listed = await _service(async_db).list_models(user=owner, key_id=key_id)
    mini = next(row for row in listed.data if row.model == "gpt-4o-mini")
    await _service(async_db).set_model_enabled(user=owner, key_id=key_id, model_id=mini.id, enabled=False)

    assert cached_org_model_restriction(workspace.id, "openai") == ["gpt-4o"]


async def test_switching_every_model_off_serves_none_rather_than_all(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The asymmetry worth a test of its own: absent means unnarrowed and empty
    means deny-all, so an organization that withdrew everything must not read as
    one that never narrowed anything."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    key_id = await _key(async_db, owner)
    await OrgProviderKeyService(async_db).set_org_default_for_user(user=owner, key_id=key_id)
    _discovery(monkeypatch, "gpt-4o")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _service(async_db).refresh_models(user=owner, key_id=key_id)
    listed = await _service(async_db).list_models(user=owner, key_id=key_id)

    await _service(async_db).set_model_enabled(user=owner, key_id=key_id, model_id=listed.data[0].id, enabled=False)

    assert cached_org_model_restriction(workspace.id, "openai") == []


async def test_a_key_that_offers_nothing_stays_unnarrowed(async_db: AsyncSession) -> None:
    """The backward-compatible arm: a key nobody has refreshed keeps reaching
    every model its provider serves."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    key_id = await _key(async_db, owner)
    await OrgProviderKeyService(async_db).set_org_default_for_user(user=owner, key_id=key_id)

    await refresh_org_provider_cache(async_db)

    assert cached_org_model_restriction(workspace.id, "openai") is None


async def test_removing_a_model_keeps_its_rate(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """Usage settled against that rate, and a model offered again should find it
    where it was left rather than reverting to a default unannounced."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, "gpt-4o", "gpt-4o-mini")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "gpt-4o-mini": ("0.15", "0.6")})
    await _service(async_db).refresh_models(user=owner, key_id=key_id)
    listed = await _service(async_db).list_models(user=owner, key_id=key_id)
    removed = next(row for row in listed.data if row.model == "gpt-4o")

    await _service(async_db).remove_model(user=owner, key_id=key_id, model_id=removed.id)

    assert await _offered(async_db, key_id) == {"gpt-4o-mini": True}
    assert sorted(row.model_key for row in await _organization_rates(async_db)) == [
        "openai:gpt-4o",
        "openai:gpt-4o-mini",
    ]


async def test_removing_the_last_model_is_refused_because_it_would_widen_the_key(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent rows mean unnarrowed, so emptying the list is not "serve nothing":
    it returns the key to reaching everything its provider does, which is the
    opposite of what pressing "stop offering" reads as."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    # Read before the refusal below. The unit of work rolls back when the error
    # leaves its block, which expires every instance in the session, and reading
    # an expired attribute back is a lazy load the async session cannot run.
    workspace_id = workspace.id
    key_id = await _key(async_db, owner)
    await OrgProviderKeyService(async_db).set_org_default_for_user(user=owner, key_id=key_id)
    _discovery(monkeypatch, "gpt-4o")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _service(async_db).refresh_models(user=owner, key_id=key_id)
    listed = await _service(async_db).list_models(user=owner, key_id=key_id)

    with pytest.raises(OrgProviderLastModelError):
        await _service(async_db).remove_model(user=owner, key_id=key_id, model_id=listed.data[0].id)

    assert await _offered(async_db, key_id) == {"gpt-4o": True}
    assert cached_org_model_restriction(workspace_id, "openai") == ["gpt-4o"]


async def test_serving_a_model_nothing_prices_is_refused(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of disabled-until-priced. The offer path records such a
    model unserved so it cannot be billed at nothing; the switch must not be a
    way straight past that."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, "gpt-6-unreleased")
    _defaults(monkeypatch, {})
    await _service(async_db).refresh_models(user=owner, key_id=key_id)
    listed = await _service(async_db).list_models(user=owner, key_id=key_id)
    assert listed.data[0].enabled is False

    with pytest.raises(OrgProviderModelUnpricedError):
        await _service(async_db).set_model_enabled(user=owner, key_id=key_id, model_id=listed.data[0].id, enabled=True)

    assert await _offered(async_db, key_id) == {"gpt-6-unreleased": False}


async def test_switching_a_model_off_is_never_refused(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard above is one-directional on purpose: a row that reached the
    served state some other way still has to be withdrawable."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, "gpt-4o", "gpt-4o-mini")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "gpt-4o-mini": ("0.15", "0.6")})
    await _service(async_db).refresh_models(user=owner, key_id=key_id)
    listed = await _service(async_db).list_models(user=owner, key_id=key_id)
    # The rate goes away underneath it, which is the state the guard refuses to
    # enter and must not refuse to leave.
    _defaults(monkeypatch, {})
    for row in await _organization_rates(async_db):
        await async_db.delete(row)
    await async_db.commit()

    updated = await _service(async_db).set_model_enabled(
        user=owner, key_id=key_id, model_id=listed.data[0].id, enabled=False
    )

    assert updated.enabled is False


# --------------------------------------------------------------------------- #
# Authorization
# --------------------------------------------------------------------------- #


async def test_a_plain_member_may_not_read_the_offered_models(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    member = await _member(async_db, organization, role="member", full_name="Member")
    key_id = await _key(async_db, owner)

    with pytest.raises(NotAuthorizedError):
        await _service(async_db).list_models(user=member, key_id=key_id)


async def test_another_organizations_key_is_not_found_rather_than_forbidden(async_db: AsyncSession) -> None:
    """Scoped rather than checked afterward, so an id from elsewhere is
    indistinguishable from one that does not exist."""
    acme = await _organization(async_db, slug="acme")
    other = await _organization(async_db, slug="other")
    acme_owner = await _member(async_db, acme, role="owner", full_name="Acme Owner")
    other_owner = await _member(async_db, other, role="owner", full_name="Other Owner")
    key_id = await _key(async_db, acme_owner)

    with pytest.raises(OrgProviderKeyNotFoundError):
        await _service(async_db).list_models(user=other_owner, key_id=key_id)


async def test_a_model_from_another_key_is_not_found(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    first = await _key(async_db, owner, name="first")
    second = await _key(async_db, owner, provider="anthropic", name="second")
    _discovery(monkeypatch, "gpt-4o")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _service(async_db).refresh_models(user=owner, key_id=first)
    listed = await _service(async_db).list_models(user=owner, key_id=first)

    with pytest.raises(OrgProviderModelNotFoundError):
        await _service(async_db).set_model_enabled(user=owner, key_id=second, model_id=listed.data[0].id, enabled=False)


async def test_no_response_carries_key_material(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """The invariant this whole surface inherits: a stored credential is named by
    its last four characters and never by its value."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    key_id = await _key(async_db, owner)
    _discovery(monkeypatch, "gpt-4o")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})

    refreshed = await _service(async_db).refresh_models(user=owner, key_id=key_id)
    listed = await _service(async_db).list_models(user=owner, key_id=key_id)
    available = await _service(async_db).available_models(user=owner, key_id=key_id)

    for payload in (refreshed, listed, available):
        assert "sk-live-1234" not in payload.model_dump_json()
