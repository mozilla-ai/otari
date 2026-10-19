"""The deployment's hosted providers, against a real database.

Exercised at the service layer: the store, the offer rule, the seeding, the
sweep and the runtime resolve. Model discovery and the community price dataset
are stubbed; a case that dialed for real would be testing the network.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from any_llm import LLMProvider
from any_llm.types.model import Model
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.adapters.model_provider_adapter import HostedProviderModelProviderAdapter
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.providers_exceptions import (
    HostedCatalogEmptyError,
    HostedModelAlreadyOfferedError,
    HostedModelNameRequiredError,
    HostedModelNotFoundError,
    HostedProviderAlreadyExistsError,
    HostedProviderClientArgsUnreadableError,
    HostedProviderKeyRequiredError,
    HostedProviderNotFoundError,
    HostedProviderUnknownOrganizationError,
    HostedProviderUnknownProviderError,
)
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.models.pricing import API_ORIGIN, SEED_ORIGIN, ModelPricing, OrganizationModelPricing
from gateway.models.providers import HostedProvider, HostedProviderModel, ProviderEndpoint
from gateway.models.secret_fields import REDACTED_VALUE
from gateway.models.tenancy import Organization
from gateway.repositories.pricing import ModelPricingRepository, OrganizationModelPricingRepository
from gateway.repositories.providers import (
    HostedProviderModelRepository,
    HostedProviderRepository,
    ProviderEndpointRepository,
)
from gateway.repositories.tenancy import OrganizationRepository, OrgProviderKeyRepository, WorkspaceRepository
from gateway.schemas.providers import (
    HostedModelCreateRequest,
    HostedModelUpdateRequest,
    HostedProviderCreateRequest,
    HostedProviderUpdateRequest,
)
from gateway.services.model_discovery_service import ProviderDiscovery, get_model_cache
from gateway.services.pricing import DeploymentPricingService
from gateway.services.pricing_service import configure_default_pricing
from gateway.services.providers import HostedProviderService, HostedSources
from gateway.services.providers._hosted_catalog import KEEP_FETCH_TOOL, KEEP_OWNED_ENDPOINT
from gateway.services.secret_box import generate_secret_key

pytestmark = pytest.mark.asyncio

KEY = "sk-live-hosted-1234"
DIAL = "gateway.services.providers._hosted_provider_service.test_provider_credentials"
DATASET = "gateway.services.pricing_service.default_model_pricing"


@pytest.fixture(autouse=True)
def _secret_key_and_default_pricing(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A secret box to store under, and the community dataset switched on, as a priced deployment runs."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    configure_default_pricing(True)
    yield
    configure_default_pricing(False)


def _service(db: AsyncSession, config: GatewayConfig | None = None) -> HostedProviderService:
    uow = UnitOfWork(db)
    config = config or GatewayConfig()
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


def _discovery(
    monkeypatch: pytest.MonkeyPatch, *models: str, error: str | None = None, unsupported: bool = False
) -> list[dict[str, object]]:
    """Make every dial answer with ``models``, or with ``error`` and nothing. Returns the calls made."""
    calls: list[dict[str, object]] = []

    async def _stub(impl_name: str, **kwargs: object) -> ProviderDiscovery:
        calls.append({"provider": impl_name, **kwargs})
        return ProviderDiscovery(
            provider=impl_name,
            models=[]
            if error or unsupported
            else [Model(id=m, object="model", created=0, owned_by=impl_name) for m in models],
            error=error,
            discovery_unsupported=unsupported,
        )

    monkeypatch.setattr(DIAL, _stub)
    return calls


def _defaults(monkeypatch: pytest.MonkeyPatch, rates: dict[str, tuple[str, str]]) -> None:
    """Make the community dataset price exactly ``rates``, and nothing else."""

    def _stub(
        provider: str | None, model: str, as_of: object, *, implementation: str | None = None
    ) -> ModelPricing | None:
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

    monkeypatch.setattr(DATASET, _stub)


async def _offered(db: AsyncSession, provider: str) -> dict[str, bool]:
    rows = (await db.execute(select(HostedProviderModel).where(HostedProviderModel.provider == provider))).scalars()
    return {row.model: row.enabled for row in rows}


async def _versions(db: AsyncSession, model_key: str) -> list[ModelPricing]:
    result = await db.execute(
        select(ModelPricing).where(ModelPricing.model_key == model_key).order_by(ModelPricing.effective_at)
    )
    return list(result.scalars().all())


async def _price(
    db: AsyncSession, model_key: str, input_rate: str, output_rate: str, *, origin: str = "config"
) -> None:
    db.add(
        ModelPricing(
            model_key=model_key,
            effective_at=datetime(2026, 1, 1, tzinfo=UTC),
            input_price_per_million=Decimal(input_rate),
            output_price_per_million=Decimal(output_rate),
            pricing_tiers=[],
            unit="tokens",
            origin=origin,
        )
    )
    await db.commit()


async def _organization(db: AsyncSession, slug: str = "acme") -> Organization:
    organization = await OrganizationRepository(db).create_organization(
        name=slug.title(), slug=slug, created_by_user_id=None
    )
    await db.commit()
    return organization


async def _override(db: AsyncSession, organization: Organization, model_key: str) -> None:
    db.add(
        OrganizationModelPricing(
            organization_id=organization.id,
            model_key=model_key,
            input_price_per_million=Decimal("1"),
            output_price_per_million=Decimal("2"),
            pricing_tiers=[],
            unit="tokens",
            origin=API_ORIGIN,
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    await db.commit()


async def _own_key(db: AsyncSession, organization: Organization, provider: str) -> None:
    await OrgProviderKeyRepository(db).create_key(
        organization_id=organization.id,
        provider=provider,
        name="own",
        encrypted_api_key=None,
        last4=None,
        api_base=None,
        client_args=None,
    )
    await db.commit()


async def _provider(service: HostedProviderService, monkeypatch: pytest.MonkeyPatch, *models: str) -> None:
    _discovery(monkeypatch, *models)
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


async def test_creating_a_provider_offers_what_it_lists_and_returns_no_key(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch, "gpt-4o", "gpt-4o-mini")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})

    public = await _service(async_db).create_provider(
        HostedProviderCreateRequest(provider="openai", api_key=KEY, client_args={"aws_secret_access_key": "shh"})
    )

    assert public.provider == "openai"
    assert public.api_key_last4 == "1234"
    assert public.client_args == {"aws_secret_access_key": REDACTED_VALUE}
    assert not any("sk-live" in str(value) for value in public.model_dump().values())
    # Priced by the dataset, so served; nothing prices the mini, so recorded off.
    assert await _offered(async_db, "openai") == {"gpt-4o": True, "gpt-4o-mini": False}
    [seeded] = await _versions(async_db, "openai:gpt-4o")
    assert seeded.input_price_per_million == Decimal("2.5")
    assert seeded.origin == SEED_ORIGIN
    assert await _versions(async_db, "openai:gpt-4o-mini") == []


async def test_client_args_are_encrypted_at_rest_and_read_back_whole(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An IAM secret in the extras is a credential; masking the response does not protect a database copy."""
    _discovery(monkeypatch)
    service = _service(async_db)
    await service.create_provider(
        HostedProviderCreateRequest(
            provider="bedrock", api_key=KEY, client_args={"region": "eu-west-1", "aws_secret_access_key": "shh"}
        )
    )

    [row] = (await async_db.execute(select(HostedProvider))).scalars().all()
    assert row.encrypted_client_args is not None
    assert "shh" not in row.encrypted_client_args
    assert "eu-west-1" not in row.encrypted_client_args
    resolved = await service.resolve("bedrock", organization_id=None)
    assert resolved is not None
    assert resolved.client_args == {"region": "eu-west-1", "aws_secret_access_key": "shh"}


async def test_with_default_pricing_off_nothing_is_seeded_and_models_arrive_switched_off(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate copied from a dataset the operator switched off would be a rate they did not choose."""
    configure_default_pricing(False)
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _price(async_db, "openai:o3", "20", "80")
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o", "o3")

    assert await _offered(async_db, "openai") == {"gpt-4o": False, "o3": True}
    assert await _versions(async_db, "openai:gpt-4o") == []


async def test_a_provider_that_will_not_list_still_lands(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch, error="upstream said no")

    public = await _service(async_db).create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))

    assert public.enabled is True
    assert await _offered(async_db, "openai") == {}


async def test_an_alias_lands_as_the_implementation_it_resolves_to(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)

    public = await _service(async_db).create_provider(
        HostedProviderCreateRequest(provider="openai-compatible", api_key=KEY, api_base="http://vllm.local/v1")
    )

    assert public.provider == "openai"
    assert public.api_base == "http://vllm.local/v1"


async def test_a_provider_the_runtime_cannot_dispatch_to_is_refused(async_db: AsyncSession) -> None:
    with pytest.raises(HostedProviderUnknownProviderError):
        await _service(async_db).create_provider(HostedProviderCreateRequest(provider="not-a-provider", api_key=KEY))


async def test_a_second_provider_for_the_same_implementation_is_refused(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    service = _service(async_db)
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))

    with pytest.raises(HostedProviderAlreadyExistsError):
        await service.create_provider(HostedProviderCreateRequest(provider="openai-compatible", api_key=KEY))


async def test_without_a_secret_key_nothing_is_stored(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY")

    with pytest.raises(SecretBoxUnavailableTenancyError) as refused:
        await _service(async_db).create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))

    assert "Fernet" not in refused.value.message
    assert (await async_db.execute(select(HostedProvider))).scalars().all() == []


async def test_a_toggle_sends_no_key_and_keeps_the_stored_one(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    service = _service(async_db)
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))

    public = await service.update_provider("openai", HostedProviderUpdateRequest(enabled=False))

    assert public.enabled is False
    assert public.api_key_last4 == "1234"
    assert await service.resolve("openai", organization_id=None) is None


async def test_rotating_the_key_replaces_it(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    _discovery(monkeypatch)
    service = _service(async_db)
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))

    public = await service.update_provider("openai", HostedProviderUpdateRequest(api_key="sk-live-rotated-9876"))

    assert public.api_key_last4 == "9876"
    resolved = await service.resolve("openai", organization_id=None)
    assert resolved is not None
    assert resolved.api_key == "sk-live-rotated-9876"


async def test_repointing_the_base_at_another_origin_needs_the_key_again(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this, an operator who cannot read the key could have it sent to an endpoint of their choosing.

    The origin is the scheme, host and port: plain HTTP on the same host, or another port, is another endpoint.
    """
    _discovery(monkeypatch)
    service = _service(async_db)
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY, api_base="https://a/v1"))

    for elsewhere in ("https://evil.example/v1", "http://a/v1", "https://a:9999/v1"):
        with pytest.raises(HostedProviderKeyRequiredError):
            await service.update_provider("openai", HostedProviderUpdateRequest(api_base=elsewhere))
    # The same origin on another path, a cleared base, and a move that brings the key are all fine.
    assert (await service.update_provider("openai", HostedProviderUpdateRequest(api_base="https://a/v2"))).api_base
    assert (await service.update_provider("openai", HostedProviderUpdateRequest(api_base="  "))).api_base is None
    moved = await service.update_provider(
        "openai", HostedProviderUpdateRequest(api_base="https://b/v1", api_key="sk-live-moved-4321")
    )
    assert (moved.api_base, moved.api_key_last4) == ("https://b/v1", "4321")


async def test_echoing_the_mask_back_keeps_the_stored_secret_and_null_clears_it(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    service = _service(async_db)
    await service.create_provider(
        HostedProviderCreateRequest(
            provider="bedrock", api_key=KEY, client_args={"region": "eu-west-1", "aws_secret_access_key": "shh"}
        )
    )

    echoed = await service.update_provider(
        "bedrock",
        HostedProviderUpdateRequest(client_args={"region": "eu-west-2", "aws_secret_access_key": REDACTED_VALUE}),
    )
    resolved = await service.resolve("bedrock", organization_id=None)
    assert resolved is not None
    assert resolved.client_args == {"region": "eu-west-2", "aws_secret_access_key": "shh"}
    assert echoed.client_args == {"region": "eu-west-2", "aws_secret_access_key": REDACTED_VALUE}

    cleared = await service.update_provider("bedrock", HostedProviderUpdateRequest(client_args=None))
    assert cleared.client_args is None


async def test_a_mask_with_nothing_readable_behind_it_is_refused_rather_than_stored(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a key rotation the stored extras will not decrypt, and the form still echoes the mask."""
    _discovery(monkeypatch)
    service = _service(async_db)
    await service.create_provider(
        HostedProviderCreateRequest(provider="bedrock", api_key=KEY, client_args={"aws_secret_access_key": "shh"})
    )
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    with pytest.raises(HostedProviderClientArgsUnreadableError):
        await service.update_provider(
            "bedrock",
            HostedProviderUpdateRequest(
                api_key="sk-live-new-0001", client_args={"aws_secret_access_key": REDACTED_VALUE}
            ),
        )
    # Entered in full, the same request lands.
    full = await service.update_provider(
        "bedrock", HostedProviderUpdateRequest(api_key="sk-live-new-0001", client_args={"aws_secret_access_key": "new"})
    )
    assert full.client_args == {"aws_secret_access_key": REDACTED_VALUE}
    resolved = await service.resolve("bedrock", organization_id=None)
    assert resolved is not None
    assert resolved.client_args == {"aws_secret_access_key": "new"}


async def test_an_unknown_provider_is_not_found_on_every_path(async_db: AsyncSession) -> None:
    service = _service(async_db)

    with pytest.raises(HostedProviderNotFoundError):
        await service.update_provider("openai", HostedProviderUpdateRequest(enabled=False))
    with pytest.raises(HostedProviderNotFoundError):
        await service.delete_provider("openai")
    with pytest.raises(HostedProviderNotFoundError):
        await service.list_models("openai")


async def test_removing_a_provider_takes_its_roster_and_leaves_its_rates(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch, "gpt-4o")
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    service = _service(async_db)
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))

    await service.delete_provider("openai")

    assert (await service.list_providers()).count == 0
    assert await _offered(async_db, "openai") == {}
    assert len(await _versions(async_db, "openai:gpt-4o")) == 1
    assert await service.resolve("openai", organization_id=None) is None


async def test_a_provider_added_again_keeps_following_the_dataset(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance lives on the price version, so removing and re-adding loses nothing."""
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")
    await service.delete_provider("openai")
    await _provider(service, monkeypatch, "gpt-4o")

    _defaults(monkeypatch, {"gpt-4o": ("2", "8")})
    outcome = await service.refresh_models("openai")

    assert outcome.repriced == ["gpt-4o"]
    assert (await _versions(async_db, "openai:gpt-4o"))[-1].input_price_per_million == Decimal("2")


# --------------------------------------------------------------------------- #
# Offered models and their prices
# --------------------------------------------------------------------------- #


async def test_the_model_list_carries_each_models_current_price(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _price(async_db, "openai:o3", "20", "80")
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o", "o3", "unpriced")

    listed = await service.list_models("openai")

    by_model = {model.model: model for model in listed.data}
    assert listed.count == 3
    assert (by_model["gpt-4o"].price_source, by_model["gpt-4o"].input_price_per_million) == ("defaults", 2.5)
    assert (by_model["o3"].price_source, by_model["o3"].input_price_per_million) == ("deployment", 20.0)
    assert by_model["unpriced"].price_source is None
    assert by_model["unpriced"].enabled is False


async def test_adding_a_model_prices_it_at_what_was_asked(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(async_db)
    await _provider(service, monkeypatch)

    public = await service.add_model(
        "openai",
        HostedModelCreateRequest(
            model=" gpt-5 ", input_price_per_million=3, output_price_per_million=12, cache_read_price_per_million=0.3
        ),
    )

    assert public.model == "gpt-5"
    assert (public.price_source, public.input_price_per_million, public.cache_read_price_per_million) == (
        "deployment",
        3.0,
        0.3,
    )
    assert public.enabled is True
    [version] = await _versions(async_db, "openai:gpt-5")
    assert version.origin == API_ORIGIN


async def test_a_model_can_be_priced_per_request(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stored per million requests, and a reprice that names no unit keeps the stored one rather than tokens."""
    service = _service(async_db)
    await _provider(service, monkeypatch)

    added = await service.add_model(
        "openai",
        HostedModelCreateRequest(
            model="rerank", input_price_per_million=2000, output_price_per_million=0, unit="requests"
        ),
    )
    repriced = await service.update_model(
        "openai", added.id, HostedModelUpdateRequest(input_price_per_million=3000, output_price_per_million=0)
    )

    assert (added.unit, added.input_price_per_million) == ("requests", 2000.0)
    assert (repriced.unit, repriced.input_price_per_million) == ("requests", 3000.0)
    assert {version.unit for version in await _versions(async_db, "openai:rerank")} == {"requests"}


async def test_an_offer_without_a_price_seeds_the_default(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-5": ("3", "12")})
    service = _service(async_db)
    await _provider(service, monkeypatch)

    public = await service.add_model("openai", HostedModelCreateRequest(model="gpt-5"))

    assert (public.price_source, public.input_price_per_million, public.enabled) == ("defaults", 3.0, True)
    assert len(await _versions(async_db, "openai:gpt-5")) == 1


async def test_a_model_nothing_prices_arrives_disabled(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    _defaults(monkeypatch, {})
    service = _service(async_db)
    await _provider(service, monkeypatch)

    public = await service.add_model("openai", HostedModelCreateRequest(model="brand-new"))

    assert public.enabled is False
    assert public.price_source is None
    assert await _versions(async_db, "openai:brand-new") == []


async def test_a_model_offered_twice_is_refused_and_a_blank_name_too(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")

    with pytest.raises(HostedModelAlreadyOfferedError):
        await service.add_model("openai", HostedModelCreateRequest(model="gpt-4o"))
    with pytest.raises(HostedModelNameRequiredError):
        await service.add_model("openai", HostedModelCreateRequest(model="   "))


async def test_a_toggle_travels_alone_and_repricing_makes_the_rate_the_operators(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")
    [model] = (await service.list_models("openai")).data

    toggled = await service.update_model("openai", model.id, HostedModelUpdateRequest(enabled=False))
    assert toggled.enabled is False
    assert toggled.price_source == "defaults"

    repriced = await service.update_model(
        "openai", model.id, HostedModelUpdateRequest(input_price_per_million=5, output_price_per_million=20)
    )
    assert (repriced.price_source, repriced.input_price_per_million) == ("deployment", 5.0)
    assert repriced.enabled is False
    assert [version.origin for version in await _versions(async_db, "openai:gpt-4o")] == [SEED_ORIGIN, API_ORIGIN]


async def test_repricing_keeps_the_cache_rates_it_does_not_name(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reprice that sends input and output only carries the stored cache rates onto the new version."""
    service = _service(async_db)
    await _provider(service, monkeypatch)
    added = await service.add_model(
        "openai",
        HostedModelCreateRequest(
            model="gpt-5",
            input_price_per_million=3,
            output_price_per_million=12,
            cache_read_price_per_million=0.3,
            cache_write_price_per_million=3.75,
        ),
    )

    repriced = await service.update_model(
        "openai", added.id, HostedModelUpdateRequest(input_price_per_million=4, output_price_per_million=16)
    )

    assert (repriced.cache_read_price_per_million, repriced.cache_write_price_per_million) == (0.3, 3.75)
    first, second = await _versions(async_db, "openai:gpt-5")
    assert (first.input_price_per_million, second.input_price_per_million) == (Decimal("3"), Decimal("4"))
    assert (second.cache_read_price_per_million, second.cache_write_price_per_million) == (
        Decimal("0.3"),
        Decimal("3.75"),
    )


async def test_new_base_rates_drop_a_seeds_tiers_and_carry_an_operators(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tier's rates are absolute, so they stay with the base rates they were written beside."""
    tier = {"min_input_tokens": 200_000, "input_price_per_million": 5.0, "output_price_per_million": 15.0}

    def _tiered(
        provider: str | None, model: str, as_of: object, *, implementation: str | None = None
    ) -> ModelPricing | None:
        return ModelPricing(
            model_key=f"{provider}:{model}",
            input_price_per_million=Decimal("2.5"),
            output_price_per_million=Decimal("10"),
            pricing_tiers=[tier],
            unit="tokens",
        )

    monkeypatch.setattr(DATASET, _tiered)
    service = _service(async_db)
    await _provider(service, monkeypatch, "gemini-pro")
    [model] = (await service.list_models("openai")).data
    [seed] = await _versions(async_db, "openai:gemini-pro")
    assert (seed.origin, seed.pricing_tiers) == (SEED_ORIGIN, [tier])

    await service.update_model(
        "openai", model.id, HostedModelUpdateRequest(input_price_per_million=3, output_price_per_million=12)
    )
    _, repriced = await _versions(async_db, "openai:gemini-pro")
    assert (repriced.origin, repriced.pricing_tiers) == (API_ORIGIN, [])

    repriced.pricing_tiers = [tier]
    await async_db.commit()
    await service.update_model(
        "openai", model.id, HostedModelUpdateRequest(input_price_per_million=4, output_price_per_million=16)
    )
    *_, latest = await _versions(async_db, "openai:gemini-pro")
    assert (latest.input_price_per_million, latest.pricing_tiers) == (Decimal("4"), [tier])


async def test_a_model_under_another_provider_is_not_found(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")
    _discovery(monkeypatch, "claude")
    await service.create_provider(HostedProviderCreateRequest(provider="anthropic", api_key=KEY))
    [model] = (await service.list_models("openai")).data

    with pytest.raises(HostedModelNotFoundError):
        await service.update_model("anthropic", model.id, HostedModelUpdateRequest(enabled=False))
    with pytest.raises(HostedModelNotFoundError):
        await service.remove_model("openai", uuid.uuid4())


async def test_removing_a_model_keeps_its_pricing_history(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")
    [model] = (await service.list_models("openai")).data

    await service.remove_model("openai", model.id)

    assert await _offered(async_db, "openai") == {}
    assert len(await _versions(async_db, "openai:gpt-4o")) == 1


async def test_refresh_offers_only_what_is_new_and_moves_a_seeded_price(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "gpt-5": ("3", "12")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")

    _defaults(monkeypatch, {"gpt-4o": ("2", "8"), "gpt-5": ("3", "12")})
    _discovery(monkeypatch, "gpt-4o", "gpt-5")
    outcome = await service.refresh_models("openai")

    assert outcome.added == ["gpt-5"]
    assert outcome.repriced == ["gpt-4o"]
    assert outcome.count == 2
    assert [version.input_price_per_million for version in await _versions(async_db, "openai:gpt-4o")] == [
        Decimal("2.5"),
        Decimal("2"),
    ]


async def test_refresh_leaves_a_price_an_operator_chose(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")
    [model] = (await service.list_models("openai")).data
    await service.update_model(
        "openai", model.id, HostedModelUpdateRequest(input_price_per_million=5, output_price_per_million=20)
    )

    _defaults(monkeypatch, {"gpt-4o": ("2", "8")})
    outcome = await service.refresh_models("openai")

    assert outcome.repriced == []
    assert (await _versions(async_db, "openai:gpt-4o"))[-1].input_price_per_million == Decimal("5")


async def test_refresh_prices_a_model_whose_rates_were_deleted_or_switches_it_off(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The offer rule holds after the fact: an unpriced model is not served."""
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "o3": ("20", "80")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o", "o3")
    await async_db.execute(delete(ModelPricing).where(ModelPricing.model_key.in_(["openai:gpt-4o", "openai:o3"])))
    await async_db.commit()

    _defaults(monkeypatch, {"gpt-4o": ("2", "8")})
    outcome = await service.refresh_models("openai")

    assert outcome.repriced == ["gpt-4o"]
    assert await _offered(async_db, "openai") == {"gpt-4o": True, "o3": False}
    [reseeded] = await _versions(async_db, "openai:gpt-4o")
    assert (reseeded.origin, reseeded.input_price_per_million) == (SEED_ORIGIN, Decimal("2"))


async def test_a_model_the_operator_switched_off_stays_off_when_a_refresh_prices_it_again(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row does not record why it is off, so a refresh only ever turns the switch off."""
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")
    [model] = (await service.list_models("openai")).data
    await service.update_model("openai", model.id, HostedModelUpdateRequest(enabled=False))
    await async_db.execute(delete(ModelPricing).where(ModelPricing.model_key == "openai:gpt-4o"))
    await async_db.commit()

    outcome = await service.refresh_models("openai")

    assert outcome.repriced == ["gpt-4o"]
    assert await _offered(async_db, "openai") == {"gpt-4o": False}
    assert await service.resolve("openai", "gpt-4o", organization_id=None) is None


async def test_two_refreshes_racing_leave_the_loser_with_a_conflict_not_a_crash(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {})
    service = _service(async_db)
    await _provider(service, monkeypatch)
    _discovery(monkeypatch, "gpt-5")
    # The other refresh offered the model between this one's read and its write.
    monkeypatch.setattr(HostedProviderModelRepository, "names_for_provider", _nothing_offered)
    await service.add_model("openai", HostedModelCreateRequest(model="gpt-5"))

    with pytest.raises(HostedModelAlreadyOfferedError):
        await service.refresh_models("openai")


async def _nothing_offered(_self: object, _provider: str) -> set[str]:
    return set()


async def test_refresh_reports_a_provider_that_will_not_answer(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")

    _discovery(monkeypatch, unsupported=True)
    outcome = await service.refresh_models("openai")

    assert outcome.discovery_unsupported is True
    assert outcome.count == 1


async def test_available_models_lists_what_the_credential_can_serve_without_storing(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(async_db)
    await _provider(service, monkeypatch)
    calls = _discovery(monkeypatch, "b", "a")

    available = await service.available_models("openai")

    assert available.models == ["a", "b"]
    assert calls[-1]["api_key"] == KEY
    assert await _offered(async_db, "openai") == {}


async def test_an_undecryptable_key_reports_instead_of_dialing(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(async_db)
    await _provider(service, monkeypatch)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    calls = _discovery(monkeypatch, "a")

    available = await service.available_models("openai")
    refresh = await service.refresh_models("openai")

    assert available.error is not None and "OTARI_SECRET_KEY" in available.error
    assert refresh.error is not None
    assert calls == []


# --------------------------------------------------------------------------- #
# The catalog sweep
# --------------------------------------------------------------------------- #


async def test_a_sweep_takes_unoffered_prices_off_the_list_and_the_preview_names_them(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _price(async_db, "nebius:llama", "1", "2")
    await _price(async_db, "otari:web_search", "0", "0")
    await _price(async_db, "my-vllm:llama", "1", "2")
    service = _service(async_db, GatewayConfig(providers={"my-vllm": {"provider_type": "openai", "api_key": "x"}}))
    await _provider(service, monkeypatch, "gpt-4o")

    preview = await service.refresh_catalog(apply=False)
    assert preview.removed == ["nebius:llama"]
    assert preview.removed_price_rows == 1
    assert {group.reason for group in preview.kept} == {
        "prices a gateway-run tool, not a model",
        "served by a provider instance configured on this deployment",
    }
    assert len(await _versions(async_db, "nebius:llama")) == 1

    applied = await service.refresh_catalog(apply=True)
    assert (applied.removed, applied.removed_price_rows, applied.removed_override_rows) == (
        preview.removed,
        preview.removed_price_rows,
        preview.removed_override_rows,
    )
    assert await _versions(async_db, "nebius:llama") == []
    assert len(await _versions(async_db, "otari:web_search")) == 1
    assert len(await _versions(async_db, "openai:gpt-4o")) == 1


async def test_a_sweep_takes_a_rate_an_organizations_own_key_reaches_and_spares_only_that_organizations_override(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rate the deployment no longer offers goes even under a provider some organization holds a key for.

    Kept, it would let every organization be served on the deployment's own
    key for a model the operator never offered. The organization with a key of
    its own keeps its override, which is where its requests settle; an
    organization without one loses its override with the rate.
    """
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _price(async_db, "openai:gpt-3.5-turbo-0613", "1", "2")
    with_key = await _organization(async_db, "with-key")
    without_key = await _organization(async_db, "without-key")
    await _own_key(async_db, with_key, "openai")
    await _override(async_db, with_key, "openai:gpt-3.5-turbo-0613")
    await _override(async_db, without_key, "openai:gpt-3.5-turbo-0613")
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")

    preview = await service.refresh_catalog(apply=False)
    applied = await service.refresh_catalog(apply=True)

    assert preview.removed == applied.removed == ["openai:gpt-3.5-turbo-0613"]
    assert preview.removed_override_rows == applied.removed_override_rows == 1
    assert preview.kept == applied.kept == []
    assert await _versions(async_db, "openai:gpt-3.5-turbo-0613") == []
    remaining = (await async_db.execute(select(OrganizationModelPricing))).scalars().all()
    assert [row.organization_id for row in remaining] == [with_key.id]


async def test_a_sweep_takes_the_overrides_above_what_it_removes_whatever_the_rates_spelling(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An override is keyed ``provider:model`` even when the rate it sits above was written ``provider/model``."""
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _price(async_db, "nebius:llama", "1", "2")
    await _price(async_db, "mistral/mistral-tiny", "1", "2")
    organization = await _organization(async_db)
    await _override(async_db, organization, "nebius:llama")
    await _override(async_db, organization, "mistral:mistral-tiny")
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")

    preview = await service.refresh_catalog(apply=False)
    assert preview.removed == ["mistral/mistral-tiny", "nebius:llama"]
    assert (preview.removed_price_rows, preview.removed_override_rows) == (2, 2)
    assert len((await async_db.execute(select(OrganizationModelPricing))).scalars().all()) == 2

    applied = await service.refresh_catalog(apply=True)
    assert (applied.removed_price_rows, applied.removed_override_rows) == (2, 2)
    assert await _versions(async_db, "mistral/mistral-tiny") == []
    assert (await async_db.execute(select(OrganizationModelPricing))).scalars().all() == []

    again = await service.refresh_catalog(apply=True)
    assert (again.removed, again.removed_price_rows, again.removed_override_rows) == ([], 0, 0)


async def test_a_sweep_keeps_the_rates_fetch_tools_are_priced_at(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The built-in fetcher's rate and a fetch instance's survive a sweep, as a search tool's do."""
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _price(async_db, "builtin:builtin_fetch", "1000", "0")
    await _price(async_db, "tabstack:tabstack", "2000", "0")
    service = _service(async_db, GatewayConfig(fetch_tools={"tabstack": {"provider": "tabstack", "api_key": "k"}}))
    await _provider(service, monkeypatch, "gpt-4o")

    applied = await service.refresh_catalog(apply=True)

    assert applied.removed == []
    assert [(group.reason, group.models) for group in applied.kept] == [
        (KEEP_FETCH_TOOL, ["builtin:builtin_fetch", "tabstack:tabstack"])
    ]
    assert len(await _versions(async_db, "builtin:builtin_fetch")) == 1
    assert len(await _versions(async_db, "tabstack:tabstack")) == 1


async def test_a_sweep_keeps_the_rates_an_owned_endpoint_settles_at(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _price(async_db, "my-vllm:llama", "1", "2")
    organization = await _organization(async_db, "owner")
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Default", organization_id=organization.id, created_by_user_id=None
    )
    async_db.add(
        ProviderEndpoint(
            workspace_id=workspace.id,
            name="my-vllm",
            provider="openai",
            api_base="https://vllm.example.com/v1",
        )
    )
    await async_db.commit()
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")

    applied = await service.refresh_catalog(apply=True)

    assert applied.removed == []
    assert [(group.reason, group.models) for group in applied.kept] == [(KEEP_OWNED_ENDPOINT, ["my-vllm:llama"])]


async def test_a_sweep_offers_what_a_provider_newly_lists_before_judging(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "gpt-5": ("3", "12")})
    await _price(async_db, "openai:gpt-5", "9", "9")
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")

    _discovery(monkeypatch, "gpt-4o", "gpt-5")
    preview = await service.refresh_catalog(apply=False)
    assert preview.removed == []
    assert preview.providers[0].added == ["gpt-5"]
    assert await _offered(async_db, "openai") == {"gpt-4o": True}

    applied = await service.refresh_catalog(apply=True)
    assert applied.providers[0].added == ["gpt-5"]
    assert await _offered(async_db, "openai") == {"gpt-4o": True, "gpt-5": True}


async def test_a_sweep_offers_nothing_on_a_provider_removed_while_it_dialed(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the foreign key the roster would keep rows with nothing to own them."""
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "gpt-5": ("3", "12")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")
    _discovery(monkeypatch, "claude")
    await service.create_provider(HostedProviderCreateRequest(provider="anthropic", api_key=KEY))

    async def _dial_and_remove(impl_name: str, **_: object) -> ProviderDiscovery:
        if impl_name == "openai":
            await service.delete_provider("openai")
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id="gpt-5", object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr(DIAL, _dial_and_remove)
    applied = await service.refresh_catalog(apply=True)

    [anthropic, openai] = sorted(applied.providers, key=lambda answer: answer.provider)
    assert openai.added == []
    assert openai.error is not None
    assert anthropic.added == ["gpt-5"]
    assert await _offered(async_db, "openai") == {}


async def test_a_preview_drops_what_a_provider_switched_off_mid_dial_listed_as_the_apply_does(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the preview keeps a price on the strength of a listing the apply discards."""
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "claude": ("3", "15")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")
    _discovery(monkeypatch, "claude")
    await service.create_provider(HostedProviderCreateRequest(provider="anthropic", api_key=KEY))
    await _price(async_db, "openai:gpt-5", "3", "12")

    async def _dial_and_switch_off(impl_name: str, **_: object) -> ProviderDiscovery:
        if impl_name == "openai":
            await service.update_provider("openai", HostedProviderUpdateRequest(enabled=False))
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id="gpt-5", object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr(DIAL, _dial_and_switch_off)
    preview = await service.refresh_catalog(apply=False)
    applied = await service.refresh_catalog(apply=True)

    openai = next(answer for answer in preview.providers if answer.provider == "openai")
    assert (openai.added, openai.error is not None) == ([], True)
    assert "openai:gpt-5" in preview.removed
    assert preview.removed == applied.removed


async def test_a_sweep_refuses_while_nothing_is_offered(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _price(async_db, "nebius:llama", "1", "2")
    service = _service(async_db)
    await _provider(service, monkeypatch)

    with pytest.raises(HostedCatalogEmptyError):
        await service.refresh_catalog(apply=True)

    assert len(await _versions(async_db, "nebius:llama")) == 1


async def test_a_sweep_reports_a_provider_whose_key_will_not_decrypt_without_dialing_it(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    calls = _discovery(monkeypatch, "gpt-5")

    preview = await service.refresh_catalog(apply=False)

    [answer] = preview.providers
    assert answer.credential_unreadable is True
    assert calls == []


# --------------------------------------------------------------------------- #
# The runtime path
# --------------------------------------------------------------------------- #


async def test_the_runtime_resolves_the_row_the_surface_wrote(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    service = _service(async_db)
    _discovery(monkeypatch, "gpt-4o")
    await service.create_provider(
        HostedProviderCreateRequest(provider="openai", api_key=KEY, api_base="http://a/v1", client_args={"x": 1})
    )

    resolved = await service.resolve("openai", "gpt-4o", organization_id=None)

    assert resolved is not None
    assert (resolved.api_key, resolved.api_base, resolved.client_args) == (KEY, "http://a/v1", {"x": 1})
    assert await service.resolve("openai-compatible", "gpt-4o", organization_id=None) is not None
    assert await service.resolve("anthropic", organization_id=None) is None


async def test_the_adapter_answers_an_alias_under_the_provider_it_folds_to(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An alias served on the openai key settles as openai, the key its rates and model switches are under."""
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    service = _service(async_db)
    _discovery(monkeypatch, "gpt-4o")
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))
    adapter = HostedProviderModelProviderAdapter(service)

    for asked in ("openai", "openai-compatible"):
        credential = await adapter.resolve_hosted_credential(
            organization_id=uuid.uuid4(), workspace_id=None, provider=asked, model="gpt-4o"
        )
        assert credential is not None, asked
        assert credential.response_provider == "openai", asked


async def test_only_a_model_switched_on_under_the_provider_serves(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is two-sided: a model switched off and a model with no row are both refused."""
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "gpt-4o-mini": ("0.15", "0.6")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o", "gpt-4o-mini")
    models = {model.model: model for model in (await service.list_models("openai")).data}
    assert await service.resolve("openai", "gpt-4o", organization_id=None) is not None

    await service.update_model("openai", models["gpt-4o"].id, HostedModelUpdateRequest(enabled=False))
    await service.remove_model("openai", models["gpt-4o-mini"].id)

    assert await service.resolve("openai", "gpt-4o", organization_id=None) is None
    assert await service.resolve("openai", "gpt-4o-mini", organization_id=None) is None
    assert await service.resolve("openai", "o3", organization_id=None) is None
    assert await service.resolve("openai", organization_id=None) is not None


async def test_the_listing_answers_for_exactly_what_resolves(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "claude": ("3", "15")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o", "unpriced")
    _discovery(monkeypatch, "claude")
    await service.create_provider(HostedProviderCreateRequest(provider="anthropic", api_key=KEY, enabled=False))

    advertised = await service.serveable_models(organization_id=None)

    assert advertised == {"openai": frozenset({"gpt-4o"})}
    assert await service.resolve("anthropic", "claude", organization_id=None) is None


async def test_a_restricted_provider_serves_its_organization_alone(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other organization is answered as if the provider were not configured, in the listing as in the resolve."""
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    owner = await _organization(async_db, "owner")
    other = await _organization(async_db, "other")
    service = _service(async_db)
    _discovery(monkeypatch, "gpt-4o")
    created = await service.create_provider(
        HostedProviderCreateRequest(provider="openai", api_key=KEY, restricted_to_organization_id=owner.id)
    )

    assert created.restricted_to_organization_id == owner.id
    assert await service.resolve("openai", "gpt-4o", organization_id=owner.id) is not None
    assert await service.resolve("openai", "gpt-4o", organization_id=other.id) is None
    assert await service.serveable_models(organization_id=owner.id) == {"openai": frozenset({"gpt-4o"})}
    assert await service.serveable_models(organization_id=other.id) == {}
    # The deployment's own view sees every provider.
    assert await service.serveable_models(organization_id=None) == {"openai": frozenset({"gpt-4o"})}


async def test_an_omitted_restriction_is_kept_and_a_null_one_lifts_it(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = await _organization(async_db, "owner")
    other = await _organization(async_db, "other")
    service = _service(async_db)
    _discovery(monkeypatch)
    await service.create_provider(
        HostedProviderCreateRequest(provider="openai", api_key=KEY, restricted_to_organization_id=owner.id)
    )

    kept = await service.update_provider("openai", HostedProviderUpdateRequest(enabled=True))
    assert kept.restricted_to_organization_id == owner.id
    assert await service.resolve("openai", organization_id=other.id) is None

    lifted = await service.update_provider(
        "openai", HostedProviderUpdateRequest.model_validate({"restricted_to_organization_id": None})
    )
    assert lifted.restricted_to_organization_id is None
    assert await service.resolve("openai", organization_id=other.id) is not None


async def test_a_restriction_naming_no_organization_is_refused(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(async_db)
    _discovery(monkeypatch)
    unknown = uuid.uuid4()

    with pytest.raises(HostedProviderUnknownOrganizationError):
        await service.create_provider(
            HostedProviderCreateRequest(provider="openai", api_key=KEY, restricted_to_organization_id=unknown)
        )
    assert (await async_db.execute(select(HostedProvider))).scalars().all() == []

    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))
    with pytest.raises(HostedProviderUnknownOrganizationError):
        await service.update_provider("openai", HostedProviderUpdateRequest(restricted_to_organization_id=unknown))
    assert await service.resolve("openai", organization_id=uuid.uuid4()) is not None


async def test_an_organization_a_provider_is_restricted_to_cannot_be_deleted_under_it(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither widened to every organization nor deleted with the organization: the delete is refused."""
    owner_id = (await _organization(async_db, "owner")).id
    service = _service(async_db)
    _discovery(monkeypatch)
    await service.create_provider(
        HostedProviderCreateRequest(provider="openai", api_key=KEY, restricted_to_organization_id=owner_id)
    )

    with pytest.raises(IntegrityError):
        await async_db.execute(delete(Organization).where(col(Organization.id) == owner_id))
    await async_db.rollback()

    row = (await async_db.execute(select(HostedProvider))).scalar_one()
    assert row.restricted_to_organization_id == owner_id


async def test_a_credential_no_configured_key_can_read_is_unserved_rather_than_fatal(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(async_db)
    await _provider(service, monkeypatch)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    assert await service.resolve("openai", organization_id=None) is None
    assert await service.serveable_models(organization_id=None) == {}
    assert (await service.list_providers()).data[0].api_key_last4 == "1234"


# --------------------------------------------------------------------------- #
# Discovery and health
# --------------------------------------------------------------------------- #


async def test_the_discovery_dials_are_the_deployments_view(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every provider switched on, a restricted one included; one switched off is not there."""
    owner = await _organization(async_db, "owner")
    service = _service(async_db)
    _discovery(monkeypatch, "gpt-4o")
    await service.create_provider(
        HostedProviderCreateRequest(provider="openai", api_key=KEY, restricted_to_organization_id=owner.id)
    )
    await service.create_provider(HostedProviderCreateRequest(provider="mistral", api_key=KEY))
    await service.create_provider(HostedProviderCreateRequest(provider="anthropic", api_key=KEY, enabled=False))

    dials = await service.discovery_dials()

    assert set(dials) == {"openai", "mistral"}


async def test_a_discovery_dial_lists_on_the_stored_credential(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(async_db)
    calls = _discovery(monkeypatch, "gpt-4o")
    await service.create_provider(
        HostedProviderCreateRequest(
            provider="openai", api_key=KEY, api_base="https://hosted.test/v1", client_args={"timeout": 9}
        )
    )
    calls.clear()

    discovery = await (await service.discovery_dials())["openai"]()

    assert [model.id for model in discovery.models] == ["gpt-4o"]
    [call] = calls
    assert call["provider"] == "openai"
    assert call["api_key"] == KEY
    assert call["api_base"] == "https://hosted.test/v1"
    assert call["client_args"] == {"timeout": 9}


async def test_a_discovery_dial_hands_any_llm_the_stored_endpoint(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The endpoint and client extras reach ``alist_models``, as a configured instance's do.

    Whether the provider's own client then honors them is any-llm's: its groq
    provider drops ``api_base`` when it builds the client.
    """
    service = _service(async_db)
    listed: list[dict[str, object]] = []

    async def _alist_models(**kwargs: object) -> list[Model]:
        listed.append(kwargs)
        return [Model(id="llama-3.3-70b", object="model", created=0, owned_by="groq")]

    monkeypatch.setattr("gateway.services.model_discovery_service.alist_models", _alist_models)
    await service.create_provider(
        HostedProviderCreateRequest(
            provider="groq", api_key=KEY, api_base="http://groq.test/openai/v1", client_args={"timeout": 9}
        )
    )
    listed.clear()

    discovery = await (await service.discovery_dials())["groq"]()

    assert discovery.error is None
    [call] = listed
    assert call["provider"] == LLMProvider.GROQ
    assert call["api_key"] == KEY
    assert call["api_base"] == "http://groq.test/openai/v1"
    assert call["client_args"] == {"timeout": 9}


async def test_an_undecryptable_provider_is_reported_to_discovery_without_a_dial(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(async_db)
    calls = _discovery(monkeypatch, "gpt-4o")
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    calls.clear()

    discovery = await (await service.discovery_dials())["openai"]()

    assert discovery.models == []
    assert discovery.error is not None and "OTARI_SECRET_KEY" in discovery.error
    assert KEY not in discovery.error
    assert calls == []


async def test_a_change_to_a_provider_drops_its_cached_listing(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next read dials the credential as it now stands, not the listing made on the old one."""
    service = _service(async_db)
    _discovery(monkeypatch, "gpt-4o")
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))
    cache = get_model_cache()
    cache.set("openai", [Model(id="stale", object="model", created=0, owned_by="openai")])

    await service.update_provider("openai", HostedProviderUpdateRequest(api_key="sk-live-rotated-5678"))
    assert cache.stale("openai") is None

    cache.set("openai", [Model(id="stale", object="model", created=0, owned_by="openai")])
    await service.delete_provider("openai")
    assert cache.stale("openai") is None
