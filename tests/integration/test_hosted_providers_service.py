"""The deployment's hosted providers, against a real database.

Exercised at the service layer: what is pinned is the store, the offer rule,
the seeding, the sweep and the runtime resolve, none of which is about whether
any-llm can reach a provider. Model discovery and the community price dataset
are stubbed throughout; a case that dialed for real would be testing the
network.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from any_llm.types.model import Model
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.providers_exceptions import (
    HostedCatalogEmptyError,
    HostedModelAlreadyOfferedError,
    HostedModelNameRequiredError,
    HostedModelNotFoundError,
    HostedProviderAlreadyExistsError,
    HostedProviderNotFoundError,
    HostedProviderSecretStorageError,
    HostedProviderUnknownProviderError,
)
from gateway.models.pricing import API_ORIGIN, ModelPricing, OrganizationModelPricing
from gateway.models.providers import HostedProvider, HostedProviderModel
from gateway.models.secret_fields import REDACTED_VALUE
from gateway.models.tenancy import Organization
from gateway.repositories.pricing import ModelPricingRepository
from gateway.repositories.providers import HostedProviderModelRepository, HostedProviderRepository
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

pytestmark = pytest.mark.asyncio

KEY = "sk-live-hosted-1234"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


def _service(db: AsyncSession, config: GatewayConfig | None = None) -> HostedProviderService:
    uow = UnitOfWork(db)
    return HostedProviderService(
        uow,
        config=config or GatewayConfig(),
        providers=HostedProviderRepository(uow),
        models=HostedProviderModelRepository(uow),
        pricing=DeploymentPricingService(uow, pricing=ModelPricingRepository(uow)),
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

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)
    return calls


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

    monkeypatch.setattr("gateway.services.pricing._deployment_pricing_service.default_model_pricing", _stub)


async def _offered(db: AsyncSession, provider: str) -> dict[str, bool]:
    rows = (
        await db.execute(select(HostedProviderModel).where(col(HostedProviderModel.provider) == provider))
    ).scalars()
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
    assert seeded.origin == API_ORIGIN
    assert await _versions(async_db, "openai:gpt-4o-mini") == []


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

    with pytest.raises(HostedProviderSecretStorageError):
        await _service(async_db).create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))

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
    assert await service.resolve("openai") is None


async def test_rotating_the_key_replaces_it(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    _discovery(monkeypatch)
    service = _service(async_db)
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))

    public = await service.update_provider("openai", HostedProviderUpdateRequest(api_key="sk-live-rotated-9876"))

    assert public.api_key_last4 == "9876"
    resolved = await service.resolve("openai")
    assert resolved is not None
    assert resolved.api_key == "sk-live-rotated-9876"


async def test_a_blank_endpoint_clears_it_rather_than_storing_an_empty_string(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _discovery(monkeypatch)
    service = _service(async_db)
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY, api_base="http://a/v1"))

    public = await service.update_provider("openai", HostedProviderUpdateRequest(api_base="  "))

    assert public.api_base is None


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
    resolved = await service.resolve("bedrock")
    assert resolved is not None
    assert resolved.client_args == {"region": "eu-west-2", "aws_secret_access_key": "shh"}
    assert echoed.client_args == {"region": "eu-west-2", "aws_secret_access_key": REDACTED_VALUE}

    cleared = await service.update_provider("bedrock", HostedProviderUpdateRequest(client_args=None))
    assert cleared.client_args is None


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
    assert await service.resolve("openai") is None


# --------------------------------------------------------------------------- #
# Offered models and their prices
# --------------------------------------------------------------------------- #


async def _provider(service: HostedProviderService, monkeypatch: pytest.MonkeyPatch, *models: str) -> None:
    _discovery(monkeypatch, *models)
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))


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
    assert len(await _versions(async_db, "openai:gpt-4o")) == 2


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
    assert applied.removed == ["nebius:llama"]
    assert await _versions(async_db, "nebius:llama") == []
    assert len(await _versions(async_db, "otari:web_search")) == 1
    assert len(await _versions(async_db, "openai:gpt-4o")) == 1


async def test_a_sweep_spares_the_rate_a_tenant_set_on_its_own_key(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _price(async_db, "nebius:llama", "1", "2")
    with_key = await _organization(async_db, "with-key")
    without_key = await _organization(async_db, "without-key")
    await _own_key(async_db, with_key, "nebius")
    await _override(async_db, with_key, "nebius:llama")
    await _override(async_db, without_key, "nebius:llama")
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")

    preview = await service.refresh_catalog(apply=False)
    applied = await service.refresh_catalog(apply=True)

    assert preview.removed_override_rows == 1
    assert applied.removed_override_rows == 1
    remaining = (await async_db.execute(select(OrganizationModelPricing))).scalars().all()
    assert [row.organization_id for row in remaining] == [with_key.id]


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

    resolved = await service.resolve("openai", "gpt-4o")

    assert resolved is not None
    assert (resolved.api_key, resolved.api_base, resolved.client_args) == (KEY, "http://a/v1", {"x": 1})
    assert await service.resolve("anthropic") is None


async def test_a_disabled_model_is_refused_and_an_unlisted_one_serves(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o")
    [model] = (await service.list_models("openai")).data
    await service.update_model("openai", model.id, HostedModelUpdateRequest(enabled=False))

    assert await service.resolve("openai", "gpt-4o") is None
    assert await service.resolve("openai", "gpt-4o-mini") is not None
    assert await service.resolve("openai") is not None


async def test_the_listing_answers_for_exactly_what_resolves(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "claude": ("3", "15")})
    service = _service(async_db)
    await _provider(service, monkeypatch, "gpt-4o", "unpriced")
    _discovery(monkeypatch, "claude")
    await service.create_provider(HostedProviderCreateRequest(provider="anthropic", api_key=KEY, enabled=False))

    advertised = await service.serveable_models()

    # The switched-off provider is absent, and under the serving one only the
    # model that is switched on is advertised.
    assert advertised == {"openai": frozenset({"gpt-4o"})}
    assert await service.resolve("anthropic", "claude") is None


async def test_a_credential_no_configured_key_can_read_is_unserved_rather_than_fatal(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(async_db)
    await _provider(service, monkeypatch)
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    assert await service.resolve("openai") is None
    assert await service.serveable_models() == {}
    # The row is intact and the page still lists it, by its tail.
    assert (await service.list_providers()).data[0].api_key_last4 == "1234"
