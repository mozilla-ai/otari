"""The offer rule: which models a hosted provider serves, and the rate each is seeded at.

A rate is written to the deployment price list. The deployment settles the
upstream bill for a hosted model, so its rate is the deployment's and applies to
every organization that holds no override of its own. That list is also what
lists an undiscovered model, so offering a model and pricing it are one write
seen from two sides.

A model offered without a rate is seeded with the community default, as a
version carrying ``SEED_ORIGIN``, while the deployment consults that dataset at
all. A refresh moves a seeded version with the dataset and leaves every other
origin alone, because those are rates somebody chose. A model nothing prices
is offered but not served, so a model the dataset has not caught up with
cannot be billed at nothing.

Every method that writes runs inside a block the caller opened. The dataset
walk runs off the loop and outside any block.
"""

from collections.abc import Collection, Mapping, Sequence

from gateway.exceptions.providers_exceptions import HostedModelAlreadyOfferedError
from gateway.models.pricing import SEED_ORIGIN, ModelPricing, PriceSource
from gateway.models.providers import HostedProviderModel
from gateway.repositories.providers import HostedModelConflict, HostedProviderModelRepository
from gateway.schemas.providers import HostedModelCreateRequest, HostedModelRates, HostedModelUpdateRequest
from gateway.services.pricing import DeploymentPricingService
from gateway.services.pricing_service import default_pricing_enabled, normalize_effective_at

PRICE_SOURCE_DEFAULT: PriceSource = "defaults"
PRICE_SOURCE_DEPLOYMENT: PriceSource = "deployment"

_CACHE_RATE_FIELDS = (
    "cache_read_price_per_million",
    "cache_write_price_per_million",
    "cache_write_1h_price_per_million",
)


def _key(provider: str, model: str) -> str:
    return f"{provider}:{model}"


class HostedOffers:
    """Offer, price and reprice the models a hosted provider serves."""

    def __init__(self, models: HostedProviderModelRepository, pricing: DeploymentPricingService) -> None:
        self.models = models
        self.pricing = pricing

    async def defaults_for(self, provider: str, models: Collection[str]) -> dict[str, ModelPricing]:
        """Today's community rate for each model that has one, keyed by model name.

        Empty while the deployment does not consult the dataset: a rate seeded
        from it would be a rate the operator switched off.
        """
        if not default_pricing_enabled():
            return {}
        return await self.pricing.defaults_for(provider, models)

    async def offer(
        self, provider: str, names: Sequence[str], defaults: Mapping[str, ModelPricing]
    ) -> Sequence[HostedProviderModel]:
        """Record models as offered, each switched on exactly when something prices it.

        Raises:
            HostedModelAlreadyOfferedError: one of ``names`` is already offered here.
        """
        if not names:
            return []
        now = normalize_effective_at(None)
        keys = {name: _key(provider, name) for name in names}
        stored = await self.pricing.keys_with_a_price(keys.values())
        seeded = {
            keys[name]: default for name, default in defaults.items() if name in keys and keys[name] not in stored
        }
        await self.pricing.store_defaults(seeded, now)
        enabled = {name: keys[name] in stored or keys[name] in seeded for name in names}
        try:
            return await self.models.create_many(provider, names, enabled=enabled)
        except HostedModelConflict as conflict:
            raise HostedModelAlreadyOfferedError(provider, conflict.model) from conflict

    async def reseed(self, provider: str, defaults: Mapping[str, ModelPricing]) -> list[str]:
        """Move every seeded rate to today's default, and price what nothing prices any more.

        A version somebody chose is left alone. A model whose rates are gone,
        because they were deleted since the offer, is seeded again where the
        dataset prices it and switched off where it does not, so the offer rule
        holds after the fact as well as at the offer. Returns the models whose
        rate moved.
        """
        rows = await self.models.list_all_for_provider(provider)
        if not rows:
            return []
        now = normalize_effective_at(None)
        keys = {row.model: _key(provider, row.model) for row in rows}
        latest = await self.pricing.latest_versions(keys.values())
        moved: dict[str, ModelPricing] = {}
        repriced: list[str] = []
        for row in rows:
            version = latest.get(keys[row.model])
            default = defaults.get(row.model)
            if version is None:
                if default is None:
                    row.enabled = False
                    continue
                moved[keys[row.model]] = default
                row.enabled = True
                repriced.append(row.model)
                continue
            if version.origin != SEED_ORIGIN or default is None or self.pricing.rates_match(version, default):
                continue
            moved[keys[row.model]] = default
            repriced.append(row.model)
        # One flush for every row edited above: a provider can offer several hundred.
        await self.models.flush()
        await self.pricing.store_defaults(moved, now)
        return sorted(repriced)

    async def current_rates(self, provider: str, rows: Sequence[HostedProviderModel]) -> dict[str, HostedModelRates]:
        """What each offered model is currently served at, keyed by model name.

        Two rungs of the deployment's ladder, in its order: the stored version,
        then the community default where the deployment consults it. A stored
        version that is still a seeded default reads as a default. A model
        neither rung prices is absent.
        """
        if not rows:
            return {}
        now = normalize_effective_at(None)
        keys = {_key(provider, row.model): row.model for row in rows}
        versions = await self.pricing.current_versions(keys, now)
        rates: dict[str, HostedModelRates] = {}
        for key, version in versions.items():
            source = PRICE_SOURCE_DEFAULT if version.origin == SEED_ORIGIN else PRICE_SOURCE_DEPLOYMENT
            rates[keys[key]] = HostedModelRates.from_version(version, source)
        missing = [model for model in keys.values() if model not in rates]
        for model, default in (await self.defaults_for(provider, missing)).items():
            rates[model] = HostedModelRates.from_version(default, PRICE_SOURCE_DEFAULT)
        return rates

    async def write_rate(
        self, provider: str, model: str, request: HostedModelCreateRequest | HostedModelUpdateRequest
    ) -> None:
        """Record the deployment's rates for a model as a version an operator chose.

        Only the cache rates the caller set travel: an omitted one inherits and
        an explicit null clears. The request's validator guarantees the pair.
        """
        if request.input_price_per_million is None or request.output_price_per_million is None:
            msg = "a rate is the input/output pair"
            raise ValueError(msg)
        await self.pricing.write_rate(
            _key(provider, model),
            input_price_per_million=request.input_price_per_million,
            output_price_per_million=request.output_price_per_million,
            cache_rates={
                field: getattr(request, field) for field in _CACHE_RATE_FIELDS if field in request.model_fields_set
            },
        )
