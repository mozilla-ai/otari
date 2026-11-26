"""The offer rule: which models a hosted provider serves, and the rate each is seeded at.

A rate is written to the deployment price list. The deployment settles the
upstream bill for a hosted model, so its rate is the deployment's and applies to
every organization that holds no override of its own. That list is also what
lists an undiscovered model, so offering a model and pricing it are one write
seen from two sides.

A model offered without a rate is seeded with the community default, as a
version carrying ``SEED_ORIGIN``, while the deployment consults that dataset at
all. An ``otari`` provider's models are seeded whether it does or not, with the
vendor's rate as an estimate. A refresh moves a seeded version with the dataset and leaves every other
origin alone, because those are rates somebody chose. A model nothing prices
is offered but not served, so a model the dataset has not caught up with
cannot be billed at nothing; a refresh serves it once the dataset prices it.
The exceptions are a model switched on without a rate on purpose, by a provider
that serves unlisted models, by a move that serves everything a provider lists,
or by an operator, which no refresh holds for one.

Every method that writes runs inside a block the caller opened. The dataset
walk runs off the loop and outside any block.
"""

from collections.abc import Collection, Mapping, Sequence

from any_llm.types.model import Model

from gateway.core.config import region_instance
from gateway.exceptions.providers_exceptions import HostedModelAlreadyOfferedError
from gateway.models.pricing import CONFIG_ORIGIN, LISTED_ORIGIN, SEED_ORIGIN, HostedPriceSource, ModelPricing
from gateway.models.providers import DISABLED_BY_OPERATOR, DISABLED_UNAVAILABLE, DISABLED_UNPRICED, HostedProviderModel
from gateway.repositories.providers import HostedModelConflict, HostedProviderModelRepository
from gateway.schemas.providers import HostedModelCreateRequest, HostedModelRates, HostedModelUpdateRequest
from gateway.services.pricing import DeploymentPricingService
from gateway.services.pricing_service import default_pricing_enabled, normalize_effective_at

# The any-llm implementation that reaches another Otari, such as otari.ai.
OTARI_IMPLEMENTATION = "otari"

PRICE_SOURCE_DEFAULT: HostedPriceSource = "defaults"
PRICE_SOURCE_PROVIDER: HostedPriceSource = "provider"

# The origins a refresh may move: nobody chose a rate that carries one.
_MOVABLE_ORIGINS = (SEED_ORIGIN, LISTED_ORIGIN)

_CACHE_RATE_FIELDS = (
    "cache_read_price_per_million",
    "cache_write_price_per_million",
)


_SOURCES: Mapping[str | None, HostedPriceSource] = {
    SEED_ORIGIN: PRICE_SOURCE_DEFAULT,
    LISTED_ORIGIN: PRICE_SOURCE_PROVIDER,
}


def _key(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def _hold_for_a_price(row: HostedProviderModel) -> None:
    """Switch a row off for want of a rate, unless it is already off for a reason a rate does not cure."""
    if row.disabled_reason not in (DISABLED_BY_OPERATOR, DISABLED_UNAVAILABLE):
        row.enabled = False
        row.disabled_reason = DISABLED_UNPRICED


def _release_price_hold(row: HostedProviderModel) -> None:
    """Switch back on a row that was off only for want of a rate."""
    if row.disabled_reason == DISABLED_UNPRICED:
        row.enabled = True
        row.disabled_reason = None


class HostedOffers:
    """Offer, price and reprice the models a hosted provider serves."""

    def __init__(self, models: HostedProviderModelRepository, pricing: DeploymentPricingService) -> None:
        self.models = models
        self.pricing = pricing
        # What each provider's own listing priced, as last dialed by this instance.
        self._listed: dict[str, dict[str, ModelPricing]] = {}

    def remember_listing(self, provider: str, listing: Sequence[Model]) -> None:
        """Keep the rates ``provider``'s own model listing named, to price its models at ahead of an estimate.

        Another Otari lists each model with the price it sells it at, in the
        ``pricing`` block this gateway's own listing publishes.
        """
        blocks = {
            model.id: block
            for model in listing
            if isinstance(block := (model.model_extra or {}).get("pricing"), Mapping)
        }
        self._listed[provider] = self.pricing.listed_rates(provider, blocks)

    async def defaults_for(
        self, provider: str, models: Collection[str], *, implementation: str | None = None
    ) -> dict[str, ModelPricing]:
        """Today's community rate for each model that has one, keyed by model name.

        ``provider`` is the hosted provider's name, which the rates are keyed
        under; ``implementation`` is what they are looked up as where the name
        differs. Empty while the deployment does not consult the dataset: a
        rate seeded from it would be a rate the operator switched off. An
        ``otari`` provider is the exception, and is seeded either way: nothing
        else prices its ``vendor:model`` ids, so with the dataset off every
        model of a provider just connected would arrive switched off. A rate
        the provider's own listing named (``remember_listing``) answers ahead
        of the dataset, and whatever the deployment consults.
        """
        listed = {model: rate for model, rate in self._listed.get(provider, {}).items() if model in models}
        if not default_pricing_enabled() and implementation != OTARI_IMPLEMENTATION:
            return listed
        estimated = [model for model in models if model not in listed]
        return {**await self.pricing.defaults_for(provider, estimated, implementation=implementation), **listed}

    async def offer(
        self,
        provider: str,
        names: Sequence[str],
        defaults: Mapping[str, ModelPricing],
        *,
        serves_unlisted: bool = False,
    ) -> Sequence[HostedProviderModel]:
        """Record models as offered, each switched on exactly when something prices it.

        ``serves_unlisted`` serves them unpriced, so each is switched on, and one
        nothing prices is marked as served without a rate.

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
        unpriced = [name for name in names if keys[name] not in stored and keys[name] not in seeded]
        try:
            if serves_unlisted:
                return await self.models.create_many(provider, names, unpriced_served=unpriced)
            return await self.models.create_many(provider, names, disabled=dict.fromkeys(unpriced, DISABLED_UNPRICED))
        except HostedModelConflict as conflict:
            raise HostedModelAlreadyOfferedError(provider, conflict.model) from conflict

    def serve_once_priced(self, row: HostedProviderModel) -> None:
        """Switch back on a row that was off only for want of a rate, as a rate just given for it allows."""
        _release_price_hold(row)

    async def switch_by_operator(self, provider: str, row: HostedProviderModel, enabled: bool) -> None:
        """Set the switch as an operator chose it: off stays off until an operator switches it on.

        A row switched on with no stored rate is marked as served without one,
        so a refresh does not hold it. One switched on while priced is not, so
        deleting its rate later still holds it rather than serving it for free.
        """
        row.enabled = enabled
        row.disabled_reason = None if enabled else DISABLED_BY_OPERATOR
        row.serves_unpriced = enabled and not await self.pricing.keys_with_a_price([_key(provider, row.model)])

    async def switch_many_priced_by_operator(self, rows: Sequence[HostedProviderModel], enabled: bool) -> None:
        """Set several rows' switches as an operator chose them, in one flush.

        The caller switches on only rows that have a rate, so none is marked as
        served without one: a refresh holds it again if its rate goes, which is
        what an operator who served "every priced model" asked for.
        """
        if not rows:
            return
        for row in rows:
            row.enabled = enabled
            row.disabled_reason = None if enabled else DISABLED_BY_OPERATOR
            row.serves_unpriced = False
        await self.models.flush()

    async def stored_priced(self, provider: str, models: Collection[str]) -> set[str]:
        """The models among ``models`` the deployment price list holds a version for, read now."""
        keys = {_key(provider, model): model for model in models}
        return {keys[key] for key in await self.pricing.keys_with_a_price(keys)}

    async def fallback_priced(
        self, provider: str, models: Collection[str], *, implementation: str | None = None
    ) -> set[str]:
        """The models among ``models`` a request would be priced for by the default rung, with no stored version.

        Empty while the deployment does not consult the dataset, as in :meth:`current_rates`.
        A dataset walk, so call it outside any block.
        """
        if not default_pricing_enabled():
            return set()
        return set(await self.defaults_for(provider, models, implementation=implementation))

    async def custom_against_defaults(
        self,
        provider: str,
        models: Collection[str],
        *,
        implementation: str | None = None,
        listed_only: bool = False,
    ) -> tuple[dict[str, tuple[ModelPricing, ModelPricing, bool]], list[str]]:
        """Each model priced by someone here, against the rate the default rung would give it.

        Returns, keyed by model, the version in effect, the default and whether
        the two charge the same, for each such model the default rung prices;
        and, sorted, the ones it does not. Left out: a rate the configuration
        stored, since the next start writes it again; one a refresh already
        moves; and a tiered rate, which a default would flatten. ``listed_only``
        takes the default from the provider's remembered listing alone, never
        from the dataset.
        """
        keys = {_key(provider, model): model for model in models}
        versions = await self.pricing.current_versions(keys, normalize_effective_at(None))
        chosen = {
            keys[key]: version
            for key, version in versions.items()
            if version.origin not in (*_MOVABLE_ORIGINS, CONFIG_ORIGIN) and not version.pricing_tiers
        }
        if not chosen:
            return {}, []
        if listed_only:
            listed = self._listed.get(provider, {})
            defaults = {model: listed[model] for model in chosen if model in listed}
        else:
            defaults = await self.defaults_for(provider, list(chosen), implementation=implementation)
        paired = {
            model: (version, defaults[model], self.pricing.rates_match(version, defaults[model]))
            for model, version in chosen.items()
            if model in defaults
        }
        return paired, sorted(model for model in chosen if model not in defaults)

    async def switch_off_by_operator(self, provider: str, models: Collection[str]) -> int:
        """Switch these models off under ``provider`` as an operator's choice. Returns how many were on."""
        return await self.models.switch_off(provider, models, reason=DISABLED_BY_OPERATOR)

    async def switch_off_unavailable(self, provider: str, model: str) -> bool:
        """Switch ``model`` off under ``provider`` because the provider refused to serve it. Whether it was on."""
        return await self.models.switch_off(provider, [model], reason=DISABLED_UNAVAILABLE) > 0

    async def reseed(
        self, provider: str, defaults: Mapping[str, ModelPricing], *, serves_unlisted: bool = False
    ) -> list[str]:
        """Move every seeded rate to today's default, price what nothing prices, and serve what is now priced.

        A version somebody chose is left alone. A model with no rate, because
        nothing priced it at the offer or its rates were deleted since, is
        seeded where the dataset prices it and held off where it does not, so
        the offer rule holds after the fact as well as at the offer. A row held
        off for want of a rate comes back on once it has one; a row an operator
        switched off stays off. A row marked as served without a rate is never
        held, and a provider that ``serves_unlisted`` models holds none and
        releases what an earlier refresh held. Returns the models whose rate
        moved.
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
                    if serves_unlisted:
                        _release_price_hold(row)
                    elif not row.serves_unpriced:
                        _hold_for_a_price(row)
                    continue
                moved[keys[row.model]] = default
                repriced.append(row.model)
                _release_price_hold(row)
                continue
            _release_price_hold(row)
            if version.origin not in _MOVABLE_ORIGINS or default is None:
                continue
            if self.pricing.rates_match(version, default) and version.origin == (default.origin or SEED_ORIGIN):
                continue
            moved[keys[row.model]] = default
            repriced.append(row.model)
        # One flush for every row edited above: a provider can offer several hundred.
        await self.models.flush()
        await self.pricing.store_defaults(moved, now)
        return sorted(repriced)

    async def current_rates(
        self, provider: str, rows: Sequence[HostedProviderModel], *, implementation: str | None = None
    ) -> dict[str, HostedModelRates]:
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
            rates[keys[key]] = HostedModelRates.from_version(version, _SOURCES.get(version.origin, "deployment"))
        if not default_pricing_enabled():
            # The dataset is what a request falls back to only while it is consulted, an
            # ``otari`` provider's included: its seeded rates are stored, and read above.
            return rates
        missing = [model for model in keys.values() if model not in rates]
        for model, default in (await self.defaults_for(provider, missing, implementation=implementation)).items():
            rates[model] = HostedModelRates.from_version(default, PRICE_SOURCE_DEFAULT)
        return rates

    async def region_rates(
        self, provider: str, regions: Sequence[str], rows: Sequence[HostedProviderModel]
    ) -> dict[str, dict[str, HostedModelRates]]:
        """The rates each region sets of its own, keyed by model name and then by region slug.

        Only a version stored under the region's own key counts, canonical
        (``tokenhub@sg:model``) ahead of the legacy ``tokenhub@sg/model``, as
        settlement reads them: a region with neither settles at the provider's
        rate, so it is absent.
        """
        if not rows or not regions:
            return {}
        now = normalize_effective_at(None)
        keys: dict[str, tuple[str, str, int]] = {}
        for slug in regions:
            instance = region_instance(provider, slug)
            for row in rows:
                keys[_key(instance, row.model)] = (row.model, slug, 0)
                keys[f"{instance}/{row.model}"] = (row.model, slug, 1)
        versions = await self.pricing.current_versions(keys, now)
        chosen: dict[tuple[str, str], tuple[int, ModelPricing]] = {}
        for key, version in versions.items():
            model, slug, rank = keys[key]
            if (model, slug) not in chosen or rank < chosen[(model, slug)][0]:
                chosen[(model, slug)] = (rank, version)
        rates: dict[str, dict[str, HostedModelRates]] = {}
        for (model, slug), (_, version) in chosen.items():
            rates.setdefault(model, {})[slug] = HostedModelRates.from_version(
                version, _SOURCES.get(version.origin, "deployment")
            )
        return rates

    async def reset_to_defaults(self, provider: str, defaults: Mapping[str, ModelPricing]) -> None:
        """Put each model back on its default rate, as a new version a refresh moves from now on.

        The version an operator chose stays in the history, so a request already
        billed keeps the cost it was billed at.
        """
        await self.pricing.store_defaults(
            {_key(provider, model): default for model, default in defaults.items()}, normalize_effective_at(None)
        )

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
            unit=request.unit,
        )
