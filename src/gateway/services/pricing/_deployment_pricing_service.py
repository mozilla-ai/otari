"""The deployment price list: the rates every organization is charged unless it holds its own.

``model_pricing`` is a version series keyed ``(model_key, effective_at)``. The
newest version at or before now is the rate a request settles at, after the
organization's own override and before the community default
(``services.pricing_service.find_model_pricing``). This service is how another
domain reads and writes that list without naming the table: the surface that
offers models on the deployment's hosted providers stores a rate here, seeds a
community default here, and asks here what a model is currently served at.

A write is a new version at the current instant: a cache rate the caller
omitted inherits the latest stored value and an explicit null clears it, and a
second write in the same instant lands on that instant's row rather than
colliding on the composite key.

Every method that reaches the database opens a block on the Unit of Work it was
built on, so a caller already inside a block has this work join its own.
"""

import asyncio
import uuid
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from decimal import Decimal

from gateway.core.metered_pricing import quantize_rate
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.money import to_usd, to_usd_or_none
from gateway.models.pricing import API_ORIGIN, SEED_ORIGIN, ModelPricing
from gateway.repositories.pricing import ModelPricingRepository
from gateway.services.pricing_service import default_model_pricing, normalize_effective_at

_CACHE_RATE_FIELDS = (
    "cache_read_price_per_million",
    "cache_write_price_per_million",
    "cache_write_1h_price_per_million",
)
_RATE_FIELDS = ("input_price_per_million", "output_price_per_million", *_CACHE_RATE_FIELDS)


def _resolve_defaults(provider: str, models: Sequence[str], as_of: datetime) -> dict[str, ModelPricing]:
    """Community default rates for several models. Synchronous; run off the loop."""
    resolved: dict[str, ModelPricing] = {}
    for model in models:
        default = default_model_pricing(provider, model, as_of)
        if default is not None:
            resolved[model] = default
    return resolved


def _stored_default(model_key: str, default: ModelPricing, effective_at: datetime) -> ModelPricing:
    """The version that stores a community default on a hosted model's behalf.

    ``SEED_ORIGIN`` is what lets a refresh move it with the dataset, where a
    version somebody chose is left alone.
    """
    return ModelPricing(
        model_key=model_key,
        effective_at=effective_at,
        input_price_per_million=default.input_price_per_million,
        output_price_per_million=default.output_price_per_million,
        cache_read_price_per_million=default.cache_read_price_per_million,
        cache_write_price_per_million=default.cache_write_price_per_million,
        cache_write_1h_price_per_million=default.cache_write_1h_price_per_million,
        pricing_tiers=list(default.pricing_tiers or []),
        unit=default.unit or "tokens",
        origin=SEED_ORIGIN,
    )


def _quantized(value: Decimal | None) -> Decimal | None:
    return None if value is None else quantize_rate(to_usd(value))


class DeploymentPricingService:
    """Read and write the deployment price list."""

    def __init__(self, uow: UnitOfWork, *, pricing: ModelPricingRepository) -> None:
        self.uow = uow
        self.pricing = pricing

    async def keys_with_a_price(self, model_keys: Collection[str]) -> set[str]:
        """The keys among ``model_keys`` the list holds a version for."""
        async with self.uow:
            return await self.pricing.keys_with_a_price(model_keys)

    async def current_versions(self, model_keys: Collection[str], as_of: datetime) -> dict[str, ModelPricing]:
        """The version in effect at ``as_of`` for each key that has one."""
        async with self.uow:
            return await self.pricing.latest_versions(model_keys, as_of=as_of)

    async def latest_versions(self, model_keys: Collection[str]) -> dict[str, ModelPricing]:
        """The newest version of each key, whatever its date."""
        async with self.uow:
            return await self.pricing.latest_versions(model_keys)

    async def defaults_for(self, provider: str, models: Collection[str]) -> dict[str, ModelPricing]:
        """Today's community rate for each model that has one, keyed by the bare model name.

        Transient rows, never added to a session. The dataset walk is
        synchronous and runs per model, so it runs off the loop, and a caller
        keeps it outside any transaction.
        """
        wanted = sorted(set(models))
        if not wanted:
            return {}
        return await asyncio.to_thread(_resolve_defaults, provider, wanted, normalize_effective_at(None))

    async def store_defaults(self, defaults: Mapping[str, ModelPricing], effective_at: datetime) -> None:
        """Store community defaults as the deployment's own versions, one per ``model_key``.

        ``defaults`` maps each ``provider:model`` key to the transient row
        :meth:`defaults_for` resolved. Flushed so a read in the same block sees
        them.
        """
        if not defaults:
            return
        async with self.uow:
            self.pricing.add_all([_stored_default(key, default, effective_at) for key, default in defaults.items()])
            await self.pricing.flush()

    @staticmethod
    def rates_match(stored: ModelPricing, default: ModelPricing) -> bool:
        """Whether a stored version already holds a default's rates.

        Compared at the rate column's own scale, because the stored value has
        been through it and the freshly resolved one has not, so an exact
        comparison would report a difference the database cannot hold and
        reprice every model on every refresh.
        """
        for field in _RATE_FIELDS:
            if _quantized(getattr(stored, field)) != _quantized(getattr(default, field)):
                return False
        return list(stored.pricing_tiers or []) == list(default.pricing_tiers or [])

    async def write_rate(
        self,
        model_key: str,
        *,
        input_price_per_million: float,
        output_price_per_million: float,
        cache_rates: Mapping[str, float | None],
    ) -> ModelPricing:
        """Record the deployment's rates for ``model_key`` as a new version at the current instant.

        ``cache_rates`` holds only the cache fields the caller set: a field
        present with ``None`` clears the rate, and a field absent inherits the
        latest stored version's value. Tiers and the unit carry forward from
        that version too, because the callers of this method collect neither.
        """
        unknown = set(cache_rates) - set(_CACHE_RATE_FIELDS)
        if unknown:
            msg = f"not a cache rate: {sorted(unknown)}"
            raise ValueError(msg)
        effective_at = normalize_effective_at(None)
        async with self.uow:
            latest = (await self.pricing.latest_versions([model_key])).get(model_key)

            def cache_rate(field: str) -> Decimal | None:
                if field in cache_rates:
                    return to_usd_or_none(cache_rates[field])
                inherited: Decimal | None = getattr(latest, field) if latest is not None else None
                return inherited

            version = latest if latest is not None and latest.effective_at == effective_at else None
            if version is None:
                version = ModelPricing(
                    model_key=model_key,
                    effective_at=effective_at,
                    pricing_tiers=list(latest.pricing_tiers or []) if latest is not None else [],
                    unit=latest.unit if latest is not None else "tokens",
                )
                self.pricing.add_all([version])
            version.input_price_per_million = to_usd(input_price_per_million)
            version.output_price_per_million = to_usd(output_price_per_million)
            version.cache_read_price_per_million = cache_rate("cache_read_price_per_million")
            version.cache_write_price_per_million = cache_rate("cache_write_price_per_million")
            version.cache_write_1h_price_per_million = cache_rate("cache_write_1h_price_per_million")
            version.origin = API_ORIGIN
            await self.pricing.flush()
            return version

    async def all_keys(self) -> list[tuple[str, int]]:
        """Every key on the list, with how many versions each holds."""
        async with self.uow:
            return await self.pricing.all_keys()

    async def doomed_overrides(self, canonical_keys: Sequence[str]) -> list[tuple[uuid.UUID, uuid.UUID, str]]:
        """Every organization override above ``canonical_keys``, as ``(id, organization_id, model_key)``.

        Listed rather than judged: which of them a sweep spares is a question
        about the organizations' own provider keys, which another domain holds.
        """
        if not canonical_keys:
            return []
        async with self.uow:
            return await self.pricing.overrides_for_keys(canonical_keys)

    async def delete_keys(self, model_keys: Sequence[str], override_ids: Sequence[uuid.UUID]) -> tuple[int, int]:
        """Take every version of ``model_keys`` off the list, with the overrides named by id."""
        if not model_keys and not override_ids:
            return 0, 0
        async with self.uow:
            return await self.pricing.delete_keys(model_keys, override_ids)


__all__ = ["DeploymentPricingService"]
