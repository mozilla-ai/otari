"""The models an organization offers on its own provider keys.

An organization's BYO provider key reaches every model its provider serves and
lists none of them: catalog discovery dials ``config.providers`` instances only,
so before this a key bought an admin a credential and no way to see, price or
withhold what it served. This service is the other half: it asks the provider
what the stored credential reaches, records each model as offered, seeds a rate
from the community dataset so the model can actually be served, and carries the
switch that decides whether it is.

Three rules are worth stating once, because each is load-bearing and none is
obvious from a signature.

**A rate is written to ``organization_model_pricing``, never to
``model_pricing``.** The deployment price list carries no tenancy column, so one
row there prices the model for every organization on the deployment; a rate this
surface seeds belongs to the one organization whose credential pays for it. The
organization table is also the rung
``services.pricing_service.find_model_pricing`` already consults first for a
request whose organization is known, so a rate seeded here is the rate those
requests settle at, with no second store to drift.

**``origin`` is what tells a seeded rate from a chosen one.** A row this service
wrote carries ``"seed"`` and a later refresh may move it to the day's default; a
row an admin edited through ``/organizations/me/pricing`` carries ``"api"`` and
a refresh leaves it alone forever. That is the whole provenance mechanism, and
it is why no timestamp column mirrors it.

**Disabled until priced.** A model no rate could be found for is recorded and
not served, so a model the pricing data has not caught up with cannot be billed
at nothing. The switch is also how a model is withdrawn: rows are disabled
rather than deleted, so the model, its rate and its history stay where an admin
can turn it back on.

No transaction is open across an upstream dial or across the thread hop that
resolves community defaults, both of which can take the whole discovery timeout.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from gateway.core.metered_pricing import quantize_rate
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.providers_exceptions import (
    OrgProviderLastModelError,
    OrgProviderModelAlreadyOfferedError,
    OrgProviderModelNameRequiredError,
    OrgProviderModelNotFoundError,
    OrgProviderModelUnpricedError,
)
from gateway.models.money import as_float, to_usd, to_usd_or_none
from gateway.models.pricing import SEED_ORIGIN, ModelPricing, OrganizationModelPricing, PriceSource
from gateway.models.provider_keys import (
    OrgProviderKey,
    OrgProviderKeyModel,
)
from gateway.models.tenancy import User
from gateway.repositories.pricing import OrganizationModelPricingRepository
from gateway.repositories.providers import OfferedModelConflict, OrgProviderKeyModelRepository
from gateway.repositories.tenancy import OrgProviderKeyRepository
from gateway.schemas.providers import (
    OrgProviderAvailableModelsPublic,
    OrgProviderKeyModelPublic,
    OrgProviderKeyModelsPublic,
    OrgProviderModelsRefreshPublic,
)
from gateway.services.model_discovery_service import ProviderDiscovery, test_provider_credentials
from gateway.services.organization_pricing_service import OrganizationPricingService
from gateway.services.pricing_service import default_model_pricing, normalize_effective_at
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError, decrypt_secret
from gateway.services.tenancy.errors import OrgProviderKeyNotFoundError
from gateway.services.tenancy.organization_service import OrganizationService

# What a client is told about where a rate came from. The spellings are
# ``models.pricing.PriceSource``, so the models panel and the Models page name
# the same rungs to the same reader.
PRICE_SOURCE_ORGANIZATION: PriceSource = "organization"
PRICE_SOURCE_DEFAULT: PriceSource = "defaults"
PRICE_SOURCE_DEPLOYMENT: PriceSource = "deployment"

# The message an operator needs when a stored credential will not decrypt. The
# row is intact; the configured OTARI_SECRET_KEY just cannot read it any more.
_UNDECRYPTABLE = (
    "The stored key cannot be decrypted. Was OTARI_SECRET_KEY rotated? Re-enter the credential, then try again."
)

_RATE_FIELDS = (
    "input_price_per_million",
    "output_price_per_million",
    "cache_read_price_per_million",
    "cache_write_price_per_million",
    "cache_write_1h_price_per_million",
)


@dataclass(frozen=True)
class _Price:
    """What one offered model currently costs, and which rung said so."""

    source: str
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    cache_read_price_per_million: float | None = None
    cache_write_price_per_million: float | None = None
    cache_write_1h_price_per_million: float | None = None
    pricing_id: uuid.UUID | None = None


def _quantized(value: Decimal | None) -> Decimal | None:
    return None if value is None else quantize_rate(value)


def _same_rates(stored: OrganizationModelPricing, default: ModelPricing) -> bool:
    """Whether a stored rate still says what today's community default says.

    Compared at the rate column's own scale, because the stored value has been
    through it and the freshly resolved one has not, so an exact comparison
    would report a difference the database cannot hold and reprice every model
    on every refresh.
    """
    for field in _RATE_FIELDS:
        if _quantized(getattr(stored, field)) != _quantized(getattr(default, field)):
            return False
    return list(stored.pricing_tiers or []) == list(default.pricing_tiers or [])


def _price_from_organization_row(row: OrganizationModelPricing) -> _Price:
    """Report a stored organization rate, saying whether it is still the seeded one."""
    return _Price(
        source=PRICE_SOURCE_DEFAULT if row.origin == SEED_ORIGIN else PRICE_SOURCE_ORGANIZATION,
        input_price_per_million=float(row.input_price_per_million),
        output_price_per_million=float(row.output_price_per_million),
        cache_read_price_per_million=as_float(row.cache_read_price_per_million),
        cache_write_price_per_million=as_float(row.cache_write_price_per_million),
        cache_write_1h_price_per_million=as_float(row.cache_write_1h_price_per_million),
        pricing_id=row.id,
    )


def _price_from_deployment_row(row: ModelPricing) -> _Price:
    return _Price(
        source=PRICE_SOURCE_DEPLOYMENT,
        input_price_per_million=float(row.input_price_per_million),
        output_price_per_million=float(row.output_price_per_million),
        cache_read_price_per_million=as_float(row.cache_read_price_per_million),
        cache_write_price_per_million=as_float(row.cache_write_price_per_million),
        cache_write_1h_price_per_million=as_float(row.cache_write_1h_price_per_million),
    )


def _seeded_row(
    organization_id: uuid.UUID, model_key: str, default: ModelPricing, effective_from: datetime
) -> OrganizationModelPricing:
    """An organization rate row carrying a community default, open ended."""
    return OrganizationModelPricing(
        organization_id=organization_id,
        model_key=model_key,
        input_price_per_million=to_usd(float(default.input_price_per_million)),
        output_price_per_million=to_usd(float(default.output_price_per_million)),
        cache_read_price_per_million=to_usd_or_none(as_float(default.cache_read_price_per_million)),
        cache_write_price_per_million=to_usd_or_none(as_float(default.cache_write_price_per_million)),
        cache_write_1h_price_per_million=to_usd_or_none(as_float(default.cache_write_1h_price_per_million)),
        pricing_tiers=list(default.pricing_tiers or []),
        unit=default.unit or "tokens",
        origin=SEED_ORIGIN,
        effective_from=effective_from,
        effective_to=None,
    )


def _apply_default(row: OrganizationModelPricing, default: ModelPricing) -> None:
    """Move a seeded row onto today's community default, in place.

    In place rather than by appending a period: two open-ended periods for one
    model overlap, which the organization pricing surface refuses outright. A
    settled cost lives on its usage row, so nothing already billed moves.
    """
    row.input_price_per_million = to_usd(float(default.input_price_per_million))
    row.output_price_per_million = to_usd(float(default.output_price_per_million))
    row.cache_read_price_per_million = to_usd_or_none(as_float(default.cache_read_price_per_million))
    row.cache_write_price_per_million = to_usd_or_none(as_float(default.cache_write_price_per_million))
    row.cache_write_1h_price_per_million = to_usd_or_none(as_float(default.cache_write_1h_price_per_million))
    row.pricing_tiers = list(default.pricing_tiers or [])


def _resolve_defaults(provider: str, models: Sequence[str], as_of: datetime) -> dict[str, ModelPricing]:
    """Community default rates for several models. Synchronous; run off the loop."""
    resolved: dict[str, ModelPricing] = {}
    for model in models:
        default = default_model_pricing(provider, model, as_of)
        if default is not None:
            resolved[model] = default
    return resolved


def _model_public(row: OrgProviderKeyModel, price: _Price | None) -> OrgProviderKeyModelPublic:
    return OrgProviderKeyModelPublic(
        id=row.id,
        org_provider_key_id=row.org_provider_key_id,
        model=row.model,
        input_price_per_million=price.input_price_per_million if price else None,
        output_price_per_million=price.output_price_per_million if price else None,
        cache_read_price_per_million=price.cache_read_price_per_million if price else None,
        cache_write_price_per_million=price.cache_write_price_per_million if price else None,
        cache_write_1h_price_per_million=price.cache_write_1h_price_per_million if price else None,
        price_source=price.source if price else None,
        pricing_id=price.pricing_id if price else None,
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class OrgProviderModelService:
    """Offer, price and withhold the models an organization's provider keys reach."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        organizations: OrganizationService,
        org_pricing: OrganizationPricingService,
        models: OrgProviderKeyModelRepository,
        pricing: OrganizationModelPricingRepository,
        keys: OrgProviderKeyRepository,
        refresh_overlay: Callable[[], Awaitable[None]],
    ) -> None:
        """Bind the unit of work and the collaborators this service composes.

        ``refresh_overlay`` reloads the dispatch-path credential and allow-list
        cache. Injected as a callable because that function takes the session,
        which a service may not name.
        """
        self.uow = uow
        self.organizations = organizations
        self.org_pricing = org_pricing
        self.models = models
        self.pricing = pricing
        self.keys = keys
        self.refresh_overlay = refresh_overlay

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    async def _key_for_user(self, user: User, key_id: uuid.UUID) -> OrgProviderKey:
        """The caller's organization's key, having checked they may administer it.

        Owners and admins on reads as well as writes, the same one-audience rule
        ``OrgProviderKeyService.list_keys_for_user`` follows: a row names the
        provider and the models the organization buys through it.

        Raises:
            OrgProviderKeyNotFoundError: no such key in the caller's organization.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(
            user=user, organization=organization
        )
        key = await self.keys.get_in_organization(key_id, organization.id)
        if key is None:
            # Scoped rather than checked afterward, so another organization's id
            # is indistinguishable from one that does not exist.
            raise OrgProviderKeyNotFoundError(key_id)
        return key

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def list_models(
        self, *, user: User, key_id: uuid.UUID, skip: int = 0, limit: int = 500
    ) -> OrgProviderKeyModelsPublic:
        """One page of a key's offered models, each priced as it currently serves."""
        async with self.uow:
            key = await self._key_for_user(user, key_id)
            rows, count = await self.models.list_for_key(key_id, skip=skip, limit=limit)
            prices = await self._current_prices(key, rows)
        return OrgProviderKeyModelsPublic(
            data=[_model_public(row, prices.get(row.model)) for row in rows], count=count
        )

    async def available_models(
        self, *, user: User, key_id: uuid.UUID, timeout: float
    ) -> OrgProviderAvailableModelsPublic:
        """What the provider says it serves on this key's stored credential.

        The credential never leaves the process: it goes into an any-llm client
        here, exactly as the runtime path uses it, and only model names come
        back. Failure is reported in the body rather than raised, because "the
        upstream would not say" is an answer the form has to render.
        """
        async with self.uow:
            key = await self._key_for_user(user, key_id)
            provider = key.provider
            credential = self._credential(key)
        if credential is None:
            return OrgProviderAvailableModelsPublic(provider=provider, models=[], error=_UNDECRYPTABLE)
        discovery = await self._dial(key, credential, timeout=timeout)
        return OrgProviderAvailableModelsPublic(
            provider=provider,
            models=sorted(model.id for model in discovery.models),
            error=discovery.error,
            discovery_unsupported=discovery.discovery_unsupported,
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def add_model(self, *, user: User, key_id: uuid.UUID, model: str) -> OrgProviderKeyModelPublic:
        """Offer one model by name, for a backend whose models cannot be listed.

        Takes no rate: an organization's rates are written through
        ``/organizations/me/pricing``, so a price set here and a price set there
        could not disagree. The offer seeds the community default like any other,
        and a model nothing prices lands disabled.

        Raises:
            OrgProviderKeyNotFoundError: no such key in the caller's organization.
            OrgProviderModelNameRequiredError: the name is blank once trimmed.
            OrgProviderModelAlreadyOfferedError: the model is already offered here.
        """
        name = model.strip()
        if not name:
            raise OrgProviderModelNameRequiredError

        async with self.uow:
            key = await self._key_for_user(user, key_id)
            if await self.models.get_by_model(key_id, name) is not None:
                raise OrgProviderModelAlreadyOfferedError(key.provider, name)
            provider = key.provider
            organization_id = key.organization_id

        defaults = await self._defaults_for(provider, [name])

        async with self.uow:
            key = await self._key_for_user(user, key_id)
            if await self.models.get_by_model(key_id, name) is not None:
                raise OrgProviderModelAlreadyOfferedError(provider, name)
            try:
                [row] = await self._offer(
                    key_id=key_id,
                    organization_id=organization_id,
                    provider=provider,
                    models=[name],
                    defaults=defaults,
                    user=user,
                )
            except OfferedModelConflict as conflict:
                # The pre-check above races the insert, and the unique index is
                # what actually decides, the same way every other create in this
                # slice resolves it. Without this the loser of that race leaves
                # with a 500 rather than the 409 it is.
                raise OrgProviderModelAlreadyOfferedError(provider, conflict.model) from conflict
            prices = await self._current_prices(key, [row])
            public = _model_public(row, prices.get(row.model))
        await self.refresh_overlay()
        return public

    async def set_model_enabled(
        self, *, user: User, key_id: uuid.UUID, model_id: uuid.UUID, enabled: bool
    ) -> OrgProviderKeyModelPublic:
        """Turn one offered model's serving switch on or off.

        Switching one on is refused where nothing prices it. The offer path
        already records such a model unserved so it cannot be billed at nothing,
        and without the same check here the switch would be a way straight past
        that rule: the model would reach the catalog, and the dispatch gate, with
        no rate behind it. Switching one *off* is never refused, so a row that
        reached that state some other way can still be withdrawn.

        Raises:
            OrgProviderKeyNotFoundError: no such key in the caller's organization.
            OrgProviderModelNotFoundError: no such model on that key.
            OrgProviderModelUnpricedError: serving was asked for an unpriced model.
        """
        async with self.uow:
            key = await self._key_for_user(user, key_id)
            row = await self.models.get_in_key(model_id, key_id)
            if row is None:
                raise OrgProviderModelNotFoundError(model_id)
            if enabled and not row.enabled:
                priced = await self._current_prices(key, [row])
                if row.model not in priced:
                    raise OrgProviderModelUnpricedError(row.model)
            row.enabled = enabled
            await self.models.save(row)
            prices = await self._current_prices(key, [row])
            public = _model_public(row, prices.get(row.model))
        await self.refresh_overlay()
        return public

    async def remove_model(self, *, user: User, key_id: uuid.UUID, model_id: uuid.UUID) -> None:
        """Stop offering one model. Its rate and its history stay.

        Deliberately: usage has already settled against those rows, and a model
        offered again should find its rate where it was left rather than
        reverting to a default unannounced.

        The last one is refused, because removing it would *widen* the key.
        Absent rows mean unnarrowed, which is what a key nobody has refreshed
        looks like, so emptying the list returns the key to reaching everything
        its provider serves. That is the opposite of what pressing "stop
        offering" reads as, and it would happen quietly. The switch is how a
        model stops being served.

        Raises:
            OrgProviderKeyNotFoundError: no such key in the caller's organization.
            OrgProviderModelNotFoundError: no such model on that key.
            OrgProviderLastModelError: it is the only model the key offers.
        """
        async with self.uow:
            await self._key_for_user(user, key_id)
            row = await self.models.get_in_key(model_id, key_id)
            if row is None:
                raise OrgProviderModelNotFoundError(model_id)
            if len(await self.models.names_for_key(key_id)) == 1:
                raise OrgProviderLastModelError
            await self.models.delete_row(row)
        await self.refresh_overlay()

    async def offer_discovered_models(
        self, *, user: User, key_id: uuid.UUID, timeout: float
    ) -> OrgProviderModelsRefreshPublic:
        """Offer everything the provider lists on a key, for a key that offers nothing yet.

        What a freshly created key runs, so it starts with its real catalog
        rather than an empty list an admin retypes by hand. A provider that will
        not say (no listing endpoint, unreachable, credential refused) yields no
        models and a body saying why, never a failed create: the credential may
        still be right for dispatch.
        """
        return await self.refresh_models(user=user, key_id=key_id, timeout=timeout)

    async def refresh_models(
        self, *, user: User, key_id: uuid.UUID, timeout: float
    ) -> OrgProviderModelsRefreshPublic:
        """Ask the provider again, offer whatever is newly listed, and move seeded rates.

        Additive only: a model the provider no longer lists stays offered,
        because delisting one is a decision the serving switch owns and an
        upstream hiccup must not empty a catalog. New models follow the offer
        rule, so one nothing prices arrives disabled.

        Raises:
            OrgProviderKeyNotFoundError: no such key in the caller's organization.
        """
        async with self.uow:
            key = await self._key_for_user(user, key_id)
            offered = await self.models.names_for_key(key_id)
            credential = self._credential(key)

        if credential is None:
            return OrgProviderModelsRefreshPublic(added=[], repriced=[], count=len(offered), error=_UNDECRYPTABLE)

        discovery = await self._dial(key, credential, timeout=timeout)
        if discovery.error is not None or discovery.discovery_unsupported:
            return OrgProviderModelsRefreshPublic(
                added=[],
                repriced=[],
                count=len(offered),
                error=discovery.error,
                discovery_unsupported=discovery.discovery_unsupported,
            )

        added = sorted({model.id for model in discovery.models} - offered)
        repriced = await self._reprice(user=user, key_id=key_id, added=added)
        return OrgProviderModelsRefreshPublic(
            added=added, repriced=repriced, count=len(offered) + len(added)
        )

    async def refresh_pricing(self, *, user: User, key_id: uuid.UUID) -> OrgProviderModelsRefreshPublic:
        """Move every seeded rate onto today's community default, with no dial.

        The other half of Refresh models, separated because re-reading community
        rates is cheap and asking a provider for its catalog is not, and an admin
        who wants yesterday's price move does not want to wait on an upstream.

        Raises:
            OrgProviderKeyNotFoundError: no such key in the caller's organization.
        """
        async with self.uow:
            await self._key_for_user(user, key_id)
            count = len(await self.models.names_for_key(key_id))
        repriced = await self._reprice(user=user, key_id=key_id, added=[])
        return OrgProviderModelsRefreshPublic(added=[], repriced=repriced, count=count)

    # ------------------------------------------------------------------
    # The two rules
    # ------------------------------------------------------------------

    async def _reprice(self, *, user: User, key_id: uuid.UUID, added: Sequence[str]) -> list[str]:
        """Reseed every still-seeded rate, then offer ``added``. Returns what moved.

        The community lookup is synchronous and walks its dataset per model, so
        it runs off the loop and outside any transaction, which is why this is
        three phases rather than one block.
        """
        async with self.uow:
            key = await self._key_for_user(user, key_id)
            provider = key.provider
            organization_id = key.organization_id
            offered_rows = list(await self.models.list_all_for_key(key_id))
            now = normalize_effective_at(None)
            keys_by_model = {row.model: f"{provider}:{row.model}" for row in offered_rows}
            stored = await self.pricing.applicable_rows(organization_id, keys_by_model.values(), now)
            deployment = await self.pricing.deployment_rows(keys_by_model.values(), now)

        # Every offered model whose rate this surface still owns, plus every
        # offered model nothing prices at all: a model that arrived disabled
        # because the dataset had not caught up is exactly what this pass is for.
        candidates: list[str] = []
        for row in offered_rows:
            existing = stored.get(keys_by_model[row.model])
            if existing is not None:
                if existing.origin == SEED_ORIGIN:
                    candidates.append(row.model)
            elif keys_by_model[row.model] not in deployment:
                candidates.append(row.model)
        defaults = await self._defaults_for(provider, [*candidates, *added])

        async with self.uow:
            # Re-authorized and re-read rather than carried across: the block
            # above ended, so nothing read there is still under a transaction,
            # and the rates are exactly what a concurrent edit moves.
            await self._key_for_user(user, key_id)
            now = normalize_effective_at(None)
            offered_rows = list(await self.models.list_all_for_key(key_id))
            keys_by_model = {row.model: f"{provider}:{row.model}" for row in offered_rows}
            stored = await self.pricing.applicable_rows(organization_id, keys_by_model.values(), now)
            deployment = await self.pricing.deployment_rows(keys_by_model.values(), now)
            allowed = await self._seedable(user, organization_id, provider, candidates)

            repriced: list[str] = []
            fresh: list[OrganizationModelPricing] = []
            for row in offered_rows:
                model_key = keys_by_model[row.model]
                default = defaults.get(row.model)
                existing = stored.get(model_key)
                if existing is not None:
                    if existing.origin != SEED_ORIGIN or default is None or _same_rates(existing, default):
                        # An admin owns this rate now, the dataset dropped the
                        # model, or nothing moved. Leaving the stored rate is the
                        # answer in all three.
                        continue
                    _apply_default(existing, default)
                    repriced.append(row.model)
                    continue
                if default is None or model_key in deployment or row.model not in allowed:
                    continue
                fresh.append(_seeded_row(organization_id, model_key, default, now))
                if not row.enabled:
                    # It arrived disabled because nothing priced it. Something
                    # does now.
                    row.enabled = True
                    await self.models.save(row)
                repriced.append(row.model)
            self.pricing.add_all(fresh)
            await self.pricing.flush()

            if added:
                await self._offer(
                    key_id=key_id,
                    organization_id=organization_id,
                    provider=provider,
                    models=added,
                    defaults=defaults,
                    user=user,
                )
        await self.refresh_overlay()
        return sorted(repriced)

    async def _offer(
        self,
        *,
        key_id: uuid.UUID,
        organization_id: uuid.UUID,
        provider: str,
        models: Sequence[str],
        defaults: dict[str, ModelPricing],
        user: User,
    ) -> Sequence[OrgProviderKeyModel]:
        """Record models as offered, seeding a rate where one is missing and permitted.

        Runs inside a write block; ``defaults`` is resolved by the caller so the
        thread hop that produces it stays outside one.
        """
        if not models:
            return []
        now = normalize_effective_at(None)
        keys_by_model = {model: f"{provider}:{model}" for model in models}
        stored = await self.pricing.applicable_rows(organization_id, keys_by_model.values(), now)
        deployment = await self.pricing.deployment_rows(keys_by_model.values(), now)
        allowed = await self._seedable(user, organization_id, provider, models)

        seeded: dict[str, OrganizationModelPricing] = {}
        for model, model_key in keys_by_model.items():
            if model_key in stored or model_key in deployment or model not in allowed:
                continue
            default = defaults.get(model)
            if default is None:
                continue
            seeded[model] = _seeded_row(organization_id, model_key, default, now)
        self.pricing.add_all(list(seeded.values()))

        rows = [
            OrgProviderKeyModel(
                organization_id=organization_id,
                org_provider_key_id=key_id,
                model=model,
                # Offered and not served when nothing prices it, so a model the
                # community data has not caught up with cannot be billed at
                # nothing.
                enabled=keys_by_model[model] in stored or keys_by_model[model] in deployment or model in seeded,
            )
            for model in models
        ]
        return await self.models.create_many(rows)

    async def _seedable(
        self, user: User, organization_id: uuid.UUID, provider: str, models: Sequence[str]
    ) -> set[str]:
        """Which of ``models`` this organization may hold its own rate for.

        A model the *deployment* supplies the credential for is priced by the
        deployment's list, because the deployment settles that upstream bill;
        the organization pricing surface refuses an override for one, and
        seeding past that refusal would store a rate nobody could have created.
        Checked here rather than raised, because such a model is still legitimately
        offered: it simply prices from the rung below.

        One question for the whole pass where the answer is the same for every
        model (the organization's own key covers this provider), and per model
        only where it is not.
        """
        if not models:
            return set()
        byo = await self.org_pricing.provider_keys.get_byo_providers(organization_id=organization_id)
        if provider in byo:
            return set(models)
        allowed: set[str] = set()
        for model in models:
            if not await self.org_pricing.is_deployment_supplied(organization_id, f"{provider}:{model}"):
                allowed.add(model)
        return allowed

    # ------------------------------------------------------------------
    # Pricing and dialing
    # ------------------------------------------------------------------

    async def _current_prices(
        self, key: OrgProviderKey, rows: Sequence[OrgProviderKeyModel]
    ) -> dict[str, _Price]:
        """What each offered model currently costs this organization, keyed by model.

        The same ladder ``pricing_service.find_model_pricing`` walks, in its
        order: the organization's own row, then the deployment price list, then
        the community dataset. A stored row this surface seeded reports as a
        default, because that is what it is.
        """
        if not rows:
            return {}
        now = normalize_effective_at(None)
        keys_by_model = {row.model: f"{key.provider}:{row.model}" for row in rows}
        stored = await self.pricing.applicable_rows(key.organization_id, keys_by_model.values(), now)
        deployment = await self.pricing.deployment_rows(keys_by_model.values(), now)

        prices: dict[str, _Price] = {}
        unpriced: list[str] = []
        for model, model_key in keys_by_model.items():
            if (organization_row := stored.get(model_key)) is not None:
                prices[model] = _price_from_organization_row(organization_row)
            elif (deployment_row := deployment.get(model_key)) is not None:
                prices[model] = _price_from_deployment_row(deployment_row)
            else:
                unpriced.append(model)

        for model, default in (await self._defaults_for(key.provider, unpriced)).items():
            prices[model] = _Price(
                source=PRICE_SOURCE_DEFAULT,
                input_price_per_million=float(default.input_price_per_million),
                output_price_per_million=float(default.output_price_per_million),
                cache_read_price_per_million=as_float(default.cache_read_price_per_million),
                cache_write_price_per_million=as_float(default.cache_write_price_per_million),
                cache_write_1h_price_per_million=as_float(default.cache_write_1h_price_per_million),
            )
        return prices

    @staticmethod
    async def _defaults_for(provider: str, models: Sequence[str]) -> dict[str, ModelPricing]:
        """Community default rates, resolved off the loop.

        Deliberately not gated on ``default_pricing_enabled``: that switch
        governs the silent billing-time fallback, and this is an admin
        explicitly offering a model, which needs a rate stored to be served at
        all on a ``require_pricing`` deployment.
        """
        wanted = sorted(set(models))
        if not wanted:
            return {}
        as_of = datetime.now(tz=UTC)
        return await asyncio.to_thread(_resolve_defaults, provider, wanted, as_of)

    @staticmethod
    def _credential(key: OrgProviderKey) -> dict[str, object] | None:
        """The client arguments this key dials with, or None when it will not decrypt."""
        if key.encrypted_api_key is None:
            return {}
        try:
            return {"api_key": decrypt_secret(key.encrypted_api_key)}
        except (SecretBoxUnavailableError, SecretDecryptionError):
            return None

    @staticmethod
    async def _dial(
        key: OrgProviderKey, credential: dict[str, object], *, timeout: float
    ) -> ProviderDiscovery:
        """Ask the provider what it serves on this key. Never raises, never echoes the key."""
        api_key = credential.get("api_key")
        return await test_provider_credentials(
            key.provider,
            api_key=api_key if isinstance(api_key, str) else None,
            api_base=key.api_base,
            client_args=dict(key.client_args) if key.client_args else None,
            timeout=timeout,
        )
