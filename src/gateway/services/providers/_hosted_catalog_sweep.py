"""The catalog sweep: re-ask every hosted provider what it lists, then take the rest off the price list.

A stored price is what lists a model the catalog did not discover, so a price
under a provider nothing here serves puts a model on every Models page that no
request can be served on. The sweep offers what each provider newly lists,
then classifies the price list against the roster and removes what it does not
account for, with the organization overrides above it, except an override held
by an organization with a live key of its own for the provider.

NOTE: this removes pricing history, which removing one model keeps. A sweep is
an operator saying the catalog should match the roster, and a model's rates are
what keeps it on the catalog.
"""

import uuid
from collections.abc import Awaitable, Callable, Collection

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.providers_exceptions import HostedCatalogEmptyError
from gateway.models.pricing import ModelPricing
from gateway.repositories.providers import HostedProviderModelRepository, ProviderEndpointRepository
from gateway.schemas.providers import HostedCatalogProviderRefreshPublic, HostedCatalogRefreshPublic
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.pricing import DeploymentPricingService
from gateway.services.provider_kwargs import split_selector
from gateway.services.providers._hosted_catalog import CATALOG_SAMPLE, classify_priced_keys, kept_groups
from gateway.services.providers._hosted_credentials import ResolvedHostedProvider
from gateway.services.providers._hosted_offers import HostedOffers
from gateway.services.providers._hosted_sources import HostedSource, HostedSources

# A provider, the connection it serves on, and the models configuration
# declares for it when it cannot list them.
Dial = Callable[[str, ResolvedHostedProvider, tuple[str, ...]], Awaitable[ProviderDiscovery]]
LiveByoPairs = Callable[[Collection[str]], Awaitable[set[tuple[uuid.UUID, str]]]]

UNDECRYPTABLE = (
    "The stored key cannot be decrypted. Was OTARI_SECRET_KEY rotated? Re-enter the credential, then try again."
)
REMOVED_MID_SWEEP = (
    "The provider was removed, switched off or replaced while the sweep ran, so nothing was offered on it."
)


def _prefix(model_key: str) -> str | None:
    split = split_selector(model_key)
    return None if split is None else split[0]


class HostedCatalogSweep:
    """Run the sweep, or preview it without a write."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        config: GatewayConfig,
        sources: HostedSources,
        models: HostedProviderModelRepository,
        endpoints: ProviderEndpointRepository,
        offers: HostedOffers,
        pricing: DeploymentPricingService,
        dial: Dial,
        live_byo_pairs: LiveByoPairs,
    ) -> None:
        """``live_byo_pairs`` answers which organizations hold a live key for each of the given providers.

        Live means unarchived, whether or not the stored secret decrypts here.
        """
        self.uow = uow
        self.config = config
        self.sources = sources
        self.models = models
        self.endpoints = endpoints
        self.offers = offers
        self.pricing = pricing
        self.dial = dial
        self.live_byo_pairs = live_byo_pairs

    async def run(self, *, apply: bool) -> HostedCatalogRefreshPublic:
        """Dial every enabled provider, offer what is new, then classify and remove.

        ``apply=False`` runs the same dials and the same classification and
        writes nothing, so the preview cannot disagree with the apply. It dials
        rather than reading the roster alone, because a model the provider
        still lists would be offered by the apply and kept.

        Raises:
            HostedCatalogEmptyError: no provider offers a model, so there is no
                roster to reconcile against, and no orphaned roster is left to
                remove.
        """
        async with self.uow:
            serving = [
                (
                    source,
                    source.credential(),
                    await self.models.names_for_provider(source.provider),
                )
                for source in await self.sources.all()
                if source.enabled
            ]

        # One provider at a time, each bounded by the discovery timeout.
        dialed = [await self._dial_one(source, credential, offered) for source, credential, offered in serving]
        defaults: dict[str, dict[str, ModelPricing]] = {}
        if apply:
            for (source, _, offered), answer in zip(serving, dialed, strict=True):
                defaults[source.provider] = await self.offers.defaults_for(source.provider, [*answer.added, *offered])

        async with self.uow:
            # Read again rather than carried across: a provider removed while
            # the dials ran must not get rows with nothing to own them.
            sources = await self.sources.all()
            defined = {source.provider for source in sources}
            # A roster outlives what defined its provider (a config entry taken
            # out, say), and serves nothing; it is not offered, and an apply
            # removes it with the prices it was keeping.
            orphaned = sorted(await self.models.rostered_providers() - defined)
            # Both paths drop what a provider removed, switched off or replaced
            # by another source mid-dial listed, so the preview classifies the
            # prices exactly as the apply, and a listing is never credited to a
            # credential that did not answer it.
            present = {source.provider: source.identity for source in sources if source.enabled}
            for (source, _, _), answer in zip(serving, dialed, strict=True):
                if present.get(answer.provider) != source.identity:
                    answer.added = []
                    answer.error = REMOVED_MID_SWEEP
                elif apply:
                    answer.repriced = await self.offers.reseed(answer.provider, defaults[answer.provider])
                    await self.offers.offer(answer.provider, answer.added, defaults[answer.provider])
            offered_keys = await self.models.offered_keys(defined)
            if not apply:
                # What an apply would offer, folded in without writing it.
                offered_keys |= {f"{answer.provider}:{model}" for answer in dialed for model in answer.added}
            if not offered_keys:
                if not orphaned:
                    raise HostedCatalogEmptyError
                # With nothing offered there is nothing to judge the price list
                # against, so it is left whole; the orphaned rosters still go.
                if apply:
                    for provider in orphaned:
                        await self.models.delete_for_provider(provider)
                return HostedCatalogRefreshPublic(
                    providers=dialed,
                    removed=[],
                    removed_count=0,
                    removed_price_rows=0,
                    removed_override_rows=0,
                    kept=[],
                    orphaned_rosters=orphaned,
                )

            verdicts = classify_priced_keys(
                await self.pricing.all_keys(),
                offered=frozenset(offered_keys),
                deployment_instances=frozenset(self.config.providers),
                search_providers=frozenset(self.config.search_tool_providers()),
                fetch_providers=frozenset(self.config.fetch_tool_providers()),
                owned_endpoints=frozenset(row.name for row in await self.endpoints.list_all()),
            )
            doomed = [verdict for verdict in verdicts if verdict.verdict == "removed"]
            removed = sorted(verdict.model_key for verdict in doomed)
            override_keys = sorted({verdict.canonical_key for verdict in doomed})
            override_ids = await self._doomed_override_ids(override_keys)
            if apply:
                price_rows, override_rows = await self.pricing.delete_keys(removed, override_ids)
                for provider in orphaned:
                    await self.models.delete_for_provider(provider)
            else:
                price_rows = sum(verdict.versions for verdict in doomed)
                override_rows = len(override_ids)

        return HostedCatalogRefreshPublic(
            providers=dialed,
            removed=removed[:CATALOG_SAMPLE],
            removed_count=len(removed),
            removed_price_rows=price_rows,
            removed_override_rows=override_rows,
            kept=kept_groups(verdicts),
            orphaned_rosters=orphaned,
        )

    async def _doomed_override_ids(self, canonical_keys: list[str]) -> list[uuid.UUID]:
        """The overrides above ``canonical_keys`` a sweep takes with them.

        An override is spared when its own organization holds a live key for
        the provider: the deployment's rate is going because nothing here
        offers the model, but that organization reaches it on its own key and
        this is its rate on it. An override of an organization without one goes,
        or it would keep pricing a model only the deployment's key could serve.
        """
        if not canonical_keys:
            return []
        overrides = await self.pricing.doomed_overrides(canonical_keys)
        providers = {prefix for _, _, key in overrides if (prefix := _prefix(key)) is not None}
        spared = await self.live_byo_pairs(providers) if providers else set()
        return [
            override_id
            for override_id, organization_id, key in overrides
            if (organization_id, _prefix(key)) not in spared
        ]

    async def _dial_one(
        self, source: HostedSource, credential: ResolvedHostedProvider | None, offered: set[str]
    ) -> HostedCatalogProviderRefreshPublic:
        """Ask one provider what it serves, reporting a refusal rather than raising."""
        provider = source.provider
        if credential is None:
            return HostedCatalogProviderRefreshPublic(
                provider=provider, added=[], error=UNDECRYPTABLE, credential_unreadable=True, source=source.source
            )
        discovery = await self.dial(provider, credential, source.declared_models)
        if discovery.error is not None or discovery.discovery_unsupported:
            return HostedCatalogProviderRefreshPublic(
                provider=provider,
                added=[],
                error=discovery.error,
                discovery_unsupported=discovery.discovery_unsupported,
                source=source.source,
            )
        return HostedCatalogProviderRefreshPublic(
            provider=provider, added=sorted({model.id for model in discovery.models} - offered), source=source.source
        )
