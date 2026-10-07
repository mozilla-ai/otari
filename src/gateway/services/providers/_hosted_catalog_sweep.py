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
from gateway.repositories.providers import HostedProviderModelRepository, HostedProviderRepository
from gateway.schemas.providers import HostedCatalogProviderRefreshPublic, HostedCatalogRefreshPublic
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.pricing import DeploymentPricingService
from gateway.services.provider_kwargs import split_selector
from gateway.services.providers._hosted_catalog import CATALOG_SAMPLE, classify_priced_keys, kept_groups
from gateway.services.providers._hosted_credentials import ResolvedHostedProvider, credential_of
from gateway.services.providers._hosted_offers import HostedOffers

Dial = Callable[[str, ResolvedHostedProvider], Awaitable[ProviderDiscovery]]
LiveByoPairs = Callable[[Collection[str]], Awaitable[set[tuple[uuid.UUID, str]]]]

UNDECRYPTABLE = (
    "The stored key cannot be decrypted. Was OTARI_SECRET_KEY rotated? Re-enter the credential, then try again."
)
REMOVED_MID_SWEEP = "The provider was removed while the sweep ran, so nothing was offered on it."


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
        providers: HostedProviderRepository,
        models: HostedProviderModelRepository,
        offers: HostedOffers,
        pricing: DeploymentPricingService,
        dial: Dial,
        live_byo_pairs: LiveByoPairs,
    ) -> None:
        """``live_byo_pairs`` answers which organizations hold a live key for each of the given providers."""
        self.uow = uow
        self.config = config
        self.providers = providers
        self.models = models
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
                roster to reconcile against.
        """
        async with self.uow:
            serving = [
                (row.provider, credential_of(row), await self.models.names_for_provider(row.provider))
                for row in await self.providers.list_all()
                if row.enabled
            ]

        # One provider at a time, each bounded by the discovery timeout.
        dialed = [await self._dial_one(provider, credential, offered) for provider, credential, offered in serving]
        defaults: dict[str, dict[str, ModelPricing]] = {}
        if apply:
            for (provider, _, offered), answer in zip(serving, dialed, strict=True):
                defaults[provider] = await self.offers.defaults_for(provider, [*answer.added, *offered])

        async with self.uow:
            if apply:
                # Read again rather than carried across: a provider removed
                # while the dials ran must not get rows with nothing to own them.
                present = {row.provider for row in await self.providers.list_all() if row.enabled}
                for answer in dialed:
                    if answer.provider not in present:
                        answer.added = []
                        answer.error = REMOVED_MID_SWEEP
                        continue
                    answer.repriced = await self.offers.reseed(answer.provider, defaults[answer.provider])
                    await self.offers.offer(answer.provider, answer.added, defaults[answer.provider])
            offered_keys = await self.models.offered_keys()
            if not apply:
                # What an apply would offer, folded in without writing it.
                offered_keys |= {f"{answer.provider}:{model}" for answer in dialed for model in answer.added}
            if not offered_keys:
                raise HostedCatalogEmptyError

            verdicts = classify_priced_keys(
                await self.pricing.all_keys(),
                offered=frozenset(offered_keys),
                deployment_instances=frozenset(self.config.providers),
                search_providers=frozenset(self.config.search_tool_providers()),
            )
            doomed = [verdict for verdict in verdicts if verdict.verdict == "removed"]
            removed = sorted(verdict.model_key for verdict in doomed)
            override_keys = sorted({verdict.canonical_key for verdict in doomed})
            override_ids = await self._doomed_override_ids(override_keys)
            if apply:
                price_rows, override_rows = await self.pricing.delete_keys(removed, override_ids)
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
        )

    async def _doomed_override_ids(self, canonical_keys: list[str]) -> list[uuid.UUID]:
        """The overrides above ``canonical_keys`` a sweep takes with them.

        An override is spared when its own organization holds a live key for
        the provider: the deployment's rate is going because nothing here serves
        the model, but that organization reaches it on its own key and this is
        its rate on it. Live means unarchived, whether or not the stored secret
        decrypts here.
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
        self, provider: str, credential: ResolvedHostedProvider | None, offered: set[str]
    ) -> HostedCatalogProviderRefreshPublic:
        """Ask one provider what it serves, reporting a refusal rather than raising."""
        if credential is None:
            return HostedCatalogProviderRefreshPublic(
                provider=provider, added=[], error=UNDECRYPTABLE, credential_unreadable=True
            )
        discovery = await self.dial(provider, credential)
        if discovery.error is not None or discovery.discovery_unsupported:
            return HostedCatalogProviderRefreshPublic(
                provider=provider,
                added=[],
                error=discovery.error,
                discovery_unsupported=discovery.discovery_unsupported,
            )
        return HostedCatalogProviderRefreshPublic(
            provider=provider, added=sorted({model.id for model in discovery.models} - offered)
        )
