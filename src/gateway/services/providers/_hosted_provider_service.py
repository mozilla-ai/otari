"""The providers this deployment serves hosted inference on, and the models it offers on them.

One encrypted upstream credential per any-llm implementation, serving a request
that brings no credential of its own and names no configured instance. The
operator surface administers the rows; the runtime path reads one back to
serve a request. Both go through this service, so the "never return the key"
rule has one home.

The providers are the stored rows merged with the ``hosted_providers:`` block
(``_hosted_sources``), a row winning over an entry of the same provider. An
entry is never written or encrypted, so the API can read one but not change it;
its models are administered as a stored provider's are.

No transaction is open across an upstream dial or across the dataset walk that
resolves community defaults: every use case reads in one block, dials, then
writes in another.
"""

import functools
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from any_llm.types.model import Model

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.providers_exceptions import (
    HostedModelAlreadyOfferedError,
    HostedModelNameRequiredError,
    HostedModelNotFoundError,
    HostedProviderAlreadyExistsError,
    HostedProviderClientArgsUnreadableError,
    HostedProviderKeyRequiredError,
    HostedProviderNotFoundError,
    HostedProviderReadOnlyError,
    HostedProviderUnknownOrganizationError,
    HostedProviderUnsafeApiBaseError,
)
from gateway.models.providers import HostedProvider, HostedProviderModel
from gateway.models.secret_fields import carries_redaction, restore_redacted_values
from gateway.repositories.providers import (
    HostedModelConflict,
    HostedProviderConflict,
    HostedProviderModelRepository,
    HostedProviderRepository,
    ProviderEndpointRepository,
)
from gateway.schemas.providers import (
    HostedAvailableModelsPublic,
    HostedCatalogRefreshPublic,
    HostedModelCreateRequest,
    HostedModelPublic,
    HostedModelsPublic,
    HostedModelsRefreshPublic,
    HostedModelUpdateRequest,
    HostedProviderCreateRequest,
    HostedProviderPublic,
    HostedProvidersPublic,
    HostedProviderUpdateRequest,
)
from gateway.services.model_discovery_service import (
    DiscoveryDial,
    ProviderDiscovery,
    get_model_cache,
    test_provider_credentials,
)
from gateway.services.pricing import DeploymentPricingService
from gateway.services.providers._hosted_catalog_sweep import UNDECRYPTABLE, HostedCatalogSweep, LiveByoPairs
from gateway.services.providers._hosted_credentials import (
    ResolvedHostedProvider,
    client_args_of,
    encrypt_client_args,
    encrypt_key,
    last4,
    normalize_api_base,
    normalize_provider,
    origin_of,
    public_api_base,
    unmask_api_base,
)
from gateway.services.providers._hosted_offers import HostedOffers
from gateway.services.providers._hosted_runtime import HostedProviderRuntime
from gateway.services.providers._hosted_sources import HostedSource, HostedSources
from gateway.services.url_safety import UnsafeURLError, validate_provider_api_base

OrganizationExists = Callable[[uuid.UUID], Awaitable[bool]]


@dataclass(frozen=True)
class HostedConfigAudit:
    """What boot reports about the ``hosted_providers:`` block.

    ``shadowed`` names the entries a stored row of the same provider wins
    over; ``unknown_restrictions`` maps an entry to the organization its
    restriction names and the deployment does not hold.
    """

    shadowed: list[str]
    unknown_restrictions: dict[str, uuid.UUID]


class HostedProviderService:
    """Configure the deployment's hosted providers, and resolve one to serve a request."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        config: GatewayConfig,
        providers: HostedProviderRepository,
        sources: HostedSources,
        models: HostedProviderModelRepository,
        endpoints: ProviderEndpointRepository,
        pricing: DeploymentPricingService,
        live_byo_pairs: LiveByoPairs,
        organization_exists: OrganizationExists,
    ) -> None:
        """Bind the unit of work and the collaborators this service composes.

        ``sources`` is the merged view of the stored rows and the configured
        entries, read over ``providers``. Rates are read and written through
        ``pricing``, the pricing domain's service on the same unit of work.
        ``live_byo_pairs`` answers which
        organizations hold a live key for each of the given providers, and
        ``organization_exists`` whether the deployment holds an organization a
        restriction names; callables, because the rows they read are another
        domain's.
        """
        self.uow = uow
        self.config = config
        self.providers = providers
        self.models = models
        self.organization_exists = organization_exists
        self.sources = sources
        self.offers = HostedOffers(models, pricing)
        self.runtime = HostedProviderRuntime(uow, sources=self.sources, models=models)
        self.sweep = HostedCatalogSweep(
            uow,
            config=config,
            sources=self.sources,
            models=models,
            endpoints=endpoints,
            offers=self.offers,
            pricing=pricing,
            dial=self._dial,
            live_byo_pairs=live_byo_pairs,
        )

    async def list_providers(self, *, skip: int = 0, limit: int = 100) -> HostedProvidersPublic:
        """One page of hosted providers by name, enabled or not, stored and configured alike.

        Merged in memory: there is at most one per any-llm implementation.
        """
        async with self.uow:
            sources = await self.sources.all()
            return HostedProvidersPublic(
                data=[self._public(source) for source in sources[skip : skip + limit]], count=len(sources)
            )

    async def create_provider(self, request: HostedProviderCreateRequest) -> HostedProviderPublic:
        """Configure a provider, offering everything it lists on the new key.

        A provider the configuration already defines may be created too, and
        the row then wins over the entry. A provider that will not list yields a provider with no models rather
        than a failed create: the credential may still be right for dispatch.

        Raises:
            HostedProviderUnknownProviderError: ``provider`` names no any-llm implementation.
            HostedProviderUnsafeApiBaseError: ``api_base`` is refused by the SSRF gate.
            SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
            HostedProviderUnknownOrganizationError: the restriction names no organization here.
            HostedProviderAlreadyExistsError: that provider is already configured.
        """
        provider = normalize_provider(request.provider)
        api_base = normalize_api_base(request.api_base)
        await self._validate_api_base(api_base)
        encrypted = encrypt_key(request.api_key)
        encrypted_client_args = encrypt_client_args(request.client_args)

        async with self.uow:
            await self._require_organization(request.restricted_to_organization_id)
            if await self.providers.get_by_provider(provider) is not None:
                raise HostedProviderAlreadyExistsError(provider)
            shadows_config = provider in self.sources.entries()
            offered = await self.models.names_for_provider(provider)

        credential = ResolvedHostedProvider(api_key=request.api_key, api_base=api_base, client_args=request.client_args)
        discovered = sorted(_listed(await self._dial(provider, credential)) - offered)
        defaults = await self.offers.defaults_for(provider, discovered)

        async with self.uow:
            try:
                row = await self.providers.insert(
                    provider=provider,
                    encrypted_api_key=encrypted,
                    api_key_last4=last4(request.api_key),
                    api_base=api_base,
                    encrypted_client_args=encrypted_client_args,
                    enabled=request.enabled,
                    restricted_to_organization_id=request.restricted_to_organization_id,
                )
            except HostedProviderConflict as conflict:
                # The pre-check races the insert; the unique constraint decides.
                raise HostedProviderAlreadyExistsError(conflict.provider) from conflict
            await self.offers.offer(provider, discovered, defaults)
            return HostedProviderPublic.from_row(
                row,
                client_args=request.client_args,
                api_base=public_api_base(row.api_base),
                shadows_config=shadows_config,
            )

    async def update_provider(self, provider: str, request: HostedProviderUpdateRequest) -> HostedProviderPublic:
        """Rotate the key, repoint the base, change the extras, or toggle a provider.

        A field left unset is left alone. An empty ``api_base`` clears the
        endpoint back to the provider's default. ``client_args`` omitted is left
        alone, null clears, and an entry echoed back as the mask keeps the
        stored value. The restriction reads the same way: omitted keeps it,
        null lifts it.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedProviderReadOnlyError: the provider is defined in configuration.
            HostedProviderUnknownOrganizationError: the restriction names no organization here.
            HostedProviderUnsafeApiBaseError: ``api_base`` is refused by the SSRF gate.
            HostedProviderApiBaseMaskedError: ``api_base`` carries the mask but is not the stored one echoed back.
            HostedProviderKeyRequiredError: the base moves to another origin and no key came with it.
            HostedProviderClientArgsUnreadableError: a masked entry has nothing stored to keep.
            SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
        """
        api_base = normalize_api_base(request.api_base)
        if request.api_base is not None:
            await self._validate_api_base(api_base)
        encrypted = encrypt_key(request.api_key) if request.api_key is not None else None

        async with self.uow:
            row, source = await self._stored_or_raise(provider)
            if request.api_base is not None:
                if api_base is not None:
                    api_base = unmask_api_base(api_base, row.api_base)
                new_origin = origin_of(api_base)
                if new_origin is not None and new_origin != origin_of(row.api_base) and encrypted is None:
                    # The stored key would otherwise be sent to an endpoint the
                    # operator chose without ever having to know the key.
                    raise HostedProviderKeyRequiredError
                row.api_base = api_base
            if request.api_key is not None and encrypted is not None:
                row.encrypted_api_key = encrypted
                row.api_key_last4 = last4(request.api_key)
            if "client_args" in request.model_fields_set:
                stored = client_args_of(row)
                if stored is None and carries_redaction(request.client_args):
                    # Taken literally, the mask would be stored as the value.
                    raise HostedProviderClientArgsUnreadableError
                row.encrypted_client_args = encrypt_client_args(restore_redacted_values(request.client_args, stored))
            if request.enabled is not None:
                row.enabled = request.enabled
            if "restricted_to_organization_id" in request.model_fields_set:
                await self._require_organization(request.restricted_to_organization_id)
                row.restricted_to_organization_id = request.restricted_to_organization_id
            await self.providers.save(row)
            public = self._public(source)
        # The cached listing was made on what this request may just have changed.
        get_model_cache().clear(row.provider)
        return public

    async def delete_provider(self, provider: str) -> None:
        """Remove a provider and the roster offered on it. Pricing history stays.

        A row that shadows a configuration entry takes no roster with it: the
        entry serves again, on the same models.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedProviderReadOnlyError: the provider is defined in configuration.
        """
        async with self.uow:
            row, source = await self._stored_or_raise(provider)
            name = row.provider
            if not source.shadows_config:
                await self.models.delete_for_provider(name)
            await self.providers.delete_row(row)
        get_model_cache().clear(name)

    async def list_models(self, provider: str, *, skip: int = 0, limit: int = 500) -> HostedModelsPublic:
        """One page of a provider's offered models, each priced as it currently serves.

        Raises:
            HostedProviderNotFoundError: no such provider.
        """
        async with self.uow:
            provider = (await self._provider_or_raise(provider)).provider
            rows, count = await self.models.list_for_provider(provider, skip=skip, limit=limit)
        rates = await self.offers.current_rates(provider, rows)
        return HostedModelsPublic(
            data=[HostedModelPublic.from_row(model, rates=rates.get(model.model)) for model in rows], count=count
        )

    async def add_model(self, provider: str, request: HostedModelCreateRequest) -> HostedModelPublic:
        """Offer a model on a provider, at the rate given or at the seeded default.

        Membership and rate land in one transaction, so a model offered at a
        rate cannot be served at another.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedModelNameRequiredError: the name is blank once trimmed.
            HostedModelAlreadyOfferedError: the model is already offered there.
        """
        name = request.model.strip()
        if not name:
            raise HostedModelNameRequiredError

        async with self.uow:
            provider = (await self._provider_or_raise(provider)).provider
            if await self.models.get_by_model(provider, name) is not None:
                raise HostedModelAlreadyOfferedError(provider, name)

        priced = request.input_price_per_million is not None and request.output_price_per_million is not None
        defaults = {} if priced else await self.offers.defaults_for(provider, [name])

        async with self.uow:
            await self._provider_or_raise(provider)
            if priced:
                try:
                    [model] = await self.models.create_many(provider, [name], enabled={name: True})
                except HostedModelConflict as conflict:
                    raise HostedModelAlreadyOfferedError(provider, conflict.model) from conflict
                await self.offers.write_rate(provider, name, request)
            else:
                [model] = await self.offers.offer(provider, [name], defaults)
        rates = await self.offers.current_rates(provider, [model])
        return HostedModelPublic.from_row(model, rates=rates.get(name))

    async def update_model(
        self, provider: str, model_id: uuid.UUID, request: HostedModelUpdateRequest
    ) -> HostedModelPublic:
        """Reprice one offered model, toggle whether it is served, or both.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedModelNotFoundError: no such model on that provider.
        """
        async with self.uow:
            provider = (await self._provider_or_raise(provider)).provider
            model = await self._model_or_raise(provider, model_id)
            priced = request.input_price_per_million is not None and request.output_price_per_million is not None
            if priced:
                await self.offers.write_rate(provider, model.model, request)
            if request.enabled is not None:
                model.enabled = request.enabled
                await self.models.save(model)
        rates = await self.offers.current_rates(provider, [model])
        return HostedModelPublic.from_row(model, rates=rates.get(model.model))

    async def remove_model(self, provider: str, model_id: uuid.UUID) -> None:
        """Stop offering a model. Its pricing history stays.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedModelNotFoundError: no such model on that provider.
        """
        async with self.uow:
            provider = (await self._provider_or_raise(provider)).provider
            model = await self._model_or_raise(provider, model_id)
            await self.models.delete_row(model)

    async def refresh_models(self, provider: str) -> HostedModelsRefreshPublic:
        """Ask the provider again, offer what is newly listed, and move seeded rates.

        Additive: a model the provider no longer lists stays offered, because
        delisting one is the switch's decision.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedModelAlreadyOfferedError: another refresh offered a model first.
        """
        async with self.uow:
            source = await self._provider_or_raise(provider)
            provider = source.provider
            offered = await self.models.names_for_provider(provider)
            credential = source.credential()
            declared = source.declared_models

        if credential is None:
            return HostedModelsRefreshPublic(added=[], repriced=[], count=len(offered), error=UNDECRYPTABLE)
        discovery = await self._dial(provider, credential, declared)
        if discovery.error is not None or discovery.discovery_unsupported:
            return HostedModelsRefreshPublic(
                added=[],
                repriced=[],
                count=len(offered),
                error=discovery.error,
                discovery_unsupported=discovery.discovery_unsupported,
            )
        added = sorted(_listed(discovery) - offered)
        defaults = await self.offers.defaults_for(provider, [*added, *offered])

        async with self.uow:
            await self._provider_or_raise(provider)
            repriced = await self.offers.reseed(provider, defaults)
            await self.offers.offer(provider, added, defaults)
        return HostedModelsRefreshPublic(added=added, repriced=repriced, count=len(offered) + len(added))

    async def available_models(self, provider: str) -> HostedAvailableModelsPublic:
        """What the provider lists on its credential, without storing anything.

        Raises:
            HostedProviderNotFoundError: no such provider.
        """
        async with self.uow:
            source = await self._provider_or_raise(provider)
            provider = source.provider
            credential = source.credential()
            declared = source.declared_models
        if credential is None:
            return HostedAvailableModelsPublic(provider=provider, models=[], error=UNDECRYPTABLE)
        discovery = await self._dial(provider, credential, declared)
        return HostedAvailableModelsPublic(
            provider=provider,
            models=sorted(model.id for model in discovery.models),
            error=discovery.error,
            discovery_unsupported=discovery.discovery_unsupported,
        )

    async def refresh_catalog(self, *, apply: bool) -> HostedCatalogRefreshPublic:
        """Sweep the deployment price list against the roster, or preview the sweep.

        Raises:
            HostedCatalogEmptyError: no provider offers a model, and no orphaned roster is left.
        """
        return await self.sweep.run(apply=apply)

    async def serveable_models(self, *, organization_id: uuid.UUID | None) -> dict[str, frozenset[str]]:
        """Each provider serveable to ``organization_id``, with the models switched on under it.

        ``None`` is the deployment's own view, which sees every provider.
        """
        return await self.runtime.serveable_models(organization_id=organization_id)

    async def resolve(
        self, provider: str, model: str | None = None, *, organization_id: uuid.UUID | None
    ) -> ResolvedHostedProvider | None:
        """A usable credential for ``provider`` on behalf of ``organization_id``, or None."""
        return await self.runtime.resolve(provider, model, organization_id=organization_id)

    async def discovery_dials(self) -> dict[str, DiscoveryDial]:
        """Each provider switched on, with the dial that lists its models on its credential.

        The deployment's view, for model discovery and provider health: a
        provider restricted to one organization is here too, and so is one
        defined in configuration. One whose ciphertext no configured key reads
        is here with a dial that reports that rather than calling out, because
        an operator checking health is the reader who has to learn it.
        """
        async with self.uow:
            serving = [
                (source.provider, source.credential(quiet=True), source.declared_models)
                for source in await self.sources.all()
                if source.enabled
            ]
        dials: dict[str, DiscoveryDial] = {}
        for provider, credential, declared in serving:
            dials[provider] = (
                functools.partial(_undecryptable, provider)
                if credential is None
                else functools.partial(self._dial, provider, credential, declared)
            )
        return dials

    async def audit_config(self) -> HostedConfigAudit:
        """What boot reports about the ``hosted_providers:`` block: shadowed entries and unknown restrictions."""
        async with self.uow:
            sources = await self.sources.all()
            unknown: dict[str, uuid.UUID] = {}
            for provider, entry in self.sources.entries().items():
                restriction = entry.restricted_to_organization_id
                if restriction is not None and not await self.organization_exists(restriction):
                    unknown[provider] = restriction
        return HostedConfigAudit(
            shadowed=[source.provider for source in sources if source.shadows_config],
            unknown_restrictions=unknown,
        )

    async def _dial(
        self, provider: str, credential: ResolvedHostedProvider, declared: tuple[str, ...] = ()
    ) -> ProviderDiscovery:
        """List the provider's models, falling back to ``declared`` where it has no listing to ask.

        Only a provider without a model-listing endpoint falls back, as a
        configured instance does. Any other failure, a bad key or an
        unreachable upstream, is reported as it came, so health still says so.
        """
        discovery = await test_provider_credentials(
            provider,
            api_key=credential.api_key,
            api_base=credential.api_base,
            client_args=credential.client_args,
            timeout=self.config.model_discovery_timeout_seconds,
        )
        if declared and discovery.discovery_unsupported:
            models = [Model(id=model, created=0, object="model", owned_by=provider) for model in declared]
            return ProviderDiscovery(provider=provider, models=models)
        return discovery

    @staticmethod
    async def _validate_api_base(api_base: str | None) -> None:
        """Refuse an ``api_base`` the SSRF gate rejects; a no-op in the default allow-all state."""
        if not api_base:
            return
        try:
            await validate_provider_api_base(api_base)
        except UnsafeURLError as exc:
            raise HostedProviderUnsafeApiBaseError(str(exc)) from None

    @staticmethod
    def _public(source: HostedSource) -> HostedProviderPublic:
        if source.row is not None:
            return HostedProviderPublic.from_row(
                source.row,
                client_args=client_args_of(source.row),
                api_base=public_api_base(source.row.api_base),
                shadows_config=source.shadows_config,
            )
        entry = source.entry
        if entry is None:  # pragma: no cover - a source holds a row or an entry
            raise HostedProviderNotFoundError(source.provider)
        return HostedProviderPublic.from_config(
            provider=entry.provider,
            api_key_last4=last4(entry.api_key) if entry.api_key is not None else None,
            api_base=public_api_base(entry.api_base),
            client_args=entry.client_args,
            enabled=entry.enabled,
            restricted_to_organization_id=entry.restricted_to_organization_id,
        )

    async def _require_organization(self, organization_id: uuid.UUID | None) -> None:
        if organization_id is not None and not await self.organization_exists(organization_id):
            raise HostedProviderUnknownOrganizationError(organization_id)

    async def _provider_or_raise(self, provider: str) -> HostedSource:
        source = await self.sources.get(provider)
        if source is None:
            raise HostedProviderNotFoundError(provider)
        return source

    async def _stored_or_raise(self, provider: str) -> tuple[HostedProvider, HostedSource]:
        """The provider's row, which this API may change, refusing a provider defined in configuration."""
        source = await self._provider_or_raise(provider)
        if source.row is None:
            raise HostedProviderReadOnlyError(source.provider)
        return source.row, source

    async def _model_or_raise(self, provider: str, model_id: uuid.UUID) -> HostedProviderModel:
        model = await self.models.get_in_provider(model_id, provider)
        if model is None:
            raise HostedModelNotFoundError(model_id)
        return model


async def _undecryptable(provider: str) -> ProviderDiscovery:
    return ProviderDiscovery(provider=provider, models=[], error=UNDECRYPTABLE)


def _listed(discovery: ProviderDiscovery) -> set[str]:
    if discovery.error is not None or discovery.discovery_unsupported:
        return set()
    return {model.id for model in discovery.models}


__all__ = [
    "HostedConfigAudit",
    "HostedProviderService",
    "LiveByoPairs",
    "OrganizationExists",
    "ResolvedHostedProvider",
]
