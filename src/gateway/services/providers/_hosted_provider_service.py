"""The providers this deployment serves hosted inference on, and the models it offers on them.

One encrypted upstream credential per any-llm implementation, serving a request
that brings no credential of its own and names no configured instance. The
operator surface administers the rows; the runtime path reads one back to
serve a request. Both go through this service, so the "never return the key"
rule has one home.

No transaction is open across an upstream dial or across the dataset walk that
resolves community defaults: every use case reads in one block, dials, then
writes in another.
"""

import uuid

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
    HostedProviderUnsafeApiBaseError,
)
from gateway.models.providers import HostedProvider, HostedProviderModel
from gateway.models.secret_fields import carries_redaction, restore_redacted_values
from gateway.repositories.providers import (
    HostedModelConflict,
    HostedProviderConflict,
    HostedProviderModelRepository,
    HostedProviderRepository,
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
from gateway.services.model_discovery_service import ProviderDiscovery, test_provider_credentials
from gateway.services.pricing import DeploymentPricingService
from gateway.services.providers._hosted_catalog_sweep import UNDECRYPTABLE, HostedCatalogSweep, LiveByoPairs
from gateway.services.providers._hosted_credentials import (
    ResolvedHostedProvider,
    client_args_of,
    credential_of,
    encrypt_client_args,
    encrypt_key,
    host_of,
    last4,
    lookup_name,
    normalize_api_base,
    normalize_provider,
)
from gateway.services.providers._hosted_offers import HostedOffers
from gateway.services.providers._hosted_runtime import HostedProviderRuntime
from gateway.services.url_safety import UnsafeURLError, validate_provider_api_base


class HostedProviderService:
    """Configure the deployment's hosted providers, and resolve one to serve a request."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        config: GatewayConfig,
        providers: HostedProviderRepository,
        models: HostedProviderModelRepository,
        pricing: DeploymentPricingService,
        live_byo_pairs: LiveByoPairs,
    ) -> None:
        """Bind the unit of work and the collaborators this service composes.

        Rates are read and written through ``pricing``, the pricing domain's
        service on the same unit of work. ``live_byo_pairs`` answers which
        organizations hold a live key for each of the given providers; a
        callable, because the rows it reads are another domain's.
        """
        self.uow = uow
        self.config = config
        self.providers = providers
        self.models = models
        self.offers = HostedOffers(models, pricing)
        self.runtime = HostedProviderRuntime(uow, providers=providers, models=models)
        self.sweep = HostedCatalogSweep(
            uow,
            config=config,
            providers=providers,
            models=models,
            offers=self.offers,
            pricing=pricing,
            dial=self._dial,
            live_byo_pairs=live_byo_pairs,
        )

    async def list_providers(self, *, skip: int = 0, limit: int = 100) -> HostedProvidersPublic:
        """One page of hosted providers, enabled or not."""
        async with self.uow:
            rows, count = await self.providers.list_page(skip=skip, limit=limit)
            return HostedProvidersPublic(data=[self._public(row) for row in rows], count=count)

    async def create_provider(self, request: HostedProviderCreateRequest) -> HostedProviderPublic:
        """Configure a provider, offering everything it lists on the new key.

        A provider that will not list yields a provider with no models rather
        than a failed create: the credential may still be right for dispatch.

        Raises:
            HostedProviderUnknownProviderError: ``provider`` names no any-llm implementation.
            HostedProviderUnsafeApiBaseError: ``api_base`` is refused by the SSRF gate.
            SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
            HostedProviderAlreadyExistsError: that provider is already configured.
        """
        provider = normalize_provider(request.provider)
        api_base = normalize_api_base(request.api_base)
        await self._validate_api_base(api_base)
        encrypted = encrypt_key(request.api_key)
        encrypted_client_args = encrypt_client_args(request.client_args)

        async with self.uow:
            if await self.providers.get_by_provider(provider) is not None:
                raise HostedProviderAlreadyExistsError(provider)
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
                )
            except HostedProviderConflict as conflict:
                # The pre-check races the insert; the unique constraint decides.
                raise HostedProviderAlreadyExistsError(conflict.provider) from conflict
            await self.offers.offer(provider, discovered, defaults)
            return HostedProviderPublic.from_row(row, client_args=request.client_args)

    async def update_provider(self, provider: str, request: HostedProviderUpdateRequest) -> HostedProviderPublic:
        """Rotate the key, repoint the base, change the extras, or toggle a provider.

        A field left unset is left alone. An empty ``api_base`` clears the
        endpoint back to the provider's default. ``client_args`` omitted is left
        alone, null clears, and an entry echoed back as the mask keeps the
        stored value.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedProviderUnsafeApiBaseError: ``api_base`` is refused by the SSRF gate.
            HostedProviderKeyRequiredError: the base moves to another host and no key came with it.
            HostedProviderClientArgsUnreadableError: a masked entry has nothing stored to keep.
            SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
        """
        api_base = normalize_api_base(request.api_base)
        if request.api_base is not None:
            await self._validate_api_base(api_base)
        encrypted = encrypt_key(request.api_key) if request.api_key is not None else None

        async with self.uow:
            row = await self._provider_or_raise(provider)
            if request.api_base is not None:
                new_host = host_of(api_base)
                if new_host is not None and new_host != host_of(row.api_base) and encrypted is None:
                    # The stored key would otherwise be sent to a host the
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
            await self.providers.save(row)
            return self._public(row)

    async def delete_provider(self, provider: str) -> None:
        """Remove a provider and the roster offered on it. Pricing history stays.

        Raises:
            HostedProviderNotFoundError: no such provider.
        """
        async with self.uow:
            row = await self._provider_or_raise(provider)
            await self.models.delete_for_provider(row.provider)
            await self.providers.delete_row(row)

    async def list_models(self, provider: str, *, skip: int = 0, limit: int = 500) -> HostedModelsPublic:
        """One page of a provider's offered models, each priced as it currently serves.

        Raises:
            HostedProviderNotFoundError: no such provider.
        """
        async with self.uow:
            row = await self._provider_or_raise(provider)
            rows, count = await self.models.list_for_provider(row.provider, skip=skip, limit=limit)
        rates = await self.offers.current_rates(row.provider, rows)
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
            row = await self._provider_or_raise(provider)
            provider = row.provider
            if await self.models.get_by_model(provider, name) is not None:
                raise HostedModelAlreadyOfferedError(provider, name)

        priced = request.input_price_per_million is not None and request.output_price_per_million is not None
        defaults = {} if priced else await self.offers.defaults_for(provider, [name])

        async with self.uow:
            await self._provider_or_raise(provider)
            if priced:
                try:
                    [model] = await self.models.create_many([HostedProviderModel(provider=provider, model=name)])
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
            row = await self._provider_or_raise(provider)
            provider = row.provider
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
            row = await self._provider_or_raise(provider)
            model = await self._model_or_raise(row.provider, model_id)
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
            row = await self._provider_or_raise(provider)
            provider = row.provider
            offered = await self.models.names_for_provider(provider)
            credential = credential_of(row)

        if credential is None:
            return HostedModelsRefreshPublic(added=[], repriced=[], count=len(offered), error=UNDECRYPTABLE)
        discovery = await self._dial(provider, credential)
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
        """What the provider lists on the stored credential, without storing anything.

        Raises:
            HostedProviderNotFoundError: no such provider.
        """
        async with self.uow:
            row = await self._provider_or_raise(provider)
            provider = row.provider
            credential = credential_of(row)
        if credential is None:
            return HostedAvailableModelsPublic(provider=provider, models=[], error=UNDECRYPTABLE)
        discovery = await self._dial(provider, credential)
        return HostedAvailableModelsPublic(
            provider=provider,
            models=sorted(model.id for model in discovery.models),
            error=discovery.error,
            discovery_unsupported=discovery.discovery_unsupported,
        )

    async def refresh_catalog(self, *, apply: bool) -> HostedCatalogRefreshPublic:
        """Sweep the deployment price list against the roster, or preview the sweep.

        Raises:
            HostedCatalogEmptyError: no provider offers a model.
        """
        return await self.sweep.run(apply=apply)

    async def serveable_models(self) -> dict[str, frozenset[str]]:
        """Each serveable provider with the models switched on under it."""
        return await self.runtime.serveable_models()

    async def resolve(self, provider: str, model: str | None = None) -> ResolvedHostedProvider | None:
        """A usable credential for ``provider``, or None."""
        return await self.runtime.resolve(provider, model)

    async def _dial(self, provider: str, credential: ResolvedHostedProvider) -> ProviderDiscovery:
        return await test_provider_credentials(
            provider,
            api_key=credential.api_key,
            api_base=credential.api_base,
            client_args=credential.client_args,
            timeout=self.config.model_discovery_timeout_seconds,
        )

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
    def _public(row: HostedProvider) -> HostedProviderPublic:
        return HostedProviderPublic.from_row(row, client_args=client_args_of(row))

    async def _provider_or_raise(self, provider: str) -> HostedProvider:
        row = await self.providers.get_by_provider(lookup_name(provider))
        if row is None:
            raise HostedProviderNotFoundError(provider)
        return row

    async def _model_or_raise(self, provider: str, model_id: uuid.UUID) -> HostedProviderModel:
        model = await self.models.get_in_provider(model_id, provider)
        if model is None:
            raise HostedModelNotFoundError(model_id)
        return model


def _listed(discovery: ProviderDiscovery) -> set[str]:
    if discovery.error is not None or discovery.discovery_unsupported:
        return set()
    return {model.id for model in discovery.models}


__all__ = ["HostedProviderService", "LiveByoPairs", "ResolvedHostedProvider"]
