"""The providers this deployment serves hosted inference on, and the models it offers on them.

The deployment's own upstream credentials: one per any-llm implementation,
holding the key a request is served on when it brings no BYO credential and
names no configured instance. Two callers with very different rights share
this service, which is why it is one: the operator API administers the rows,
and the ``ModelProviderPort`` adapter reads one back to serve a request.
Keeping both here means the decryption path and the "never return the key"
rule sit in one place.

Three rules are worth stating once.

**A rate is written to ``model_pricing``, the deployment price list.** The
deployment settles the upstream bill for a hosted model, so its rate is the
deployment's and applies to every organization that has no override of its
own. That table is also what the catalog lists an undiscovered model from, so
offering a model and pricing it are the same write seen from two sides.

**``seeded_price_at`` is what tells a seeded rate from a chosen one.** A model
offered without a rate gets the community default stored as the deployment's
own version, and the row remembers that version's timestamp. While the latest
version still carries it, a refresh may move the rate with the dataset; any
later version, from this surface or ``POST /pricing``, is an operator's and is
left alone forever.

**Disabled until priced.** A model no rate could be found for is recorded and
not served, so a model the pricing data has not caught up with cannot be billed
at nothing. The switch is also how a model is withheld: rows are disabled rather
than deleted, so the model, its rate and its history stay where an operator can
turn it back on.

No transaction is open across an upstream dial or across the thread hop that
resolves community defaults, both of which can take the whole discovery
timeout. Every method reads in one block, dials, then writes in another.
"""

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from any_llm import LLMProvider

from gateway.core.config import PROVIDER_TYPE_ALIASES, GatewayConfig
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
    HostedProviderUnsafeApiBaseError,
)
from gateway.log_config import logger
from gateway.models.pricing import ModelPricing, PriceSource
from gateway.models.providers import HostedProvider, HostedProviderModel
from gateway.models.secret_fields import restore_redacted_values
from gateway.repositories.providers import (
    HostedModelConflict,
    HostedProviderConflict,
    HostedProviderModelRepository,
    HostedProviderRepository,
)
from gateway.schemas.providers import (
    HostedAvailableModelsPublic,
    HostedCatalogProviderRefreshPublic,
    HostedCatalogRefreshPublic,
    HostedModelCreateRequest,
    HostedModelPublic,
    HostedModelRates,
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
from gateway.services.pricing_service import default_pricing_enabled, normalize_effective_at
from gateway.services.providers._hosted_catalog import CATALOG_SAMPLE, classify_priced_keys, kept_groups
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.url_safety import UnsafeURLError, validate_provider_api_base

# What a client is told about where a rate came from, in ``models.pricing.PriceSource``'s
# spellings so this panel and the Models page name the same rungs.
PRICE_SOURCE_DEFAULT: PriceSource = "defaults"
PRICE_SOURCE_DEPLOYMENT: PriceSource = "deployment"

# The message an operator needs when a stored credential will not decrypt. The
# row is intact; the configured OTARI_SECRET_KEY just cannot read it any more.
_UNDECRYPTABLE = (
    "The stored key cannot be decrypted. Was OTARI_SECRET_KEY rotated? Re-enter the credential, then try again."
)

_CACHE_RATE_FIELDS = (
    "cache_read_price_per_million",
    "cache_write_price_per_million",
    "cache_write_1h_price_per_million",
)


@dataclass(frozen=True)
class ResolvedHostedProvider:
    """A decrypted, usable upstream connection. Never crosses the API boundary."""

    api_key: str
    api_base: str | None
    client_args: dict[str, Any] | None


def _last4(api_key: str) -> str | None:
    """The tail of a key, for telling one stored credential from another.

    None for anything four characters or shorter, where "the last four" would
    be the whole credential.
    """
    return api_key[-4:] if len(api_key) > 4 else None


def _normalize_api_base(api_base: str | None) -> str | None:
    """Fold a blank endpoint to "use the provider's default".

    A form that clears the box sends an empty string, and storing that verbatim
    would hand ``api_base=""`` to the upstream SDK. Empty and absent mean the
    same thing here, so an empty string is also how a caller clears an endpoint.
    """
    if api_base is None:
        return None
    trimmed = api_base.strip()
    return trimmed or None


def _lookup_name(provider: str) -> str:
    """The stored spelling of a provider name a caller addressed a row by."""
    candidate = provider.strip()
    return PROVIDER_TYPE_ALIASES.get(candidate, candidate)


def _normalize_provider(provider: str) -> str:
    """Resolve a provider name to the any-llm implementation that serves it.

    Raises:
        HostedProviderUnknownProviderError: no any-llm implementation goes by that name.
    """
    implementation = _lookup_name(provider)
    if not implementation:
        raise HostedProviderUnknownProviderError(implementation)
    try:
        LLMProvider(implementation)
    except ValueError as exc:
        raise HostedProviderUnknownProviderError(provider) from exc
    return implementation


def _encrypt_client_args(client_args: dict[str, Any] | None) -> str | None:
    """Encrypt the SDK client extras for storage, as one JSON document.

    Raises:
        HostedProviderSecretStorageError: no usable ``OTARI_SECRET_KEY``.
    """
    if client_args is None:
        return None
    try:
        return encrypt_secret(json.dumps(client_args, sort_keys=True))
    except SecretBoxUnavailableError as exc:
        raise HostedProviderSecretStorageError(str(exc)) from None


def _decrypt_client_args(ciphertext: str | None) -> dict[str, Any] | None:
    """Read the stored SDK client extras back.

    Raises:
        SecretDecryptionError, SecretBoxUnavailableError: no configured key reads them.
    """
    if ciphertext is None:
        return None
    decrypted: dict[str, Any] = json.loads(decrypt_secret(ciphertext))
    return decrypted


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
    ) -> None:
        """Bind the unit of work and the collaborators this service composes.

        Only this domain's own repositories are injected. Rates are read and
        written through ``pricing``, which owns the deployment price list and
        the rules a new version follows. The discovery timeout comes from the
        config held here, because it is a property of the deployment and not of
        the request.
        """
        self.uow = uow
        self.config = config
        self.providers = providers
        self.models = models
        self.pricing = pricing

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    async def list_providers(self, *, skip: int = 0, limit: int = 100) -> HostedProvidersPublic:
        """One page of hosted providers, enabled or not."""
        async with self.uow:
            rows, count = await self.providers.list_page(skip=skip, limit=limit)
            return HostedProvidersPublic(data=[self._public(row) for row in rows], count=count)

    async def create_provider(self, request: HostedProviderCreateRequest) -> HostedProviderPublic:
        """Configure a provider this deployment will serve hosted inference on.

        Everything the provider says it serves on the new key is offered at
        once, so a fresh provider starts with its real catalog rather than an
        empty list the operator retypes by hand. A provider that will not say
        (no listing, unreachable, key refused) yields a provider with no models,
        not a failed create: the credential may still be right for dispatch,
        and the models panel is where the operator adds by hand.

        Raises:
            HostedProviderUnknownProviderError: ``provider`` names no any-llm implementation.
            HostedProviderUnsafeApiBaseError: ``api_base`` is refused by the SSRF gate.
            HostedProviderSecretStorageError: no usable ``OTARI_SECRET_KEY``.
            HostedProviderAlreadyExistsError: that provider is already configured.
        """
        provider = _normalize_provider(request.provider)
        api_base = _normalize_api_base(request.api_base)
        await self._validate_api_base(api_base)
        encrypted = self._encrypt(request.api_key)
        encrypted_client_args = _encrypt_client_args(request.client_args)

        async with self.uow:
            if await self.providers.get_by_provider(provider) is not None:
                raise HostedProviderAlreadyExistsError(provider)
            offered = await self.models.names_for_provider(provider)

        credential = ResolvedHostedProvider(api_key=request.api_key, api_base=api_base, client_args=request.client_args)
        discovery = await self._dial(provider, credential)
        discovered = sorted(self._listed(discovery) - offered)
        defaults = await self.pricing.defaults_for(provider, discovered)

        async with self.uow:
            try:
                row = await self.providers.insert(
                    provider=provider,
                    encrypted_api_key=encrypted,
                    api_key_last4=_last4(request.api_key),
                    api_base=api_base,
                    encrypted_client_args=encrypted_client_args,
                    enabled=request.enabled,
                )
            except HostedProviderConflict as conflict:
                # The pre-check above races the insert, and the unique
                # constraint is what decides. Without this the loser leaves with
                # a 500 rather than the 409 it is.
                raise HostedProviderAlreadyExistsError(conflict.provider) from conflict
            await self._offer(provider, discovered, defaults)
            return HostedProviderPublic.from_row(row, client_args=request.client_args)

    async def update_provider(self, provider: str, request: HostedProviderUpdateRequest) -> HostedProviderPublic:
        """Rotate the key, repoint the base, or toggle a provider.

        A field left unset is left alone, which is what lets the page toggle a
        provider off without re-sending a secret it never received. An empty
        ``api_base`` is not "unset": it clears the endpoint back to the
        provider's default, which is the only way to undo one. ``client_args``
        distinguishes the two by presence: omitted is left alone, null clears,
        and an entry echoed back as the mask keeps the stored value.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedProviderUnsafeApiBaseError: ``api_base`` is refused by the SSRF gate.
            HostedProviderSecretStorageError: no usable ``OTARI_SECRET_KEY``.
        """
        api_base = _normalize_api_base(request.api_base)
        if request.api_base is not None:
            await self._validate_api_base(api_base)
        encrypted = self._encrypt(request.api_key) if request.api_key is not None else None

        async with self.uow:
            row = await self._provider_or_raise(provider)
            if request.api_base is not None:
                row.api_base = api_base
            if request.api_key is not None and encrypted is not None:
                row.encrypted_api_key = encrypted
                row.api_key_last4 = _last4(request.api_key)
            if "client_args" in request.model_fields_set:
                # The form was never shown the real values, so an entry echoed
                # back as the mask keeps what is stored under that name.
                stored = self._client_args(row, quiet=True)
                row.encrypted_client_args = _encrypt_client_args(restore_redacted_values(request.client_args, stored))
            if request.enabled is not None:
                row.enabled = request.enabled
            await self.providers.save(row)
            return self._public(row)

    async def delete_provider(self, provider: str) -> None:
        """Remove a provider and the roster offered on it. The runtime then has no hosted path for it.

        The model rows go with the provider here rather than by a foreign key,
        because the roster is keyed on the name so it can also belong to a
        provider declared in configuration. Pricing history stays, as it does
        when one model is removed; a catalog sweep is what takes it.

        Raises:
            HostedProviderNotFoundError: no such provider.
        """
        async with self.uow:
            row = await self._provider_or_raise(provider)
            await self.models.delete_for_provider(row.provider)
            await self.providers.delete_row(row)

    # ------------------------------------------------------------------
    # Offered models and their prices
    # ------------------------------------------------------------------

    async def list_models(self, provider: str, *, skip: int = 0, limit: int = 500) -> HostedModelsPublic:
        """One page of a provider's offered models, each priced as it currently serves.

        Raises:
            HostedProviderNotFoundError: no such provider.
        """
        async with self.uow:
            row = await self._provider_or_raise(provider)
            rows, count = await self.models.list_for_provider(row.provider, skip=skip, limit=limit)
        prices = await self._current_prices(row.provider, rows)
        return HostedModelsPublic(
            data=[HostedModelPublic.from_row(model, rates=prices.get(model.model)) for model in rows], count=count
        )

    async def add_model(self, provider: str, request: HostedModelCreateRequest) -> HostedModelPublic:
        """Offer a model on a provider, with its custom price when one was given.

        Membership and price land in one transaction: a model offered at a rate
        the commit lost would be served at whatever the ladder answered instead,
        which is the drift this surface exists to prevent. Without an explicit
        rate the offer seeds the community default, and a model nothing prices
        lands disabled.

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
        defaults = {} if priced else await self.pricing.defaults_for(provider, [name])

        async with self.uow:
            await self._provider_or_raise(provider)
            try:
                if priced:
                    [model] = await self.models.create_many([HostedProviderModel(provider=provider, model=name)])
                    await self._write_rate(provider, name, request)
                else:
                    [model] = await self._offer(provider, [name], defaults)
            except HostedModelConflict as conflict:
                raise HostedModelAlreadyOfferedError(provider, conflict.model) from conflict
        prices = await self._current_prices(provider, [model])
        return HostedModelPublic.from_row(model, rates=prices.get(name))

    async def update_model(
        self, provider: str, model_id: uuid.UUID, request: HostedModelUpdateRequest
    ) -> HostedModelPublic:
        """Reprice one offered model, toggle whether it is served, or both.

        A price written here is an operator's rate from then on: the row stops
        tracking the seeded version, so a refresh leaves it alone.

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
                await self._write_rate(provider, model.model, request)
                model.seeded_price_at = None
            if request.enabled is not None:
                model.enabled = request.enabled
            if priced or request.enabled is not None:
                await self.models.save(model)
        prices = await self._current_prices(provider, [model])
        return HostedModelPublic.from_row(model, rates=prices.get(model.model))

    async def remove_model(self, provider: str, model_id: uuid.UUID) -> None:
        """Stop offering a model. Its pricing history stays.

        Deliberately: usage already settled against those rows, and a model
        offered again finds its price where it was left rather than reverting
        to the defaults unannounced. A catalog sweep takes them, which is the
        one operation that is asking for exactly that.

        Raises:
            HostedProviderNotFoundError: no such provider.
            HostedModelNotFoundError: no such model on that provider.
        """
        async with self.uow:
            row = await self._provider_or_raise(provider)
            model = await self._model_or_raise(row.provider, model_id)
            await self.models.delete_row(model)

    async def refresh_models(self, provider: str) -> HostedModelsRefreshPublic:
        """Ask the provider again, offer whatever is newly listed, and move seeded rates.

        Additive only: a model the provider no longer lists is left offered,
        because delisting it is a serving decision the switch owns and an
        upstream hiccup must not unlist a catalog. New models follow the offer
        rule. A seeded rate nobody has changed moves to today's community
        default.

        Raises:
            HostedProviderNotFoundError: no such provider.
        """
        async with self.uow:
            row = await self._provider_or_raise(provider)
            provider = row.provider
            offered = await self.models.names_for_provider(provider)
            seeded = [model.model for model in await self.models.list_seeded(provider)]
            credential = self._credential(row)

        if credential is None:
            return HostedModelsRefreshPublic(added=[], repriced=[], count=len(offered), error=_UNDECRYPTABLE)
        discovery = await self._dial(provider, credential)
        if discovery.error is not None or discovery.discovery_unsupported:
            return HostedModelsRefreshPublic(
                added=[],
                repriced=[],
                count=len(offered),
                error=discovery.error,
                discovery_unsupported=discovery.discovery_unsupported,
            )
        added = sorted(self._listed(discovery) - offered)
        defaults = await self.pricing.defaults_for(provider, [*added, *seeded])

        async with self.uow:
            await self._provider_or_raise(provider)
            repriced = await self._reseed(provider, defaults)
            await self._offer(provider, added, defaults)
        return HostedModelsRefreshPublic(added=added, repriced=repriced, count=len(offered) + len(added))

    async def available_models(self, provider: str) -> HostedAvailableModelsPublic:
        """Ask the provider what it serves on the stored credential.

        The key never leaves the process: it goes into an any-llm client here,
        exactly as the runtime path uses it, and only the model names come back.
        Failure is reported in the body rather than raised, because "the
        upstream would not say" is an answer the form has to render.

        Raises:
            HostedProviderNotFoundError: no such provider.
        """
        async with self.uow:
            row = await self._provider_or_raise(provider)
            provider = row.provider
            credential = self._credential(row)
        if credential is None:
            return HostedAvailableModelsPublic(provider=provider, models=[], error=_UNDECRYPTABLE)
        discovery = await self._dial(provider, credential)
        return HostedAvailableModelsPublic(
            provider=provider,
            models=sorted(model.id for model in discovery.models),
            error=discovery.error,
            discovery_unsupported=discovery.discovery_unsupported,
        )

    # ------------------------------------------------------------------
    # The catalog sweep
    # ------------------------------------------------------------------

    async def refresh_catalog(self, *, apply: bool) -> HostedCatalogRefreshPublic:
        """Re-ask every enabled provider what it lists, then take the rest off the catalog.

        The deployment-wide half of ``refresh_models``. A stored price is what
        lists a model the catalog did not discover, so a price under a provider
        nothing here serves puts a model on every tenant's Models page that no
        request can be served on, and a per-provider refresh cannot reach it.

        Two passes, in this order. Every enabled provider is dialed and whatever
        it newly lists is offered, exactly as a single refresh would, so a model
        that appeared upstream this morning is on the roster before anything is
        judged against it. Then the price list is classified against that
        roster and everything it does not account for is deleted, along with
        the organization overrides above it, except an override held by an
        organization with a live key of its own for the provider.

        ``apply=False`` is the preview a confirm dialog shows: the same dials
        and the same classification, and not one write. It dials rather than
        reading the roster alone, because a model the provider still lists
        would be offered by the apply and kept, and a preview that named it as
        doomed would be a lie. ``repriced`` is empty in a preview.

        **This removes pricing history, which ``remove_model`` deliberately
        keeps.** Removing one model is an edit; a sweep is an operator saying
        the catalog should match this surface, and a model's rates are exactly
        what keeps it on the catalog. Settled usage is unaffected either way.

        A disabled provider is not dialed, and its offered models are kept: the
        switch is a serving decision, so its rows still belong to this surface.

        Raises:
            HostedCatalogEmptyError: no provider offers a model, so there is no
                roster to reconcile against.
        """
        async with self.uow:
            serving: list[tuple[str, ResolvedHostedProvider | None, set[str], list[str]]] = []
            for row in await self.providers.list_all():
                if not row.enabled:
                    continue
                serving.append(
                    (
                        row.provider,
                        self._credential(row),
                        await self.models.names_for_provider(row.provider),
                        [model.model for model in await self.models.list_seeded(row.provider)],
                    )
                )

        # Dialed one provider at a time, each bounded by the discovery timeout.
        # A sweep is an operator pressing a button rather than a request-path
        # read, and the alternative is holding every provider's answer in flight
        # to save an operator a few seconds.
        dialed = [
            await self._dial_for_sweep(provider, credential, offered) for provider, credential, offered, _ in serving
        ]
        defaults: dict[str, dict[str, ModelPricing]] = {}
        if apply:
            for (provider, _, _, seeded), answer in zip(serving, dialed, strict=True):
                defaults[provider] = await self.pricing.defaults_for(provider, [*answer.added, *seeded])

        async with self.uow:
            if apply:
                for (provider, _, _, _), answer in zip(serving, dialed, strict=True):
                    answer.repriced = await self._reseed(provider, defaults[provider])
                    await self._offer(provider, answer.added, defaults[provider])
            offered_keys = await self.models.offered_keys()
            if not apply:
                # What an apply would offer, folded in without writing it, so a
                # model the provider still lists is not previewed as doomed.
                offered_keys |= {f"{answer.provider}:{model}" for answer in dialed for model in answer.added}
            if not offered_keys:
                raise HostedCatalogEmptyError

            verdicts = classify_priced_keys(
                await self.pricing.all_keys(),
                offered=frozenset(offered_keys),
                deployment_instances=frozenset(self.config.providers),
                # Keyed ``<provider>:<tool>`` off ``config.search_tools``, so
                # without this a deployment with a search tool configured loses
                # the rate its own search route reserves against.
                search_providers=frozenset(self.config.search_tool_providers()),
            )
            doomed = [verdict for verdict in verdicts if verdict.verdict == "removed"]
            removed = sorted(verdict.model_key for verdict in doomed)
            override_keys = sorted({verdict.canonical_key for verdict in doomed})
            if apply:
                price_rows, override_rows = await self.pricing.delete_keys(removed, override_keys)
            else:
                price_rows = sum(verdict.versions for verdict in doomed)
                override_rows = await self.pricing.count_doomed_overrides(override_keys)

        return HostedCatalogRefreshPublic(
            providers=dialed,
            removed=removed[:CATALOG_SAMPLE],
            removed_count=len(removed),
            removed_price_rows=price_rows,
            removed_override_rows=override_rows,
            kept=kept_groups(verdicts),
        )

    async def _dial_for_sweep(
        self, provider: str, credential: ResolvedHostedProvider | None, offered: set[str]
    ) -> HostedCatalogProviderRefreshPublic:
        """Ask one provider what it serves, reporting a refusal rather than raising.

        A sweep covers every provider, so one that will not answer must leave
        the others' work standing and say which one it was.
        """
        if credential is None:
            # Distinguished from an upstream refusal so the page does not offer
            # to delete the provider: the credential is unreadable here, not
            # rejected there.
            return HostedCatalogProviderRefreshPublic(
                provider=provider, added=[], error=_UNDECRYPTABLE, credential_unreadable=True
            )
        discovery = await self._dial(provider, credential)
        if discovery.error is not None or discovery.discovery_unsupported:
            return HostedCatalogProviderRefreshPublic(
                provider=provider,
                added=[],
                error=discovery.error,
                discovery_unsupported=discovery.discovery_unsupported,
            )
        return HostedCatalogProviderRefreshPublic(provider=provider, added=sorted(self._listed(discovery) - offered))

    # ------------------------------------------------------------------
    # The runtime path
    # ------------------------------------------------------------------

    async def serveable_models(self) -> dict[str, frozenset[str]]:
        """What this deployment advertises: each serveable provider with the models switched on under it.

        The per-model switch's other half. :meth:`resolve` refuses a row turned
        off, and this is what keeps the catalog from listing it in the first
        place. A model with no row is served but not advertised: the offered
        list is discovery's best effort, and a name the provider grew between
        refreshes is the operator's to offer.

        The same refusals as :meth:`resolve`, because the port requires the
        listing and the resolve to agree exactly: a provider turned off and one
        whose ciphertext no configured key can read are both absent here, as
        they are there. The decryption is bounded by the number of providers
        the deployment serves, which is a handful of rows with one key each.
        """
        async with self.uow:
            serveable = [
                row.provider
                for row in await self.providers.list_all()
                if row.enabled and self._credential(row, quiet=True) is not None
            ]
            served = await self.models.enabled_models_for_providers(serveable)
        return {provider: frozenset(served.get(provider, set())) for provider in serveable}

    async def resolve(self, provider: str, model: str | None = None) -> ResolvedHostedProvider | None:
        """A usable credential for ``provider``, or None.

        None covers every "this build cannot serve it" case, which the caller is
        not asked to tell apart: no row, a provider turned off, a model the
        operator turned off, and a ciphertext no configured key can read. The
        last is logged, because a rotated-away ``OTARI_SECRET_KEY`` degrades one
        provider silently otherwise, and it is the operator's cue to re-enter
        the key.

        The model gate is one-sided on purpose: a row that exists and is
        switched off refuses, while a model with no row serves.
        """
        async with self.uow:
            row = await self.providers.get_by_provider(provider)
            if row is None or not row.enabled:
                return None
            if model is not None:
                offered = await self.models.get_by_model(row.provider, model)
                if offered is not None and not offered.enabled:
                    return None
            return self._credential(row)

    # ------------------------------------------------------------------
    # The two rules
    # ------------------------------------------------------------------

    async def _offer(
        self, provider: str, models: Sequence[str], defaults: Mapping[str, ModelPricing]
    ) -> Sequence[HostedProviderModel]:
        """Record models as offered, each switched on exactly when something prices it.

        A model with no stored rate gets the community default stored as the
        deployment's own version, so a ``require_pricing`` deployment can serve
        it and the catalog can list it; the row remembers the version so a
        refresh can move it until an operator sets a rate. Deliberately not
        gated on ``default_pricing_enabled``: that switch governs the silent
        billing-time fallback, and this is an operator explicitly offering a
        model. Runs inside a write block; ``defaults`` is resolved by the caller
        so the thread hop that produces it stays outside one.

        Raises:
            HostedModelConflict: one of ``models`` is already offered here.
        """
        if not models:
            return []
        now = normalize_effective_at(None)
        keys_by_model = {model: f"{provider}:{model}" for model in models}
        stored = await self.pricing.keys_with_a_price(keys_by_model.values())
        seeded = {
            keys_by_model[model]: default
            for model, default in defaults.items()
            if model in keys_by_model and keys_by_model[model] not in stored
        }
        await self.pricing.store_defaults(seeded, now)
        rows = [
            HostedProviderModel(
                provider=provider,
                model=model,
                enabled=keys_by_model[model] in stored or keys_by_model[model] in seeded,
                seeded_price_at=now if keys_by_model[model] in seeded else None,
            )
            for model in models
        ]
        return await self.models.create_many(rows)

    async def _reseed(self, provider: str, defaults: Mapping[str, ModelPricing]) -> list[str]:
        """Move each seeded rate nobody has changed to today's community default.

        A model's seeded version stops being tracked once a newer version
        exists, whichever path wrote it. A default that disappeared from the
        dataset leaves the last stored rate in place. Returns the models
        repriced. Runs inside a write block.
        """
        seeded = await self.models.list_seeded(provider)
        if not seeded:
            return []
        now = normalize_effective_at(None)
        keys_by_model = {row.model: f"{provider}:{row.model}" for row in seeded}
        latest = await self.pricing.latest_versions(keys_by_model.values())
        repriced: list[str] = []
        moved: dict[str, ModelPricing] = {}
        for row in seeded:
            version = latest.get(keys_by_model[row.model])
            if version is None or not _same_instant(version.effective_at, row.seeded_price_at):
                # An operator priced it since, or its versions are gone. Either
                # way it is not this surface's to move.
                row.seeded_price_at = None
                continue
            default = defaults.get(row.model)
            if default is None or self.pricing.rates_match(version, default):
                continue
            moved[keys_by_model[row.model]] = default
            row.seeded_price_at = now
            repriced.append(row.model)
        # One flush for every row edited above, not one per row: a provider can
        # offer several hundred seeded models.
        await self.models.flush()
        await self.pricing.store_defaults(moved, now)
        return sorted(repriced)

    # ------------------------------------------------------------------
    # Pricing, dialing, credentials
    # ------------------------------------------------------------------

    async def _write_rate(
        self, provider: str, model: str, request: HostedModelCreateRequest | HostedModelUpdateRequest
    ) -> None:
        """Record the deployment's rates for ``provider:model`` as a new pricing version.

        The same store ``POST /pricing`` writes and billing reads, so a price set
        here is the price a request settles at. Only the cache rates the caller
        set travel; an omitted one inherits and an explicit null clears, which
        is that route's rule too. The validator on the request guarantees the
        pair is present when this is reached.
        """
        if request.input_price_per_million is None or request.output_price_per_million is None:
            msg = "a rate is the input/output pair"
            raise ValueError(msg)
        await self.pricing.write_rate(
            f"{provider}:{model}",
            input_price_per_million=request.input_price_per_million,
            output_price_per_million=request.output_price_per_million,
            cache_rates={
                field: getattr(request, field) for field in _CACHE_RATE_FIELDS if field in request.model_fields_set
            },
        )

    async def _current_prices(self, provider: str, rows: Sequence[HostedProviderModel]) -> dict[str, HostedModelRates]:
        """What each offered model is currently served at, keyed by model name.

        Two rungs of the deployment's ladder, in its order: the stored version,
        else the community default. The organization override rung is
        deliberately absent, because this is a deployment-wide surface and a
        negotiated rate is one tenant's. A stored version that is still the
        default this surface seeded reports as a default. A model neither rung
        prices is absent from the map.
        """
        if not rows:
            return {}
        now = normalize_effective_at(None)
        keys = {f"{provider}:{row.model}": row.model for row in rows}
        seeded_at = {row.model: row.seeded_price_at for row in rows}
        versions = await self.pricing.current_versions(keys, now)
        prices: dict[str, HostedModelRates] = {}
        for key, version in versions.items():
            model = keys[key]
            seeded = _same_instant(version.effective_at, seeded_at[model])
            prices[model] = HostedModelRates.from_version(
                version, PRICE_SOURCE_DEFAULT if seeded else PRICE_SOURCE_DEPLOYMENT
            )
        if default_pricing_enabled():
            missing = [model for model in keys.values() if model not in prices]
            for model, default in (await self.pricing.defaults_for(provider, missing)).items():
                prices[model] = HostedModelRates.from_version(default, PRICE_SOURCE_DEFAULT)
        return prices

    async def _dial(self, provider: str, credential: ResolvedHostedProvider) -> ProviderDiscovery:
        """Ask the provider what it serves on ``credential``. Reports failure rather than raising."""
        return await test_provider_credentials(
            provider,
            api_key=credential.api_key,
            api_base=credential.api_base,
            client_args=credential.client_args,
            timeout=self.config.model_discovery_timeout_seconds,
        )

    @staticmethod
    def _listed(discovery: ProviderDiscovery) -> set[str]:
        """The model names a dial listed, or none when the provider would not say."""
        if discovery.error is not None or discovery.discovery_unsupported:
            return set()
        return {model.id for model in discovery.models}

    @staticmethod
    def _credential(row: HostedProvider, *, quiet: bool = False) -> ResolvedHostedProvider | None:
        """Decrypt a row's key, or None when no configured key can read it.

        Logged unless ``quiet``, because a rotated-away ``OTARI_SECRET_KEY``
        degrades one provider silently otherwise; a listing passes ``quiet`` so
        it does not repeat the warning per page load.
        """
        try:
            api_key = decrypt_secret(row.encrypted_api_key)
            client_args = _decrypt_client_args(row.encrypted_client_args)
        except (SecretDecryptionError, SecretBoxUnavailableError):
            if not quiet:
                logger.error(
                    "Hosted provider %r could not be decrypted; treating it as unserved. "
                    "Was OTARI_SECRET_KEY rotated? The credential must be re-entered.",
                    row.provider,
                )
            return None
        return ResolvedHostedProvider(api_key=api_key, api_base=row.api_base, client_args=client_args)

    @staticmethod
    def _client_args(row: HostedProvider, *, quiet: bool = False) -> dict[str, Any] | None:
        """The stored SDK client extras, or None when there are none or no configured key reads them.

        The listing cannot tell the two apart and does not need to: a row whose
        extras will not decrypt has a key that will not either, and the page
        shows that row by its tail for the operator to re-enter.
        """
        try:
            return _decrypt_client_args(row.encrypted_client_args)
        except (SecretDecryptionError, SecretBoxUnavailableError):
            if not quiet:
                logger.error("Hosted provider %r client arguments could not be decrypted", row.provider)
            return None

    def _public(self, row: HostedProvider) -> HostedProviderPublic:
        """Render a stored row for the API, decrypting its extras so the schema can mask them."""
        return HostedProviderPublic.from_row(row, client_args=self._client_args(row, quiet=True))

    @staticmethod
    def _encrypt(api_key: str) -> str:
        """Encrypt a key for storage, reporting an unusable secret box as a 400.

        Without ``OTARI_SECRET_KEY`` there is nowhere safe to put the credential,
        and storing it in the clear is not the fallback.
        """
        try:
            return encrypt_secret(api_key)
        except SecretBoxUnavailableError as exc:
            raise HostedProviderSecretStorageError(str(exc)) from None

    @staticmethod
    async def _validate_api_base(api_base: str | None) -> None:
        """Refuse an ``api_base`` the SSRF gate rejects.

        A no-op in the default allow-all state, exactly as on the organization
        provider-key write path: the gate turns on only when an operator sets
        ``OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS=false``.
        """
        if not api_base:
            return
        try:
            await validate_provider_api_base(api_base)
        except UnsafeURLError as exc:
            raise HostedProviderUnsafeApiBaseError(str(exc)) from None

    async def _provider_or_raise(self, provider: str) -> HostedProvider:
        """Load a provider by the name a caller addressed it with, or refuse with the surface's 404.

        Raises:
            HostedProviderNotFoundError: no such provider.
        """
        row = await self.providers.get_by_provider(_lookup_name(provider))
        if row is None:
            raise HostedProviderNotFoundError(provider)
        return row

    async def _model_or_raise(self, provider: str, model_id: uuid.UUID) -> HostedProviderModel:
        """Load one offered model, scoped to its provider.

        Raises:
            HostedModelNotFoundError: no such model on that provider.
        """
        model = await self.models.get_in_provider(model_id, provider)
        if model is None:
            raise HostedModelNotFoundError(model_id)
        return model


def _same_instant(version_at: datetime | None, seeded_at: datetime | None) -> bool:
    """Whether a version's timestamp is the one the row recorded when it seeded it.

    Normalized on both sides because ``model_pricing`` reads back naive on
    SQLite while the row's own column is timezone-aware everywhere, and Python
    answers False to any comparison across that line.
    """
    if seeded_at is None or version_at is None:
        return False
    return normalize_effective_at(version_at) == normalize_effective_at(seeded_at)


__all__ = ["HostedProviderService", "ResolvedHostedProvider"]
