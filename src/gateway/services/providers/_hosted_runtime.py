"""The runtime path: what the deployment advertises, and the credential that serves a candidate."""

import uuid
from collections.abc import Collection
from dataclasses import replace
from typing import Any, Protocol

from any_llm import LLMProvider

from gateway.core.config import GatewayConfig, region_instance
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.ports.model_provider_port import (
    NO_ORGANIZATION,
    HostedAccessDeniedError,
    HostedCredential,
    HostedModels,
    ModelProviderPort,
)
from gateway.repositories.providers import HostedProviderModelRepository, HostedProviderVendorRepository
from gateway.services.provider_kwargs import (
    ResolvedProvider,
    hosted_lookup,
    keyless_placeholder_api_key,
    missing_declared_credential,
)
from gateway.services.providers._country_gate import ensure_hosted_country
from gateway.services.providers._hosted_credentials import ResolvedHostedProvider
from gateway.services.providers._hosted_name_index import hosted_provider_named, remember_served_location
from gateway.services.providers._hosted_sources import HostedSnapshot, HostedSource, HostedSources
from gateway.services.providers._hosted_vendors import vendor_of
from gateway.services.providers._workspace_hosted_access import (
    ensure_workspace_may_use_hosted,
    narrow_to_workspace_locations,
    workspace_allowed_data_locations,
)
from gateway.types.country_gate import CountryGate


def organization_to_serve(organization_id: uuid.UUID | None) -> uuid.UUID:
    """The organization the hosted rung asks the port about for a request.

    A request with no organization (its workspace belongs to none) is served on
    what is restricted to no organization, so it is asked as one no restriction
    names; billing then has no organization to charge, and charges none.
    """
    return organization_id if organization_id is not None else NO_ORGANIZATION


async def hosted_models_for(port: ModelProviderPort, *, organization_id: uuid.UUID | None) -> HostedModels:
    """What the port advertises to a request's organization, a request with no organization included."""
    return await port.get_hosted_models(organization_id=organization_to_serve(organization_id))


class HostedLocalityRefusedError(HostedAccessDeniedError):
    """No region of the hosted provider runs where the request's routing policy allows."""


def narrow_to_localities(
    credential: HostedCredential, allowed: Collection[str], workspace_id: uuid.UUID | None, provider: str
) -> HostedCredential:
    """``credential`` and its alternates kept to those whose declared routing locality is in ``allowed``.

    An undeclared locality never matches: a routing policy with ``eligibility``
    is served only where the operator said the inference runs.

    Raises:
        HostedLocalityRefusedError: no region of the provider declares an allowed locality.
    """
    kept = [candidate for candidate in credential.candidates() if candidate.routing_locality in allowed]
    if not kept:
        raise HostedLocalityRefusedError(
            f"No region of hosted provider '{provider}' runs where this routing policy allows", workspace_id
        )
    return replace(kept[0], alternates=tuple(kept[1:]))


async def hosted_credential_for(
    port: ModelProviderPort,
    *,
    organization_id: uuid.UUID | None,
    workspace_id: uuid.UUID | None,
    provider: str,
    model: str | None,
    allowed_localities: Collection[str] | None = None,
) -> HostedCredential | None:
    """The deployment-owned credential the port resolves for a request, a request with no organization included.

    The workspace's own rule is applied before the port is asked, because the
    port decides for the organization alone, and its data locations narrow the
    regions the port answers with afterwards. ``allowed_localities``, given for
    a routing policy with ``eligibility``, narrows them further to the regions
    whose provider declares one of those localities; None leaves them as they are.

    Raises:
        HostedAccessDeniedError: the workspace or the organization may not use the upstream.
        HostedLocalityRefusedError: no region runs where ``allowed_localities`` allows.
    """
    # Read before the port is awaited, so the rule and the locations come from one cache snapshot.
    ensure_workspace_may_use_hosted(workspace_id, provider, model)
    allowed = workspace_allowed_data_locations(workspace_id)
    credential = await port.resolve_hosted_credential(
        organization_id=organization_to_serve(organization_id),
        workspace_id=workspace_id,
        provider=provider,
        model=model,
    )
    if credential is None:
        return None
    credential = narrow_to_workspace_locations(credential, allowed, workspace_id, provider)
    if allowed_localities is not None:
        credential = narrow_to_localities(credential, allowed_localities, workspace_id, provider)
    return credential


class UnknownHostedUpstreamError(ValueError):
    """The port named an upstream any-llm does not implement, so nothing can be dispatched."""


def dispatch_on_hosted_credential(
    credential: HostedCredential, *, model: str, alias: str | None = None
) -> ResolvedProvider:
    """The dispatch a hosted credential serves: re-keyed onto its ``response_provider`` and deployment-paid.

    Raises:
        UnknownHostedUpstreamError: the credential names an upstream any-llm does not implement.
    """
    remember_served_location(credential.response_provider, credential.data_location)
    try:
        upstream = LLMProvider(credential.upstream)
    except ValueError as exc:
        raise UnknownHostedUpstreamError(credential.upstream) from exc
    # A keyless backend's credential carries no key, so it takes the placeholder
    # a keyless ``config.yml`` instance takes.
    api_key = credential.api_key or keyless_placeholder_api_key(credential.api_base, None)
    kwargs: dict[str, Any] = {} if api_key is None else {"api_key": api_key}
    if credential.api_base is not None:
        kwargs["api_base"] = credential.api_base
    if credential.client_args:
        # The key a ``config.yml`` instance's extras travel under, so Bedrock's
        # region reaches the SDK client as it would from the file.
        kwargs["client_args"] = dict(credential.client_args)
    return ResolvedProvider(
        instance=credential.response_provider,
        provider=upstream,
        model=model,
        kwargs=kwargs,
        alias=alias,
        deployment_paid=True,
    )


async def serve_on_hosted_provider(
    config: GatewayConfig,
    port: ModelProviderPort,
    resolved: ResolvedProvider,
    *,
    organization_id: uuid.UUID | None,
    workspace_id: uuid.UUID | None,
    any_model: bool = False,
    country_gate: CountryGate | None = None,
    allowed_localities: Collection[str] | None = None,
) -> ResolvedProvider | None:
    """``resolved`` served on the deployment's own credential where nothing declared serves it, else ``None``.

    The last rung of the credential ladder: a candidate a declared credential
    serves is never asked about. ``None`` also when the port has nothing for
    the candidate, or answers for another provider where only its own name may.
    ``any_model`` asks for the provider rather than for ``resolved.model``, for
    a call that names no model (a batch's status, say). ``country_gate`` holds
    the request to the provider's own ``supported_countries``, served by the
    first region whose countries admit it. ``allowed_localities`` holds it to
    the regions whose provider declares one of those routing localities, for a
    routing policy with ``eligibility``.

    Raises:
        HostedAccessDeniedError: the organization may not use the upstream.
        HostedLocalityRefusedError: no region runs where ``allowed_localities`` allows.
        CountryNotServedError: the provider does not serve the request's country.
        UnknownHostedUpstreamError: the port named an upstream any-llm does not implement.
    """
    if missing_declared_credential(config, resolved.instance, resolved.provider, resolved.kwargs) is None:
        return None
    credential = await hosted_credential_for(
        port,
        organization_id=organization_id,
        workspace_id=workspace_id,
        provider=resolved.instance,
        model=None if any_model else resolved.model,
        allowed_localities=allowed_localities,
    )
    if credential is None or not hosted_lookup(resolved.provider).accepts(
        credential.response_provider, resolved.instance
    ):
        return None
    credential = await ensure_hosted_country(country_gate, credential)
    return dispatch_on_hosted_credential(credential, model=resolved.model, alias=resolved.alias)


class HostedServingBuilder(Protocol):
    """Build a :class:`HostedServing` on the caller's unit of work; the composition root supplies it."""

    def __call__(self, *, read_all: bool) -> "HostedServing": ...


class HostedProviderRuntime:
    """Answer the two questions the dispatch path and the catalog ask."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        sources: HostedSources,
        models: HostedProviderModelRepository,
        vendors: HostedProviderVendorRepository,
        serving: HostedServingBuilder,
    ) -> None:
        self.uow = uow
        self.sources = sources
        self.models = models
        self.vendors = vendors
        self.build_serving = serving

    async def serveable_models(self, *, organization_id: uuid.UUID | None) -> dict[str, frozenset[str]]:
        """Each provider serveable to ``organization_id``, with the models switched on under it.

        The same refusals as :meth:`resolve`, so the listing and the resolve
        agree: a provider switched off, one restricted to another organization
        and one whose ciphertext no configured key reads are all absent, and so
        is every model of a vendor its provider has switched off.
        """
        async with self.uow:
            serveable = {
                source.name: source.implementation
                for source in await self.sources.all()
                if source.enabled
                and not source.withheld_from(organization_id)
                and source.credential(quiet=True) is not None
            }
            served = await self.models.enabled_models_for_providers(serveable)
            disabled = await self.vendors.disabled_for_providers(serveable)
        return {
            name: frozenset(
                model
                for model in served.get(name, set())
                if name not in disabled or vendor_of(implementation, model) not in disabled[name]
            )
            for name, implementation in serveable.items()
        }

    def serving(self) -> "HostedServing":
        """A resolver that reads the hosted providers once and answers each later resolve from that read.

        For one request: what it read is never read again, so a change
        committed after its first resolve is not seen by the next.
        """
        return self.build_serving(read_all=True)

    async def resolve(
        self, name: str, model: str | None = None, *, organization_id: uuid.UUID | None
    ) -> list[ResolvedHostedProvider]:
        """The usable connections of the hosted provider ``name`` on behalf of ``organization_id``, in order.

        Read afresh, and for ``name`` alone; see :meth:`HostedServing.resolve` for the rules.
        """
        return await self.build_serving(read_all=False).resolve(name, model, organization_id=organization_id)


class HostedServing:
    """Resolve hosted providers for one request, reading each table at most once where it can.

    The first resolve reads every hosted provider and its regions in one
    transaction, so each candidate of a routing policy after it costs no read
    of either. The vendors switched off are read for every provider the first
    time a model is checked, and a model's row once per provider and model.
    The rows are detached, because a rollback later in the request would
    otherwise expire them. Where there are more providers than one listing
    holds, or ``read_all`` is false, each name is read on its own instead.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        sources: HostedSources,
        models: HostedProviderModelRepository,
        vendors: HostedProviderVendorRepository,
        read_all: bool = True,
    ) -> None:
        self.uow = uow
        self.sources = sources
        self.models = models
        self.vendors = vendors
        self._read_all = read_all
        self._snapshot: HostedSnapshot | None = None
        self._snapshot_read = False
        # provider -> the vendors switched off on it
        self._disabled: dict[str, set[str]] = {}
        self._disabled_read = False
        # (provider, model) -> whether its row is switched on, or None for no row
        self._offered: dict[tuple[str, str], bool | None] = {}

    async def resolve(
        self, name: str, model: str | None = None, *, organization_id: uuid.UUID | None
    ) -> list[ResolvedHostedProvider]:
        """The usable connections of the hosted provider ``name`` on behalf of ``organization_id``, in order.

        One for a provider with no regions; otherwise each enabled region, or
        only the one ``name`` pins (``tokenhub@sg``). Empty covers every case
        the caller is not asked to tell apart: no provider or region, one
        switched off or restricted to another organization, a model the
        provider does not offer switched on, and a ciphertext no configured key
        reads. A model with no row is refused unless the
        provider serves unlisted models, so otherwise the operator's roster is
        the whole of what the deployment's credential serves. A model of a
        vendor the provider has switched off is refused whatever its own row says.
        """
        async with self.uow:
            found = await self._get_serving(name)
            if found is None:
                return []
            source, pinned = found
            if not source.enabled or source.withheld_from(organization_id):
                return []
            if model is not None and not await self._offers(source, model):
                return []
            credentials = source.region_credentials(slug=pinned)
        # A provider of a region's whole name answers that selector, so the
        # region would settle under, and be priced as, a provider it is not.
        # Asked of the snapshot where there is one, so every candidate agrees on which names exist.
        named = hosted_provider_named if self._snapshot is None else self._snapshot.provider_named
        shadowed = [each for each in credentials if each.region and named(region_instance(source.name, each.region))]
        for each in shadowed:
            logger.warning(
                "Region %r of hosted provider %r is not served: a provider is named %r",
                each.region,
                source.name,
                region_instance(source.name, each.region),
            )
        return [each for each in credentials if each not in shadowed]

    async def _get_serving(self, name: str) -> tuple[HostedSource, str | None] | None:
        if self._read_all and not self._snapshot_read:
            self._snapshot = await self.sources.snapshot()
            self._snapshot_read = True
        if self._snapshot is None:
            return await self.sources.get_serving(name)
        return self._snapshot.get_serving(name)

    async def _offers(self, source: HostedSource, model: str) -> bool:
        """Whether ``source`` serves ``model``, by its roster row and by its vendor."""
        key = (source.name, model)
        if key not in self._offered:
            offered = await self.models.get_by_model(source.name, model)
            self._offered[key] = None if offered is None else offered.enabled
        enabled = self._offered[key]
        if enabled is None and not source.serves_unlisted_models:
            return False
        if enabled is False:
            return False
        disabled = await self._disabled_vendors(source.name)
        return not (disabled and vendor_of(source.implementation, model) in disabled)

    async def _disabled_vendors(self, provider: str) -> set[str]:
        if not self._disabled_read and self._snapshot is not None:
            names = list(self._snapshot.by_name)
            found = await self.vendors.disabled_for_providers(names)
            self._disabled = {name: found.get(name, set()) for name in names}
            self._disabled_read = True
        if provider not in self._disabled:
            self._disabled[provider] = await self.vendors.disabled_for(provider)
        return self._disabled[provider]
