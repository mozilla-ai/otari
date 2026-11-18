"""The runtime path: what the deployment advertises, and the credential that serves a candidate."""

import uuid
from typing import Any

from any_llm import LLMProvider

from gateway.core.config import GatewayConfig, region_instance
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.ports.model_provider_port import NO_ORGANIZATION, HostedCredential, HostedModels, ModelProviderPort
from gateway.repositories.providers import HostedProviderModelRepository, HostedProviderVendorRepository
from gateway.services.provider_kwargs import (
    ResolvedProvider,
    hosted_lookup,
    keyless_placeholder_api_key,
    missing_declared_credential,
)
from gateway.services.providers._country_gate import CountryGate, ensure_hosted_country
from gateway.services.providers._hosted_credentials import ResolvedHostedProvider
from gateway.services.providers._hosted_name_index import hosted_provider_named, remember_served_location
from gateway.services.providers._hosted_sources import HostedSources
from gateway.services.providers._hosted_vendors import vendor_of
from gateway.services.providers._workspace_hosted_access import (
    ensure_workspace_may_use_hosted,
    narrow_to_workspace_locations,
    workspace_allowed_data_locations,
)


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


async def hosted_credential_for(
    port: ModelProviderPort,
    *,
    organization_id: uuid.UUID | None,
    workspace_id: uuid.UUID | None,
    provider: str,
    model: str | None,
) -> HostedCredential | None:
    """The deployment-owned credential the port resolves for a request, a request with no organization included.

    The workspace's own rule is applied before the port is asked, because the
    port decides for the organization alone, and its data locations narrow the
    regions the port answers with afterwards.

    Raises:
        HostedAccessDeniedError: the workspace or the organization may not use the upstream.
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
    return None if credential is None else narrow_to_workspace_locations(credential, allowed, workspace_id, provider)


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
) -> ResolvedProvider | None:
    """``resolved`` served on the deployment's own credential where nothing declared serves it, else ``None``.

    The last rung of the credential ladder: a candidate a declared credential
    serves is never asked about. ``None`` also when the port has nothing for
    the candidate, or answers for another provider where only its own name may.
    ``any_model`` asks for the provider rather than for ``resolved.model``, for
    a call that names no model (a batch's status, say). ``country_gate`` holds
    the request to the provider's own ``supported_countries``, served by the
    first region whose countries admit it.

    Raises:
        HostedAccessDeniedError: the organization may not use the upstream.
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
    )
    if credential is None or not hosted_lookup(resolved.provider).accepts(
        credential.response_provider, resolved.instance
    ):
        return None
    credential = ensure_hosted_country(country_gate, credential)
    return dispatch_on_hosted_credential(credential, model=resolved.model, alias=resolved.alias)


class HostedProviderRuntime:
    """Answer the two questions the dispatch path and the catalog ask."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        sources: HostedSources,
        models: HostedProviderModelRepository,
        vendors: HostedProviderVendorRepository,
    ) -> None:
        self.uow = uow
        self.sources = sources
        self.models = models
        self.vendors = vendors

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
            found = await self.sources.get_serving(name)
            if found is None:
                return []
            source, pinned = found
            if not source.enabled or source.withheld_from(organization_id):
                return []
            if model is not None:
                offered = await self.models.get_by_model(source.name, model)
                if offered is None and not source.serves_unlisted_models:
                    return []
                if offered is not None and not offered.enabled:
                    return []
                disabled = await self.vendors.disabled_for(source.name)
                if disabled and vendor_of(source.implementation, model) in disabled:
                    return []
            credentials = source.region_credentials(slug=pinned)
        # A provider of a region's whole name answers that selector, so the
        # region would settle under, and be priced as, a provider it is not.
        shadowed = [
            each
            for each in credentials
            if each.region and hosted_provider_named(region_instance(source.name, each.region))
        ]
        for each in shadowed:
            logger.warning(
                "Region %r of hosted provider %r is not served: a provider is named %r",
                each.region,
                source.name,
                region_instance(source.name, each.region),
            )
        return [each for each in credentials if each not in shadowed]
