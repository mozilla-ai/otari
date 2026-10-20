"""The runtime path: what the deployment advertises, and the credential that serves a candidate."""

import uuid
from typing import Any

from any_llm import LLMProvider

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.model_provider_port import NO_ORGANIZATION, HostedCredential, HostedModels, ModelProviderPort
from gateway.repositories.providers import HostedProviderModelRepository
from gateway.services.provider_kwargs import (
    ResolvedProvider,
    hosted_lookup,
    keyless_placeholder_api_key,
    missing_declared_credential,
)
from gateway.services.providers._hosted_credentials import ResolvedHostedProvider
from gateway.services.providers._hosted_sources import HostedSources


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
    """The deployment-owned credential the port resolves for a request, a request with no organization included."""
    return await port.resolve_hosted_credential(
        organization_id=organization_to_serve(organization_id),
        workspace_id=workspace_id,
        provider=provider,
        model=model,
    )


class UnknownHostedUpstreamError(ValueError):
    """The port named an upstream any-llm does not implement, so nothing can be dispatched."""


def dispatch_on_hosted_credential(
    credential: HostedCredential, *, model: str, alias: str | None = None
) -> ResolvedProvider:
    """The dispatch a hosted credential serves: re-keyed onto its ``response_provider`` and deployment-paid.

    Raises:
        UnknownHostedUpstreamError: the credential names an upstream any-llm does not implement.
    """
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
) -> ResolvedProvider | None:
    """``resolved`` served on the deployment's own credential where nothing declared serves it, else ``None``.

    The last rung of the credential ladder: a candidate a declared credential
    serves is never asked about. ``None`` also when the port has nothing for
    the candidate, or answers for another provider where only its own name may.
    ``any_model`` asks for the provider rather than for ``resolved.model``, for
    a call that names no model (a batch's status, say).

    Raises:
        HostedAccessDeniedError: the organization may not use the upstream.
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
    return dispatch_on_hosted_credential(credential, model=resolved.model, alias=resolved.alias)


class HostedProviderRuntime:
    """Answer the two questions the dispatch path and the catalog ask."""

    def __init__(self, uow: UnitOfWork, *, sources: HostedSources, models: HostedProviderModelRepository) -> None:
        self.uow = uow
        self.sources = sources
        self.models = models

    async def serveable_models(self, *, organization_id: uuid.UUID | None) -> dict[str, frozenset[str]]:
        """Each provider serveable to ``organization_id``, with the models switched on under it.

        The same refusals as :meth:`resolve`, so the listing and the resolve
        agree: a provider switched off, one restricted to another organization
        and one whose ciphertext no configured key reads are all absent.
        """
        async with self.uow:
            serveable = [
                source.name
                for source in await self.sources.all()
                if source.enabled
                and not source.withheld_from(organization_id)
                and source.credential(quiet=True) is not None
            ]
            served = await self.models.enabled_models_for_providers(serveable)
        return {name: frozenset(served.get(name, set())) for name in serveable}

    async def resolve(
        self, name: str, model: str | None = None, *, organization_id: uuid.UUID | None
    ) -> ResolvedHostedProvider | None:
        """A usable credential for the hosted provider ``name`` on behalf of ``organization_id``, or None.

        None covers every case the caller is not asked to tell apart: no
        provider, one switched off or restricted to another organization, a
        model the provider does not offer switched on, and a ciphertext no
        configured key reads. A model with no row is refused unless the
        provider serves unlisted models, so otherwise the operator's roster is
        the whole of what the deployment's credential serves.
        """
        async with self.uow:
            source = await self.sources.get(name)
            if source is None or not source.enabled or source.withheld_from(organization_id):
                return None
            if model is not None:
                offered = await self.models.get_by_model(source.name, model)
                if offered is None and not source.serves_unlisted_models:
                    return None
                if offered is not None and not offered.enabled:
                    return None
            return source.credential()
