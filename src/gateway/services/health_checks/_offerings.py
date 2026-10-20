"""What a run checks: every model the deployment's hosted providers serve, with the credential to reach it."""

from typing import Any

from gateway.core.config import GatewayConfig
from gateway.services.catalog import metadata_entry
from gateway.services.health_checks._probe import HealthCheckOffering, is_chat_model
from gateway.services.model_catalog_service import cached_models_dev_catalog
from gateway.services.provider_kwargs import keyless_placeholder_api_key
from gateway.services.providers import HostedProviderService, ResolvedHostedProvider, hosted_implementation


def _call_kwargs(credential: ResolvedHostedProvider) -> dict[str, Any]:
    """The connection arguments dispatch sends a hosted provider, so a check reaches it as a request would."""
    api_key = credential.api_key or keyless_placeholder_api_key(credential.api_base, None)
    kwargs: dict[str, Any] = {} if api_key is None else {"api_key": api_key}
    if credential.api_base is not None:
        kwargs["api_base"] = credential.api_base
    if credential.client_args:
        kwargs["client_args"] = dict(credential.client_args)
    return kwargs


async def list_hosted_offerings(hosted: HostedProviderService, config: GatewayConfig) -> list[HealthCheckOffering]:
    """Each model switched on under an enabled hosted provider whose credential reads, in selector order.

    The deployment's own view, so a provider restricted to one organization is
    checked too. Opens its own units of work, so call it before any probe.
    """
    catalog = cached_models_dev_catalog(config)
    offerings: list[HealthCheckOffering] = []
    for name, models in sorted((await hosted.serveable_models(organization_id=None)).items()):
        credential = await hosted.resolve(name, organization_id=None)
        if credential is None:
            continue
        implementation = credential.implementation or hosted_implementation(name) or name
        kwargs = _call_kwargs(credential)
        for model in sorted(models):
            metadata = metadata_entry(catalog, implementation, model)
            offerings.append(
                HealthCheckOffering(
                    selector=f"{name}:{model}",
                    provider=name,
                    reasoning=bool(metadata and metadata.reasoning),
                    chat=is_chat_model(model, metadata.output_modalities if metadata else []),
                    implementation=implementation,
                    model=model,
                    call_kwargs=kwargs,
                )
            )
    return offerings
