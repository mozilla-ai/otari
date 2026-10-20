"""Provider metadata for the dashboard.

The ``/api/v1/providers`` endpoint reports static, network-free metadata for every
configured provider, and ``/api/v1/providers/health`` their reachability. They
describe the gateway's own configuration, so their router is operator-gated;
standalone-mode only (it is not mounted in hybrid).

``/api/v1/providers/catalog`` is the exception, on ``catalog_router``: it lists the
providers any-llm knows, which is a property of the build rather than of this
deployment, and the organization provider-key form needs it to offer a BYO provider
at all.
"""

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from gateway.api.deps import (
    HostedProviderServiceDep,
    get_config,
    require_deployment_operator,
    verify_catalog_reader,
)
from gateway.core.config import GatewayConfig
from gateway.services.model_discovery_service import (
    background_discovery_enabled,
)
from gateway.services.provider_health_service import ProviderHealth, check_all_provider_health
from gateway.services.provider_metadata_service import (
    KnownProvider,
    KnownProviderSummary,
    ProviderInfo,
    known_provider_detail,
    list_known_provider_summaries,
    list_provider_info,
)

router = APIRouter(
    tags=["providers"],
    dependencies=[Depends(require_deployment_operator)],
)
# The registry reads, which name providers any-llm knows rather than anything this
# deployment configured. A tenant reaches them: the organization provider-key form
# is the picker's other caller, and it is owners and admins who fill it, never an
# operator. Gated like the other catalog reads so admitting a session is spelled at
# the router (see ``deps.verify_catalog_reader``).
catalog_router = APIRouter(
    tags=["providers"],
    dependencies=[Depends(verify_catalog_reader)],
)


class ProviderCapabilitiesSchema(BaseModel):
    """Curated capability flags for a provider."""

    streaming: bool
    reasoning: bool
    vision: bool
    pdf: bool
    embeddings: bool
    image_generation: bool
    audio: bool
    rerank: bool
    responses_api: bool
    moderation: bool
    list_models: bool


class ProviderInfoSchema(BaseModel):
    """Static, network-free metadata for one configured provider instance."""

    instance: str = Field(description="Configured provider key (may differ from the type).")
    provider_type: str = Field(description="Underlying any-llm provider type.")
    name: str = Field(description="Human-friendly provider name.")
    doc_url: str | None = None
    description: str | None = None
    env_key: str | None = Field(default=None, description="Env var the credential is read from.")
    pricing_urls: list[str] = Field(default_factory=list)
    capabilities: ProviderCapabilitiesSchema
    source: Literal["config"] = Field(
        default="config",
        description="Where the instance is defined: 'config' for config.yml (or OTARI_CONFIG_YAML).",
    )


class ProvidersResponse(BaseModel):
    """Metadata for every configured provider."""

    providers: list[ProviderInfoSchema]


def _to_schema(info: ProviderInfo) -> ProviderInfoSchema:
    caps = info.capabilities
    return ProviderInfoSchema(
        source=info.source,
        instance=info.instance,
        provider_type=info.provider_type,
        name=info.name,
        doc_url=info.doc_url,
        description=info.description,
        env_key=info.env_key,
        pricing_urls=info.pricing_urls,
        capabilities=ProviderCapabilitiesSchema(
            streaming=caps.streaming,
            reasoning=caps.reasoning,
            vision=caps.vision,
            pdf=caps.pdf,
            embeddings=caps.embeddings,
            image_generation=caps.image_generation,
            audio=caps.audio,
            rerank=caps.rerank,
            responses_api=caps.responses_api,
            moderation=caps.moderation,
            list_models=caps.list_models,
        ),
    )


@router.get("/providers")
async def list_providers(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ProvidersResponse:
    """List static metadata for every configured provider.

    Operator-facing: reports each provider's capabilities, documentation and
    pricing links, and display name from the bundled any-llm and models.dev
    datasets. No provider is contacted, so this is cheap and always available.
    """
    return ProvidersResponse(providers=[_to_schema(info) for info in list_provider_info(config)])


class KnownProviderSummarySchema(BaseModel):
    """A provider offered in the add-provider picker: id and display name only."""

    id: str = Field(description="any-llm provider id, used as the default instance name.")
    name: str = Field(description="Human-friendly display name.")


class KnownProviderSchema(BaseModel):
    """A selected provider's autofill hints for the add-provider form."""

    id: str = Field(description="any-llm provider id, used as the default instance name.")
    name: str = Field(description="Human-friendly display name.")
    env_key: str | None = Field(default=None, description="Env var the SDK reads this provider's key from.")
    default_api_base: str | None = Field(default=None, description="Built-in endpoint; blank means the SDK's default.")
    requires_api_key: bool = Field(
        description=(
            "False for providers that need no key: keyless local backends (Ollama, llama.cpp) and ambient "
            "cloud credentials (Bedrock, Vertex AI)."
        )
    )
    env_key_present: bool = Field(
        default=False,
        description=(
            "True when env_key is set on the server. Informational: a stored provider never reads it, "
            "so a key the provider requires must still be supplied."
        ),
    )


def _to_summary_schema(summary: KnownProviderSummary) -> KnownProviderSummarySchema:
    return KnownProviderSummarySchema(id=summary.id, name=summary.name)


def _to_known_schema(provider: KnownProvider) -> KnownProviderSchema:
    return KnownProviderSchema(
        id=provider.id,
        name=provider.name,
        env_key=provider.env_key,
        default_api_base=provider.default_api_base,
        requires_api_key=provider.requires_api_key,
        env_key_present=provider.env_key_present,
    )


@catalog_router.get("/providers/catalog")
async def provider_catalog() -> list[KnownProviderSummarySchema]:
    """List every known provider for the add-provider picker: id and name only.

    Lightweight by design so the picker never lags: provider ids come from the
    any-llm registry and names from the models.dev catalog, so no
    provider SDK is imported. The autofill hints for a chosen provider come from
    GET /api/v1/providers/catalog/{provider_id}, which imports only that one SDK.
    """
    return [_to_summary_schema(summary) for summary in list_known_provider_summaries()]


@catalog_router.get("/providers/catalog/{provider_id}")
async def provider_catalog_detail(provider_id: str) -> KnownProviderSchema:
    """Autofill hints for one provider the add-provider form has selected.

    Imports only the selected provider's any-llm module (not the whole catalog)
    to report its credential env var, default endpoint, whether a key is required,
    and whether that env var is already set on the server. Returns 404 for an
    unknown provider id.

    The SDK import is offloaded to a worker thread: the first fetch for a given
    provider imports that provider's module, which would otherwise block the event
    loop (and thus every concurrent request) for the import's duration.
    """
    detail = await asyncio.to_thread(known_provider_detail, provider_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown provider: {provider_id}")
    return _to_known_schema(detail)


class ProviderHealthSchema(BaseModel):
    """One provider instance's reachability, from the model-discovery test path."""

    instance: str
    ok: bool = Field(description="True when the provider's credentials could list models.")
    model_count: int = Field(description="Number of models the last successful listing returned.")
    error: str | None = Field(default=None, description="Sanitized provider error when unreachable.")
    checked_at: str | None = Field(
        default=None,
        description="ISO 8601 wall-clock time the provider's reachability was last checked (null if never).",
    )
    discovery_unsupported: bool = Field(
        default=False,
        description=(
            "True when the check failed only because this backend serves no model-listing endpoint. "
            "The provider may still handle requests; only model discovery is unavailable."
        ),
    )


class ProviderHealthResponse(BaseModel):
    """Provider connectivity across the whole gateway, for the health monitor.

    Carries per-provider results plus the ``healthy`` / ``total`` counts and the
    most recent ``checked_at`` so the overview page can render a summary tile
    without re-deriving them.
    """

    providers: list[ProviderHealthSchema]
    healthy: int = Field(description="How many providers are currently reachable.")
    degraded: int = Field(
        default=0,
        description=(
            "How many providers are not counted as reachable only because model discovery is "
            "unavailable for them. These may still serve requests."
        ),
    )
    total: int = Field(description="How many providers are configured.")
    checked_at: str | None = Field(
        default=None,
        description="ISO 8601 time of the most recent per-provider check (null if none yet).",
    )


def _to_health_schema(health: ProviderHealth) -> ProviderHealthSchema:
    return ProviderHealthSchema(
        instance=health.instance,
        ok=health.ok,
        model_count=health.model_count,
        error=health.error,
        checked_at=health.checked_at.isoformat() if health.checked_at else None,
        discovery_unsupported=health.discovery_unsupported,
    )


@router.get("/providers/health")
async def provider_health(
    config: Annotated[GatewayConfig, Depends(get_config)],
    hosted: HostedProviderServiceDep,
    refresh: bool = False,
) -> ProviderHealthResponse:
    """Report every configured provider's reachability, with a last-checked time.

    Covers the configured provider instances and every hosted provider switched
    on, each hosted provider under its provider name. A hosted provider that a
    configured instance of the same name shadows is not reported. One whose
    stored key cannot be decrypted with the configured `OTARI_SECRET_KEY` is
    reported unreachable with that reason.

    Reuses the per-provider model-discovery test path, so a provider is healthy
    when its credentials can list models. Results are served from the discovery
    cache (cheap enough to poll), so ``checked_at`` reflects when each provider
    was actually dialed. Pass ``refresh=true`` to force a live re-dial of every
    provider.

    A provider whose backend serves no model-listing endpoint cannot be verified
    this way, but it is not unreachable either: it is reported with
    ``discovery_unsupported`` and counted under ``degraded`` rather than as a
    reachability failure.
    """
    results = await check_all_provider_health(
        config,
        hosted=hosted.discovery_dials,
        refresh=refresh,
        serve_stale=background_discovery_enabled(config) and not refresh,
    )
    checked_ats = [health.checked_at for health in results if health.checked_at is not None]
    return ProviderHealthResponse(
        providers=[_to_health_schema(health) for health in sorted(results, key=lambda item: item.instance)],
        healthy=sum(1 for health in results if health.ok),
        degraded=sum(1 for health in results if not health.ok and health.discovery_unsupported),
        total=len(results),
        checked_at=max(checked_ats).isoformat() if checked_ats else None,
    )
