"""Static provider metadata for the operator dashboard."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from gateway.api.deps import get_config, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.services.provider_metadata_service import ProviderInfo, list_provider_info

router = APIRouter(prefix="/v1", tags=["providers"])


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


class ProvidersResponse(BaseModel):
    """Metadata for every configured provider."""

    providers: list[ProviderInfoSchema]


def _to_schema(info: ProviderInfo) -> ProviderInfoSchema:
    caps = info.capabilities
    return ProviderInfoSchema(
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


@router.get("/providers", dependencies=[Depends(verify_master_key)])
async def list_providers(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ProvidersResponse:
    """List static metadata for every configured provider.

    Operator-facing: reports each provider's capabilities, documentation and
    pricing links, and display name from the bundled any-llm and genai-prices
    datasets. No provider is contacted, so this is cheap and always available.
    Master-key gated because it describes the gateway's own configuration.
    """
    return ProvidersResponse(providers=[_to_schema(info) for info in list_provider_info(config)])
