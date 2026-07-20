"""Static, network-free metadata for the configured providers.

Powers the dashboard's provider detail view: which capabilities a provider
exposes (streaming, vision, embeddings, ...), where its docs and pricing pages
live, and a human-friendly name. Everything here comes from two bundled
datasets, so no provider is contacted:

- ``any_llm`` :class:`ProviderMetadata` (keyed by the provider *type*) supplies
  the capability flags, documentation URL, and credential env var.
- ``genai-prices`` (keyed by the provider *id*) supplies the display name,
  description, and pricing-doc URLs.

Model *counts* are deliberately not computed here; those need a live discovery
call, which the dashboard already has from GET /v1/models/discoverable and can
join client-side.
"""

from dataclasses import dataclass, field

from any_llm import AnyLLM

from gateway.core.config import GatewayConfig
from gateway.log_config import logger


@dataclass
class ProviderCapabilities:
    """Curated capability flags for a provider, from any-llm metadata."""

    streaming: bool = False
    reasoning: bool = False
    vision: bool = False
    pdf: bool = False
    embeddings: bool = False
    image_generation: bool = False
    audio: bool = False
    rerank: bool = False
    responses_api: bool = False
    moderation: bool = False
    list_models: bool = False


@dataclass
class ProviderInfo:
    """Static metadata for one configured provider instance."""

    instance: str
    provider_type: str
    name: str
    doc_url: str | None = None
    description: str | None = None
    env_key: str | None = None
    pricing_urls: list[str] = field(default_factory=list)
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)


def _any_llm_metadata(provider_type: str) -> object | None:
    """any-llm ``ProviderMetadata`` for a provider type, or ``None`` if unknown.

    A custom or misspelled provider type simply has no bundled metadata; that is
    not an error worth failing the whole listing over.
    """
    try:
        return AnyLLM.get_provider_class(provider_type).get_provider_metadata()
    except Exception as exc:
        logger.debug("no any-llm metadata for provider type %r: %s", provider_type, exc)
        return None


def _genai_provider(provider_type: str) -> object | None:
    """genai-prices provider record for a provider id, or ``None`` if unknown."""
    try:
        from genai_prices.data_snapshot import get_snapshot

        for provider in get_snapshot().providers:
            if provider.id == provider_type:
                return provider
    except Exception as exc:
        logger.debug("genai-prices provider lookup failed for %r: %s", provider_type, exc)
    return None


def _capabilities(meta: object | None) -> ProviderCapabilities:
    """Map any-llm's raw capability flags to the curated subset the UI shows."""
    if meta is None:
        return ProviderCapabilities()

    def flag(name: str) -> bool:
        return bool(getattr(meta, name, False))

    return ProviderCapabilities(
        streaming=flag("streaming"),
        reasoning=flag("reasoning"),
        vision=flag("image"),
        pdf=flag("pdf"),
        embeddings=flag("embedding"),
        image_generation=flag("image_generation"),
        # any-llm splits audio into transcription and speech; the UI shows one
        # "Audio" badge, so either capability lights it.
        audio=flag("audio_transcription") or flag("audio_speech"),
        rerank=flag("rerank"),
        responses_api=flag("responses"),
        moderation=flag("moderation"),
        list_models=flag("list_models"),
    )


def _clean(text: str | None) -> str | None:
    """Trim to a non-empty string, or ``None`` (the dataset uses "" for absent)."""
    if text is None:
        return None
    stripped = text.strip()
    return stripped or None


def provider_info(config: GatewayConfig, instance: str) -> ProviderInfo:
    """Assemble static metadata for one configured provider instance."""
    provider_type = config.provider_instance_type(instance)
    meta = _any_llm_metadata(provider_type)
    gp = _genai_provider(provider_type)

    # Prefer genai-prices' display name ("OpenAI") over any-llm's lowercase type
    # name ("openai"); fall back to the configured instance key.
    name = _clean(getattr(gp, "name", None)) or _clean(getattr(meta, "name", None)) or instance

    return ProviderInfo(
        instance=instance,
        provider_type=provider_type,
        name=name,
        doc_url=_clean(getattr(meta, "doc_url", None)),
        description=_clean(getattr(gp, "description", None)),
        env_key=_clean(getattr(meta, "env_key", None)),
        pricing_urls=list(getattr(gp, "pricing_urls", None) or []),
        capabilities=_capabilities(meta),
    )


def list_provider_info(config: GatewayConfig) -> list[ProviderInfo]:
    """Static metadata for every configured provider, sorted by instance name."""
    return sorted(
        (provider_info(config, instance) for instance in config.providers),
        key=lambda info: info.instance,
    )
