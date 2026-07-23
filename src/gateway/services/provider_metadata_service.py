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

import os
from dataclasses import dataclass, field

from any_llm import AnyLLM, LLMProvider

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


@dataclass
class KnownProvider:
    """One any-llm provider the add-provider picker can offer."""

    id: str
    name: str
    env_key: str | None = None
    default_api_base: str | None = None
    requires_api_key: bool = True
    # Whether ``env_key`` is already populated in the gateway's environment. When
    # true, any-llm can read the key from the env, so the add-provider form can
    # treat a pasted key as optional. Always False when there is no ``env_key``.
    env_key_present: bool = False


@dataclass
class KnownProviderSummary:
    """One provider offered in the add-provider picker: just an id and a name."""

    id: str
    name: str


def list_known_provider_summaries() -> list[KnownProviderSummary]:
    """Every any-llm provider the add-provider picker can offer: id + display name.

    Deliberately lightweight, so opening the picker does not lag: provider ids
    come from the any-llm registry and display names from the bundled
    genai-prices dataset, so *no provider SDK is imported*. The per-provider
    autofill hints (credential env var, default endpoint, whether a key is
    required) live on the provider class and are resolved lazily by
    :func:`known_provider_detail`, only for the provider an operator actually
    selects. A provider absent from genai-prices falls back to its id as the name.
    """
    summaries = [
        KnownProviderSummary(id=pid, name=_clean(getattr(_genai_provider(pid), "name", None)) or pid)
        for pid in AnyLLM.get_supported_providers()
    ]
    return sorted(summaries, key=lambda summary: summary.name.lower())


def known_provider_detail(provider_id: str) -> KnownProvider | None:
    """Autofill hints for one provider, importing only that provider's SDK.

    Imports a single any-llm provider module (not the whole catalog) to read its
    display name, credential env var, and default endpoint, then resolves
    ``env_key_present`` live against the current environment. Returns ``None`` when
    the id is not a known any-llm provider or its class cannot be imported (a
    missing optional dependency), which the route maps to a 404.
    """
    try:
        pid = LLMProvider.from_string(provider_id).value
    except Exception:
        return None
    try:
        cls = AnyLLM.get_provider_class(pid)
        meta = cls.get_provider_metadata()
    except Exception as exc:
        logger.debug("no provider detail for %r: %s", pid, exc)
        return None
    gp = _genai_provider(pid)
    name = _clean(getattr(gp, "name", None)) or _clean(getattr(meta, "name", None)) or pid
    raw_env = _clean(getattr(meta, "env_key", None))
    # any-llm uses the literal string "None" for keyless backends (Ollama, llama.cpp).
    env_key = None if raw_env in (None, "None") else raw_env
    return KnownProvider(
        id=pid,
        name=name,
        env_key=env_key,
        default_api_base=_clean(getattr(cls, "API_BASE", None)),
        requires_api_key=env_key is not None,
        env_key_present=env_key is not None and bool((os.getenv(env_key) or "").strip()),
    )
