"""Auto-discovery of models from configured providers with in-memory TTL caching."""

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from any_llm import AnyLLM, LLMProvider, alist_models
from any_llm.types.model import Model

from gateway.api.routes.chat import get_provider_kwargs
from gateway.core.config import GatewayConfig
from gateway.log_config import logger


@dataclass
class _CacheEntry:
    """A single provider's cached model list."""

    models: list[Model]
    cached_at: float  # time.monotonic()


@dataclass
class ModelCache:
    """In-memory TTL cache for discovered models, keyed by provider name."""

    _store: dict[str, _CacheEntry] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def get(self, provider: str, ttl: int) -> list[Model] | None:
        """Return cached models if still valid, otherwise None.

        Returns a shallow copy so callers cannot mutate the internal cache.
        """
        entry = self._store.get(provider)
        if entry is None:
            return None
        if ttl <= 0:
            return None
        elapsed = time.monotonic() - entry.cached_at
        if elapsed >= ttl:
            return None
        return list(entry.models)

    def set(self, provider: str, models: list[Model]) -> None:
        """Store a provider's model list with the current timestamp."""
        self._store[provider] = _CacheEntry(models=list(models), cached_at=time.monotonic())

    def get_all_cached(self, ttl: int | None = None) -> dict[str, list[Model]]:
        """Return stored entries, optionally filtering by TTL.

        Returns shallow copies so callers cannot mutate the internal cache.

        Args:
            ttl: If set, only return entries that are still valid. If None, return all.
        """
        result: dict[str, list[Model]] = {}
        now = time.monotonic()
        for provider, entry in self._store.items():
            if ttl is not None:
                elapsed = now - entry.cached_at
                if ttl <= 0 or elapsed >= ttl:
                    continue
            result[provider] = list(entry.models)
        return result

    def clear(self, provider: str | None = None) -> None:
        """Invalidate one or all providers."""
        if provider is None:
            self._store.clear()
        else:
            self._store.pop(provider, None)


# Module-level singleton shared across requests within a worker process.
_model_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """Return the module-level model cache singleton."""
    return _model_cache


def _supports_list_models(provider_name: str) -> bool:
    """Check whether a provider supports model listing without instantiating it."""
    try:
        provider_class = AnyLLM.get_provider_class(provider_name)
        metadata = provider_class.get_provider_metadata()
        return metadata.list_models
    except (ImportError, AttributeError, Exception):
        return False


async def _discover_for_provider(
    provider_name: str,
    config: GatewayConfig,
) -> tuple[str, list[Model]]:
    """Discover models for a single provider. Returns (provider_name, models)."""
    provider_enum = LLMProvider(provider_name)
    kwargs = get_provider_kwargs(config, provider_enum)

    api_key = kwargs.pop("api_key", None)
    api_base = kwargs.pop("api_base", None)
    client_args = kwargs.pop("client_args", None)

    models: Sequence[Model] = await alist_models(
        provider=provider_enum,
        api_key=api_key,
        api_base=api_base,
        client_args=client_args,
        **kwargs,
    )
    return provider_name, list(models)


async def discover_all_models(
    config: GatewayConfig,
    provider_filter: str | None = None,
) -> list[tuple[str, Model]]:
    """Discover models from configured providers with caching.

    Args:
        config: Gateway configuration with provider credentials.
        provider_filter: If set, only discover models for this provider.

    Returns:
        List of (provider_name, Model) tuples so callers can build model_key
        from the configured provider key rather than relying on ``owned_by``.

    """
    cache = get_model_cache()
    ttl = config.model_cache_ttl_seconds

    if provider_filter:
        providers_to_query = [provider_filter] if provider_filter in config.providers else []
    else:
        providers_to_query = list(config.providers.keys())

    # Separate providers into cache-hit and cache-miss groups.
    result_models: list[tuple[str, Model]] = []
    providers_needing_fetch: list[str] = []

    for provider_name in providers_to_query:
        cached = cache.get(provider_name, ttl)
        if cached is not None:
            result_models.extend((provider_name, m) for m in cached)
        else:
            if _supports_list_models(provider_name):
                providers_needing_fetch.append(provider_name)
            else:
                logger.debug("Provider '%s' does not support model listing, skipping", provider_name)

    if not providers_needing_fetch:
        return result_models

    # Fetch all cache-miss providers concurrently.
    async with cache._lock:
        # Double-check cache under lock to prevent thundering herd.
        still_needed: list[str] = []
        for provider_name in providers_needing_fetch:
            cached = cache.get(provider_name, ttl)
            if cached is not None:
                result_models.extend((provider_name, m) for m in cached)
            else:
                still_needed.append(provider_name)

        if still_needed:
            tasks = [_discover_for_provider(name, config) for name in still_needed]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                provider_name = still_needed[i]
                if isinstance(result, BaseException):
                    logger.warning(
                        "Model discovery failed for provider '%s': %s",
                        provider_name,
                        result,
                    )
                    continue
                _, models = result
                cache.set(provider_name, models)
                result_models.extend((provider_name, m) for m in models)

    return result_models
