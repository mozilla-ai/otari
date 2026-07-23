"""Auto-discovery of models from configured providers with in-memory TTL caching.

One process-level cache backs all discovery consumers (``GET /v1/models``,
``GET /v1/models/discoverable``, and the stored-provider connection test). It
does three things that keep a broken or slow provider from turning into a storm
of upstream calls:

- **Positive caching**: a successful ``list_models`` result is reused for
  ``model_cache_ttl_seconds``.
- **Negative caching**: a *failure* is remembered for
  ``model_discovery_negative_ttl_seconds`` (shorter), so an unreachable provider
  is not re-dialed on every single request. Without this, failures are never
  cached and each request re-pays the provider's full timeout.
- **Single-flight**: concurrent callers for the same provider share one
  in-flight ``list_models`` call instead of each firing their own. This is what
  stops ``/v1/models`` and ``/v1/models/discoverable`` (both mounted on the
  dashboard's Models page) from doubling every fanout.
"""

import asyncio
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from any_llm import AnyLLM, LLMProvider, alist_models
from any_llm.types.model import Model

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.provider_kwargs import get_provider_kwargs

# Fallback bound for ad-hoc credential tests when a caller does not pass one. The
# route passes ``model_discovery_timeout_seconds`` so the saved and unsaved paths
# agree even when an operator raises the configured timeout; this default only
# applies to direct callers (e.g. tests).
_ADHOC_DISCOVERY_TIMEOUT_SECONDS = 10.0


@dataclass
class ProviderDiscovery:
    """One instance's discovery result, including why it came back empty."""

    provider: str
    models: list[Model]
    error: str | None = None


def _copy_discovery(discovery: ProviderDiscovery) -> ProviderDiscovery:
    """Shallow copy so the cache stays immutable from the outside.

    The cache stores one canonical result per provider; every read hands back a
    copy so a caller mutating ``.models`` or ``.error`` cannot corrupt the cached
    entry (or the view another concurrent awaiter holds).
    """
    return ProviderDiscovery(provider=discovery.provider, models=list(discovery.models), error=discovery.error)


@dataclass
class _CacheEntry:
    """A single provider's cached discovery result (success or failure)."""

    result: ProviderDiscovery
    cached_at: float  # time.monotonic(), for TTL math (immune to wall-clock jumps)
    checked_at: datetime  # wall-clock time the result was produced, for "last checked" display


@dataclass
class ModelCache:
    """In-memory cache for discovered models, keyed by provider instance name.

    Holds both successful and failed discoveries (freshness is governed per read
    by separate positive/negative TTLs) and coalesces concurrent discoveries of
    the same provider through ``_inflight`` so only one upstream call is made.
    """

    _store: dict[str, _CacheEntry] = field(default_factory=dict)
    _inflight: dict[str, "asyncio.Task[ProviderDiscovery]"] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def get(self, provider: str, ttl: int) -> list[Model] | None:
        """Peek at a fresh, *successful* cached listing, else ``None``.

        Read-only: never triggers discovery and never returns a negatively
        cached failure. Returns a shallow copy so callers cannot mutate the
        internal cache. Freshness is bound by the caller-supplied ``ttl``.
        """
        if ttl <= 0:
            return None
        entry = self._store.get(provider)
        if entry is None or entry.result.error is not None:
            return None
        if time.monotonic() - entry.cached_at >= ttl:
            return None
        return list(entry.result.models)

    def set(self, provider: str, models: list[Model]) -> None:
        """Store a successful listing (priming/test helper)."""
        self._store[provider] = _CacheEntry(
            result=ProviderDiscovery(provider=provider, models=list(models)),
            cached_at=time.monotonic(),
            checked_at=datetime.now(UTC),
        )

    def checked_at(self, provider: str) -> datetime | None:
        """Wall-clock time this provider's cached result was produced, or ``None``.

        The provider-health monitor reports this as each instance's "last checked"
        time, so a status served from the cache honestly shows when the underlying
        provider was actually dialed rather than when the dashboard last asked.
        """
        entry = self._store.get(provider)
        return entry.checked_at if entry is not None else None

    def clear(self, provider: str | None = None) -> None:
        """Invalidate one or all cached results.

        Also detaches any in-flight discovery, so a caller arriving after the
        clear starts a fresh one instead of riding a result computed from the
        now-stale credentials. The detached task still finishes for callers that
        were already awaiting it, but it no longer repopulates the cache (see
        ``_run``). This is what keeps the post-write "test connection" a live
        check rather than the pre-change listing.
        """
        if provider is None:
            self._store.clear()
            self._inflight.clear()
        else:
            self._store.pop(provider, None)
            self._inflight.pop(provider, None)

    def _fresh(self, provider: str, positive_ttl: float, negative_ttl: float) -> ProviderDiscovery | None:
        """Return the cached result if still fresh under the applicable TTL."""
        entry = self._store.get(provider)
        if entry is None:
            return None
        ttl = negative_ttl if entry.result.error is not None else positive_ttl
        if ttl <= 0:
            return None
        if time.monotonic() - entry.cached_at >= ttl:
            return None
        return entry.result

    async def get_or_discover(
        self,
        provider: str,
        *,
        positive_ttl: float,
        negative_ttl: float,
        discover: Callable[[], Awaitable[ProviderDiscovery]],
    ) -> ProviderDiscovery:
        """Return a cached result, or run ``discover`` once (single-flight).

        A hit (positive or negative) returns immediately. On a miss, the first
        caller starts the discovery and every concurrent caller for the same
        provider awaits that one task, so a slow provider is dialed once, not
        once per request. ``discover`` is expected to report failure by
        returning a ``ProviderDiscovery`` with ``error`` set rather than raising.
        """
        cached = self._fresh(provider, positive_ttl, negative_ttl)
        if cached is not None:
            return _copy_discovery(cached)

        async with self._lock:
            cached = self._fresh(provider, positive_ttl, negative_ttl)
            if cached is not None:
                return _copy_discovery(cached)
            task = self._inflight.get(provider)
            if task is None:
                task = asyncio.ensure_future(self._run(provider, discover))
                self._inflight[provider] = task

        # shield so one caller's cancellation does not abort the shared discovery
        # that other callers are still awaiting. Copy on the way out so the cached
        # entry (stored by _run) is never handed to a caller by reference.
        return _copy_discovery(await asyncio.shield(task))

    async def _run(
        self,
        provider: str,
        discover: Callable[[], Awaitable[ProviderDiscovery]],
    ) -> ProviderDiscovery:
        task = asyncio.current_task()
        try:
            result = await discover()
            async with self._lock:
                # Cache only if still the registered in-flight: a clear() (e.g. a
                # credential change) between start and finish detaches us, and this
                # now-stale result must neither repopulate the cache nor clobber a
                # newer discovery's entry. Storing ``result`` directly is safe
                # because ``get_or_discover`` hands every caller a copy, so no
                # external reference to this stored object survives.
                if self._inflight.get(provider) is task:
                    self._store[provider] = _CacheEntry(
                        result=result, cached_at=time.monotonic(), checked_at=datetime.now(UTC)
                    )
            return result
        finally:
            async with self._lock:
                if self._inflight.get(provider) is task:
                    self._inflight.pop(provider, None)


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


def _env_provider_instances(configured_impls: set[str]) -> list[str]:
    """any-llm provider implementations made callable by their credential env var.

    Completion routing works for any any-llm provider once its native credential
    env var is set: ``get_provider_kwargs`` returns ``{}`` for a provider absent
    from ``config.providers``, so any-llm reads the env var directly. Discovery
    would otherwise only see ``config.providers``, so such a provider is callable
    yet missing from GET /v1/models. This closes that gap by treating a provider
    whose credential env var is present as discoverable under its implementation
    name, which is exactly the request selector ``<impl>:<model>`` that routing
    resolves.

    Only providers that support live model listing are included; one that cannot
    list models would just error out and be dropped from the catalog. A provider
    whose implementation already backs a configured instance is skipped
    (``configured_impls`` holds those implementation names): otherwise a custom
    instance such as ``my-anthropic`` (``provider_type: anthropic``) alongside
    ``ANTHROPIC_API_KEY`` would dial the same credential twice and list the same
    models under two keys.

    A keyless local provider (``ollama``/``llamacpp``/``llamafile``, whose
    ``env_key`` is absent) has no credential signal to detect and so is not
    surfaced here; discovering it would mean dialing a fixed localhost endpoint on
    spec. That narrow callable-but-undiscoverable case is tracked in issue #389.
    """
    detected: list[str] = []
    try:
        all_metadata = AnyLLM.get_all_provider_metadata()
    except Exception:
        # Match the module's defensive posture (see _supports_list_models and the
        # return_exceptions fanouts): a registry hiccup must not 500 GET /v1/models
        # or /v1/models/discoverable. Fall back to configured-only discovery.
        logger.warning("Could not enumerate provider metadata for env-based discovery", exc_info=True)
        return detected
    for metadata in all_metadata:
        name = metadata.name
        if name in configured_impls or not metadata.list_models:
            continue
        env_key = metadata.env_key
        if not env_key:
            continue
        # ENV_API_KEY_NAME may name interchangeable alternatives separated by "/"
        # (e.g. gemini's "GEMINI_API_KEY/GOOGLE_API_KEY"); any one set is enough.
        # Compound descriptions ("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        # and the literal "None" are not real single variable names and simply
        # never match os.environ. A blank or whitespace-only value (common from
        # container templating) is treated as unset, so it does not add a provider
        # that would only fail to authenticate.
        candidates = (part.strip() for part in env_key.split("/"))
        if any(part and os.environ.get(part, "").strip() for part in candidates):
            detected.append(name)
    return detected


def _discoverable_instances(config: GatewayConfig) -> list[str]:
    """Instance names to run discovery for.

    Configured instances first, then any-llm providers made discoverable by an
    env var alone (see ``_env_provider_instances``), so GET /v1/models matches
    what completion routing can actually reach. Env detection is deduplicated
    against the *implementations* the configured instances resolve to, so a
    custom-named instance is never shadowed by a bare env-detected duplicate.
    """
    configured = list(config.providers.keys())
    configured_impls = {config.provider_instance_type(instance) for instance in configured}
    return configured + _env_provider_instances(configured_impls)


def _declared_models(config: GatewayConfig, instance: str) -> list[Model]:
    """Build Model rows from an instance's declared ``models:`` list.

    Used for instances whose backend has no ``/v1/models`` endpoint, so the
    operator declares the served model ids in config instead. ``owned_by`` is the
    instance name so the listing key (``instance:model``) matches the request
    selector.
    """
    entry = config.providers.get(instance) or {}
    declared = entry.get("models") or []
    return [Model(id=model_id, created=0, object="model", owned_by=instance) for model_id in declared]


async def _discover_for_provider(
    provider_name: str,
    config: GatewayConfig,
) -> tuple[str, list[Model]]:
    """Discover models for a single instance. Returns (instance_name, models).

    ``provider_name`` is the configured instance; the underlying implementation
    is resolved from its ``provider_type``. When the live ``list_models`` call
    fails (e.g. an OpenAI-compatible backend that does not implement
    ``/v1/models``), fall back to the instance's declared ``models:`` list.
    """
    provider_enum = LLMProvider(config.provider_instance_type(provider_name))
    kwargs = get_provider_kwargs(config, provider_enum, instance=provider_name)

    api_key = kwargs.pop("api_key", None)
    api_base = kwargs.pop("api_base", None)
    client_args = kwargs.pop("client_args", None)

    try:
        # Bound the live call so an unreachable or slow provider fails fast
        # instead of pinning discovery for the underlying client's default
        # timeout (any-llm builds an AsyncOpenAI with no timeout, so that
        # default is ~600s with 2 retries).
        models: Sequence[Model] = await asyncio.wait_for(
            alist_models(
                provider=provider_enum,
                api_key=api_key,
                api_base=api_base,
                client_args=client_args,
                **kwargs,
            ),
            timeout=config.model_discovery_timeout_seconds,
        )
    except Exception as exc:
        declared = _declared_models(config, provider_name)
        if declared:
            # Log the underlying error at debug so a real misconfig (bad auth,
            # wrong api_base) on a backend that *does* support /v1/models is
            # diagnosable, rather than silently masked by the declared fallback.
            logger.debug("list_models failed for instance '%s': %s", provider_name, exc)
            logger.info(
                "list_models failed for instance '%s'; using declared models: list (%d)",
                provider_name,
                len(declared),
            )
            return provider_name, declared
        raise
    return provider_name, list(models)


# A provider error is echoed to a master-key caller, who already holds more
# authority than anything it could reveal; it is capped so a stack-trace-sized
# message cannot fill the response.
_ERROR_MAX_CHARS = 300


def _short_error(exc: BaseException, provider: str | None = None) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    # any-llm prefixes a provider error with a "[provider]" tag (e.g.
    # "[anthropic] No anthropic API key provided…"). Every surface that shows this
    # is already provider-specific, so the tag is redundant noise; drop it when it
    # names the provider being tested. Any other bracketed text is left intact.
    if provider and message.startswith(f"[{provider}]"):
        # Re-apply the class-name fallback: a message that was only the tag (e.g.
        # "[anthropic]") strips to "", which would render as a blank error.
        message = message[len(provider) + 2 :].lstrip() or exc.__class__.__name__
    if len(message) > _ERROR_MAX_CHARS:
        return message[: _ERROR_MAX_CHARS - 1] + "…"
    return message


async def _discover_uncached(config: GatewayConfig, instance: str) -> ProviderDiscovery:
    """Resolve one instance's models, reporting failure rather than raising.

    Pure discovery with no cache interaction: the cache layer
    (``ModelCache.get_or_discover``) is responsible for reuse and single-flight.
    """
    impl = config.provider_instance_type(instance)
    if not _supports_list_models(impl):
        declared = _declared_models(config, instance)
        if declared:
            return ProviderDiscovery(provider=instance, models=declared)
        return ProviderDiscovery(
            provider=instance,
            models=[],
            error=(
                f"Provider '{impl}' cannot list models. Declare the model ids this instance "
                "serves under its 'models:' key in config.yml."
            ),
        )

    try:
        _, models = await _discover_for_provider(instance, config)
    except Exception as exc:
        # Log the class only, never str(exc): some providers echo a partial key
        # or endpoint in the message. The capped text still reaches the
        # master-key caller in the response, where it is a useful diagnostic.
        logger.info("Model discovery failed for instance '%s' (%s)", instance, type(exc).__name__)
        return ProviderDiscovery(provider=instance, models=[], error=_short_error(exc, provider=impl))

    return ProviderDiscovery(provider=instance, models=models)


async def discover_provider_models(config: GatewayConfig, instance: str) -> ProviderDiscovery:
    """Discover one instance's models, cached (positive + negative) and single-flighted.

    ``discover_all_models`` drops a failing provider and logs it, which is right
    for a catalog served to API callers: one broken provider should not blank the
    listing. An operator choosing a model needs the opposite, because an empty
    dropdown and a provider whose key is wrong look identical.
    """
    cache = get_model_cache()
    return await cache.get_or_discover(
        instance,
        positive_ttl=config.model_cache_ttl_seconds,
        negative_ttl=config.model_discovery_negative_ttl_seconds,
        discover=lambda: _discover_uncached(config, instance),
    )


async def test_provider_credentials(
    impl_name: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, object] | None = None,
    timeout: float = _ADHOC_DISCOVERY_TIMEOUT_SECONDS,
) -> ProviderDiscovery:
    """List models for ad-hoc credentials without storing them.

    Backs the dashboard's "test connection" before a provider is saved. Reports
    failure rather than raising, and never echoes the api key (only sanitized,
    capped provider errors, which may include the api_base but never the key).
    ``timeout`` defaults to the module bound; the route passes
    ``model_discovery_timeout_seconds`` so the saved and unsaved paths agree.
    """
    if not _supports_list_models(impl_name):
        return ProviderDiscovery(
            provider=impl_name,
            models=[],
            error=f"Provider '{impl_name}' cannot list models, so a connection cannot be verified this way.",
        )
    try:
        provider_enum = LLMProvider(impl_name)
    except ValueError:
        return ProviderDiscovery(
            provider=impl_name,
            models=[],
            error=f"'{impl_name}' is not a known provider implementation.",
        )
    try:
        # Bounded like the stored-provider path so a black-holed endpoint cannot
        # hang the test button for the SDK's ~600s default.
        models = await asyncio.wait_for(
            alist_models(
                provider=provider_enum,
                api_key=api_key,
                api_base=api_base,
                client_args=client_args,
            ),
            timeout=timeout,
        )
    except Exception as exc:
        # Class only in the log (see _discover_uncached); the capped
        # message still goes back to the master-key caller who owns the key.
        logger.info("Provider connection test failed for '%s' (%s)", impl_name, type(exc).__name__)
        return ProviderDiscovery(provider=impl_name, models=[], error=_short_error(exc, provider=impl_name))
    return ProviderDiscovery(provider=impl_name, models=list(models))


async def discover_models_with_status(config: GatewayConfig) -> list[ProviderDiscovery]:
    """Discover every configured instance's models concurrently, keeping errors.

    Deliberately not gated on ``config.model_discovery``: that flag governs what
    GET /v1/models publishes to API callers, and an operator who curates that
    listing still has to be able to see what their own credentials can reach.
    Includes providers made callable by an env var alone, so this operator view
    agrees with completion routing and with GET /v1/models.
    """
    instances = _discoverable_instances(config)
    # return_exceptions so one provider that somehow escapes _discover_uncached
    # cannot 500 the whole operator listing (this route awaits with no guard);
    # surface it as a per-provider error instead. Real cancellation still bubbles.
    results = await asyncio.gather(
        *(discover_provider_models(config, name) for name in instances),
        return_exceptions=True,
    )
    discoveries: list[ProviderDiscovery] = []
    for name, result in zip(instances, results, strict=True):
        if isinstance(result, BaseException):
            if not isinstance(result, Exception):
                raise result
            logger.warning("Model discovery raised for provider '%s': %s", name, result)
            discoveries.append(ProviderDiscovery(provider=name, models=[], error=_short_error(result, provider=name)))
        else:
            discoveries.append(result)
    return discoveries


async def discover_all_models(
    config: GatewayConfig,
    provider_filter: str | None = None,
) -> list[tuple[str, Model]]:
    """Discover models from discoverable providers with caching.

    Discoverable providers are the configured instances plus any-llm providers
    made callable by an env var alone (see ``_discoverable_instances``), so the
    catalog matches what completion routing can actually reach.

    Args:
        config: Gateway configuration with provider credentials.
        provider_filter: If set, only discover models for this provider.

    Returns:
        List of (provider_name, Model) tuples so callers can build model_key
        from the configured provider key rather than relying on ``owned_by``.

    """
    discoverable = _discoverable_instances(config)
    if provider_filter:
        instances = [provider_filter] if provider_filter in discoverable else []
    else:
        instances = discoverable

    # Each instance goes through the shared cache + single-flight path, so a
    # failing provider is dialed at most once per negative-TTL window and the
    # concurrent discoverable listing reuses the same in-flight call.
    # return_exceptions so a single provider cannot abort the catalog build; real
    # cancellation still bubbles.
    results = await asyncio.gather(
        *(discover_provider_models(config, name) for name in instances),
        return_exceptions=True,
    )

    result_models: list[tuple[str, Model]] = []
    for name, discovery in zip(instances, results, strict=True):
        if isinstance(discovery, BaseException):
            if not isinstance(discovery, Exception):
                raise discovery
            logger.warning("Model discovery raised for provider '%s': %s", name, discovery)
            continue
        # A failed provider comes back with error set and no models: drop it from
        # the API catalog so one bad provider does not blank the whole listing.
        result_models.extend((discovery.provider, model) for model in discovery.models)
    return result_models
