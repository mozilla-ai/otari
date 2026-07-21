"""Per-provider reachability health for the dashboard's provider health monitor.

This is a thin read-only view over the existing model-discovery test path: it
reuses ``discover_provider_models`` (the same call that backs the per-provider
"test connection" button) rather than introducing a second probe mechanism. A
provider is "healthy" when its credentials can list models, which is exactly what
that path already answers, cached and single-flighted.

The ``checked_at`` on each result is the wall-clock time the provider's
reachability was last evaluated (read from the discovery cache): a live dial for
most providers, or the moment a config-declared ``models:`` fallback was cached
for a backend that cannot list models. Either way it reflects when the gateway
last produced this verdict, not when the dashboard asked. The overview page
(issue #302) consumes this as a summary tile, so the counts (``healthy`` /
``total``) are computed here and kept reusable.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.model_discovery_service import discover_provider_models, get_model_cache

# A forced refresh only clears the cached listing when the provider's last dial is
# older than this. clear(instance) also detaches the in-flight discovery, so
# without this a burst of concurrent "Re-check all" refreshes would each start an
# independent dial per provider and defeat the discovery subsystem's single-flight
# coalescing. The window is short so a deliberate re-check after a fix still forces
# a fresh dial; a credential write already clears the cache for its instance
# (`_apply_write`), so fixing a key through the dashboard is unaffected either way.
_REFRESH_DEBOUNCE_SECONDS = 5.0


@dataclass
class ProviderHealth:
    """One provider instance's reachability, from the model-discovery test path."""

    instance: str
    ok: bool
    model_count: int
    error: str | None = None
    # Wall-clock time reachability was last evaluated (None if never checked yet).
    checked_at: datetime | None = None


def _refresh_should_redial(instance: str) -> bool:
    """Whether a forced refresh should clear the cache and re-dial ``instance``.

    True when the instance has never been checked or its last dial is older than
    the debounce window; False when a recent dial should be reused so rapid
    re-checks coalesce onto one in-flight discovery instead of each starting their
    own.
    """
    checked_at = get_model_cache().checked_at(instance)
    if checked_at is None:
        return True
    return datetime.now(UTC) - checked_at >= timedelta(seconds=_REFRESH_DEBOUNCE_SECONDS)


async def check_provider_health(
    config: GatewayConfig,
    instance: str,
    *,
    refresh: bool = False,
) -> ProviderHealth:
    """Report one instance's reachability via the shared model-discovery path.

    ``refresh`` forces a live re-dial by clearing the cached listing first (the
    same cache clear the stored-provider test uses), so an operator's explicit
    "re-check now" is a fresh probe rather than a cached verdict. A refresh that
    lands within ``_REFRESH_DEBOUNCE_SECONDS`` of the last dial is coalesced onto
    that recent result instead, so a burst of re-checks keeps the discovery
    subsystem's single-flight coalescing rather than detaching the in-flight task.
    """
    if refresh and _refresh_should_redial(instance):
        get_model_cache().clear(instance)
    discovery = await discover_provider_models(config, instance)
    return ProviderHealth(
        instance=instance,
        ok=discovery.error is None,
        model_count=len(discovery.models),
        error=discovery.error,
        checked_at=get_model_cache().checked_at(instance),
    )


async def check_all_provider_health(
    config: GatewayConfig,
    *,
    refresh: bool = False,
) -> list[ProviderHealth]:
    """Check every configured instance concurrently, keeping per-provider errors.

    One provider that somehow escapes ``discover_provider_models`` (which reports
    failure rather than raising) must not sink the whole health view, so a stray
    exception is surfaced as that instance's error, mirroring
    ``discover_models_with_status``.
    """
    instances = list(config.providers)
    results = await asyncio.gather(
        *(check_provider_health(config, name, refresh=refresh) for name in instances),
        return_exceptions=True,
    )
    health: list[ProviderHealth] = []
    for name, result in zip(instances, results, strict=True):
        if isinstance(result, BaseException):
            if not isinstance(result, Exception):
                raise result  # real cancellation still bubbles
            logger.warning("Provider health check raised for '%s': %s", name, type(result).__name__)
            health.append(
                ProviderHealth(
                    instance=name,
                    ok=False,
                    model_count=0,
                    error="Health check failed unexpectedly.",
                    checked_at=get_model_cache().checked_at(name),
                )
            )
        else:
            health.append(result)
    return health
