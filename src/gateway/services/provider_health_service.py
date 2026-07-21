"""Per-provider reachability health for the dashboard's provider health monitor.

This is a thin read-only view over the existing model-discovery test path: it
reuses ``discover_provider_models`` (the same call that backs the per-provider
"test connection" button) rather than introducing a second probe mechanism. A
provider is "healthy" when its credentials can list models, which is exactly what
that path already answers, cached and single-flighted.

The ``checked_at`` on each result is the wall-clock time the underlying dial was
made (read from the discovery cache), so a status served from cache honestly
reports when the provider was last reached, not when the dashboard last asked.
The overview page (issue #302) consumes this as a summary tile, so the counts
(``healthy`` / ``total``) are computed here and kept reusable.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.model_discovery_service import discover_provider_models, get_model_cache


@dataclass
class ProviderHealth:
    """One provider instance's reachability, from the model-discovery test path."""

    instance: str
    ok: bool
    model_count: int
    error: str | None = None
    # Wall-clock time the underlying dial was made (None if never checked yet).
    checked_at: datetime | None = None


async def check_provider_health(
    config: GatewayConfig,
    instance: str,
    *,
    refresh: bool = False,
) -> ProviderHealth:
    """Report one instance's reachability via the shared model-discovery path.

    ``refresh`` clears the cached listing for this instance first, forcing a live
    re-dial (the same cache clear the stored-provider test uses), so an operator's
    explicit "re-check now" is a fresh probe rather than a cached verdict.
    """
    if refresh:
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
