"""A startup check on the localities ``providers:`` instances declare for routing eligibility.

A policy's ``eligibility`` trusts an instance's ``routing_locality``: the
operator knows where a proxy forwards a request, and the gateway does not. A
``local`` or ``private`` instance whose endpoint resolves to a public address is
still most likely a mistake, though, so the start says so once. It never
changes what the declaration admits.
"""

import asyncio
from urllib.parse import urlsplit

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.routing import DeploymentLocality
from gateway.services.url_safety import public_addresses_of

__all__ = ["warn_about_public_endpoints_declared_private"]

# A name that has not resolved by then is not worth holding the start for.
_RESOLVE_SECONDS = 5.0


async def _resolves_public(host: str) -> bool:
    try:
        return bool(await asyncio.wait_for(public_addresses_of(host), timeout=_RESOLVE_SECONDS))
    except TimeoutError:
        return False


async def warn_about_public_endpoints_declared_private(config: GatewayConfig) -> list[str]:
    """Warn for each ``local`` or ``private`` instance whose ``api_base`` resolves to a public address.

    Returns the instances warned about, for tests.
    """
    hosts: dict[str, str] = {}
    for instance, entry in config.providers.items():
        locality = entry.get("routing_locality")
        if locality is None or locality == DeploymentLocality.CLOUD:
            continue
        base = entry.get("api_base")
        host = urlsplit(base).hostname if isinstance(base, str) else None
        if host:
            hosts[instance] = host
    if not hosts:
        return []
    verdicts = await asyncio.gather(*(_resolves_public(host) for host in hosts.values()))
    flagged = [instance for instance, public in zip(hosts, verdicts, strict=True) if public]
    for instance in flagged:
        logger.warning(
            "providers.%s declares routing_locality '%s', but its api_base host '%s' resolves to a public "
            "address. Routing policies trust the declaration; check that it describes where requests go.",
            instance,
            config.providers[instance]["routing_locality"],
            hosts[instance],
        )
    return flagged
