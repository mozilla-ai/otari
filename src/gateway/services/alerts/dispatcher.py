"""Delivering one alert to one Apprise URL, without ever failing a caller.

This module knows nothing about budgets. It takes a destination URL, a title and
a body, and reports whether the send worked.

**Why Apprise.** One string covers Slack, PagerDuty, mail and a plain webhook,
so the schema is one column rather than one per vendor. There is no port over it
because Apprise is itself the vendor-neutral layer, so a port would be a seam
with one possible implementation on either side.

**Threads, not the event loop.** ``Apprise.async_notify`` is a coroutine but its
plugins are synchronous ``requests`` calls it offloads with
``run_in_executor(None, ...)``. A send never blocks the loop; it does hold a
default-executor thread for an HTTP round trip, which is what
:data:`SEND_TIMEOUT_SECONDS` and :func:`_bound_sockets` between them bound.

**Logging.** The destination is a credential: ``slack://`` embeds a bot token,
``json://`` can embed basic-auth. Nothing here passes a URL to the gateway
logger; callers log the stored redaction. :data:`_ASSET` sets
``secure_logging`` explicitly so Apprise's own redaction survives a change to
its defaults.
"""

import asyncio
from dataclasses import dataclass
from typing import Final

import apprise

from gateway.log_config import logger

# How long this coroutine waits for a send before giving up on it.
SEND_TIMEOUT_SECONDS: Final = 15.0

# ``asyncio.wait_for`` cancels the await, not the work: a thread already inside
# a socket read cannot be cancelled. Apprise defaults both timeouts to 4s but
# lets the destination URL raise them (``?cto=600``), which would park a shared
# executor thread for ten minutes a tick, so :func:`_bound_sockets` clamps them.
SOCKET_CONNECT_TIMEOUT_SECONDS: Final = 5.0
SOCKET_READ_TIMEOUT_SECONDS: Final = 10.0

# ``secure_logging`` keeps a bot token out of Apprise's own log records.
_ASSET: Final = apprise.AppriseAsset(secure_logging=True)


class UnsupportedAlertDestinationError(ValueError):
    """Apprise cannot parse the destination, or Otari does not accept its schema.

    Raised at rule-write time, so an operator learns their URL is unusable while
    the form is open rather than when a budget crosses a threshold weeks later.
    """


@dataclass(frozen=True)
class ParsedDestination:
    """What a destination URL turned out to be, before any policy is applied.

    ``host`` is the address the plugin will actually connect to, or None when
    the plugin has none because its endpoint is compiled in. It is the value the
    SSRF gate checks; deciding *whether* to check it is the caller's job, since
    the netloc of a vendor schema is a token rather than a host.
    """

    scheme: str
    service_name: str
    host: str | None


@dataclass(frozen=True)
class AlertDispatchResult:
    """Whether one send succeeded, and a short reason when it did not.

    ``detail`` is stored on the delivery row and shown to the operator, so it
    says what happened without naming the destination.
    """

    delivered: bool
    detail: str | None = None


def parse_destination(destination: str) -> ParsedDestination:
    """Check a destination is one Apprise can deliver to, and report what it is.

    ``suppress_exceptions`` keeps a malformed URL from surfacing as whatever the
    plugin's constructor happened to raise, so every rejection reaches the caller
    as one domain error.

    ``smtp_host`` before ``host`` because Apprise's mail plugin maps a well-known
    domain onto its provider's server (``mailto://u:p@gmail.com`` sends to
    ``smtp.gmail.com``), and the gate has to check the address that is dialed.
    """
    stripped = destination.strip()
    if not stripped:
        raise UnsupportedAlertDestinationError("An alert destination cannot be empty")
    plugin = apprise.Apprise.instantiate(stripped, asset=_ASSET, suppress_exceptions=True)
    if plugin is None:
        # Deliberately not echoing the URL back: an invalid one is still a
        # string the operator may have pasted a real token into.
        raise UnsupportedAlertDestinationError(
            "Apprise does not recognize this destination. Expected a URL such as "
            "slack://token/channel, discord://webhook_id/webhook_token, or json://host/path"
        )
    host = getattr(plugin, "smtp_host", None) or getattr(plugin, "host", None)
    service_name = getattr(plugin, "service_name", None)
    return ParsedDestination(
        scheme=stripped.split(":", 1)[0].lower(),
        service_name=str(service_name) if service_name else "Unknown",
        host=str(host) if host else None,
    )

def _bound_sockets(client: apprise.Apprise) -> None:
    """Clamp every loaded plugin's socket timeouts to this module's ceilings.

    Set on the plugin objects rather than appended to the URL as ``?cto=&rto=``:
    a destination can already carry a query string and a fragment
    (``slack://tok/#channel``), so concatenating onto an operator's URL risks
    corrupting one that worked. Clamped rather than overwritten, so a shorter
    timeout survives.

    The bound is per socket operation, not per send, because Apprise may retry a
    plugin. Closing that gap would mean owning Apprise's retry loop.
    """
    for server in client:
        for attribute, ceiling in (
            ("socket_connect_timeout", SOCKET_CONNECT_TIMEOUT_SECONDS),
            ("socket_read_timeout", SOCKET_READ_TIMEOUT_SECONDS),
        ):
            current = getattr(server, attribute, None)
            if isinstance(current, int | float) and current > ceiling:
                setattr(server, attribute, ceiling)


async def send_alert(destination: str, *, title: str, body: str) -> AlertDispatchResult:
    """Deliver one alert, reporting failure rather than raising it.

    Never raises for a delivery problem, which is the contract every caller
    depends on: the evaluator runs inside a periodic task that must survive a
    destination outage, and a rule pointed at a decommissioned webhook must not
    stop the rules after it in the same tick from being evaluated.

    ``asyncio.CancelledError`` is re-raised rather than swallowed, so shutdown
    still cancels a send in flight. Everything else is reported.
    """
    try:
        client = apprise.Apprise(asset=_ASSET)
        if not client.add(destination):
            return AlertDispatchResult(delivered=False, detail="Apprise rejected the destination URL")
        _bound_sockets(client)
        sent = await asyncio.wait_for(
            client.async_notify(body=body, title=title),
            timeout=SEND_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        logger.warning("Alert delivery timed out after %ss", SEND_TIMEOUT_SECONDS)
        return AlertDispatchResult(delivered=False, detail=f"Timed out after {SEND_TIMEOUT_SECONDS:.0f}s")
    except Exception as exc:
        # Apprise surfaces a plugin's own exceptions, so the type is whatever
        # the transport raised. The class name is logged and the message is not:
        # a plugin that echoes its URL into the message would otherwise put the
        # token in our log.
        logger.warning("Alert delivery failed: %s", type(exc).__name__)
        return AlertDispatchResult(delivered=False, detail=f"Delivery failed ({type(exc).__name__})")

    # ``async_notify`` returns None when no server matched the notification (an
    # empty client, or every server filtered out by tag), which is a
    # configuration problem rather than a transport one and reads as falsy.
    if not sent:
        return AlertDispatchResult(delivered=False, detail="The destination rejected the notification")
    return AlertDispatchResult(delivered=True)
