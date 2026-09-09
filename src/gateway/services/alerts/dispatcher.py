"""Delivering one alert to one Apprise URL, without ever failing a caller.

This module knows nothing about budgets. It takes a destination URL, a title and
a body, and reports whether the send worked.

**Why Apprise.** An alert destination is the kind of thing every operator wants
spelled differently: Slack here, PagerDuty there, a plain webhook into somebody
else's internal tooling. Apprise turns that into one string, so the schema is one
column, and a destination this codebase has never heard of needs no adapter and
no migration. That is also why there is no port here, unlike
``ports/growth_signal_port.py``: a port exists to keep a *vendor* choice out of
the core, and Apprise is itself the vendor-neutral layer, so wrapping it in one
would be a seam with a single possible implementation on either side.

**Threads, not the event loop.** ``Apprise.async_notify`` is a real coroutine but
its plugins are synchronous ``requests`` calls that it offloads with
``run_in_executor(None, ...)`` (Apprise's ``plugins/base.py``). So a send never
blocks the loop, but it does occupy a default-executor thread for the length of
an HTTP round trip. Bounded by :data:`SEND_TIMEOUT_SECONDS` and by the evaluator
never fanning out more than one send per rule per tick, which is what keeps a
slow destination from starving the executor the rest of the process shares.

**Logging.** The destination is a credential: a ``slack://`` URL embeds a bot
token and a ``json://`` one can embed basic-auth. Nothing here passes a URL to
the gateway logger; callers log the stored redaction instead. Apprise's own
logger redacts credentials (its ``AppriseAsset.secure_logging`` defaults true and
its CWE-312 handling routes failures through ``cwe312_url``), and
:data:`_ASSET` sets that flag explicitly rather than inheriting it, so the
guarantee survives a change to Apprise's defaults.
"""

import asyncio
from dataclasses import dataclass
from typing import Final

import apprise

from gateway.log_config import logger

# How long this coroutine waits for a send before giving up on it.
SEND_TIMEOUT_SECONDS: Final = 15.0

# The socket bounds that make the number above mean something.
#
# ``asyncio.wait_for`` cancels the *await*, not the work: ``async_notify`` runs
# each plugin's blocking ``requests`` call through ``run_in_executor``, and a
# thread already inside a socket read cannot be cancelled. So the timeout alone
# stops this coroutine waiting while leaving the thread occupied, and the thing
# that actually bounds the thread is the socket timeout the plugin uses.
#
# Apprise defaults both to 4 seconds, but they are per-plugin values an operator
# can raise from the destination URL itself (``?cto=600&rto=600``), which would
# park a shared executor thread for ten minutes per tick. :func:`_bound_sockets`
# clamps them instead of trusting the URL.
SOCKET_CONNECT_TIMEOUT_SECONDS: Final = 5.0
SOCKET_READ_TIMEOUT_SECONDS: Final = 10.0

# Explicit rather than inherited: see the module docstring. ``secure_logging``
# is what keeps a bot token out of Apprise's own log records.
_ASSET: Final = apprise.AppriseAsset(secure_logging=True)


class UnsupportedAlertDestinationError(ValueError):
    """Apprise cannot parse the destination, or its schema is not built in.

    Raised by :func:`parse_destination` at rule-write time, so an operator
    learns their URL is unusable while they are looking at the form rather than
    when a budget crosses a threshold three weeks later.
    """


@dataclass(frozen=True)
class AlertDispatchResult:
    """Whether one send succeeded, and a short reason when it did not.

    ``detail`` is stored on the delivery row and shown to the operator, so it
    says what happened without naming the destination: the row already carries
    the redaction, and the reason is the part that is missing.
    """

    delivered: bool
    detail: str | None = None


def parse_destination(destination: str) -> str:
    """Check a destination is one Apprise can deliver to, and name its schema.

    Returns the plugin's service name, which the create path stores nothing of
    but the API echoes back so a form can confirm it understood ``slack://`` as
    Slack.

    ``suppress_exceptions`` keeps a malformed URL from surfacing as whatever the
    plugin's constructor happened to raise, so every rejection reaches the caller
    as one domain error.
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
    service_name = getattr(plugin, "service_name", None)
    return str(service_name) if service_name else "Unknown"


def _bound_sockets(client: apprise.Apprise) -> None:
    """Clamp every loaded plugin's socket timeouts to this module's ceilings.

    Set on the plugin objects rather than appended to the URL as ``?cto=&rto=``,
    which is what an Apprise reader would reach for first: a destination can
    already carry a query string and a fragment (``slack://tok/#channel``), so
    concatenating parameters onto an operator's URL risks corrupting a
    destination that worked. The attributes are ``URLBase``'s own and are what
    ``cto``/``rto`` set anyway.

    Clamped, not overwritten, so an operator who asked for a *shorter* timeout
    keeps it and only an unreasonably long one is brought down.

    The bound is per socket operation rather than per send: Apprise may retry a
    plugin, so the guarantee here is that no single operation parks a thread
    indefinitely, not that a send finishes within
    :data:`SEND_TIMEOUT_SECONDS`. Closing the remaining gap would mean owning
    Apprise's retry loop, which is not worth the coupling.

    Defensive about the attributes existing: they come from ``URLBase``, so
    every built-in plugin has them, but an entry loaded from a custom plugin
    path need not, and a missing one must not turn one bad rule into a failed
    pass.
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
