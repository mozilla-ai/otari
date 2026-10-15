"""Notification of each usage row a request settles, for whichever build wants it.

The seam between the core and whichever build reacts to a request having been
served: product analytics, a CRM snapshot, an activation milestone. The core
keeps its own record of every request in the usage table and needs nothing
told to anyone, and an operator running Otari should not have their traffic
reported to a vendor the deployment never chose. So the core adapter is a Null
Object, and an overlay binds one that fans the facts out to the vendors it runs.

The method is fire-and-forget: it must return promptly and never raise, and it
schedules any slow or network work itself. It is called on the settlement path
of the request it describes, so a slow adapter is a slow request and a vendor
outage must never reach the caller. That is why it returns nothing: there is
nothing for a caller to act on, only something to fire and forget.

It is called once per usage row, after the row was handed to the usage log
writer: with the default ``single`` strategy the row is committed by then, and
with ``batch`` it is only queued, so it may still be dropped. A request that
failed over writes one row per attempt and is therefore reported once per
attempt; ``request_group_id`` ties them together. A hybrid gateway writes no
usage rows here, so nothing is reported on one; its usage goes to its control
plane instead.

The facts carry no organization, which the row does not record (an adapter
resolves it from ``workspace_id``), and no caller content: no error message,
no request tags, nothing a request or a response said.

Stability: this interface is not frozen while Otari is pre-1.0.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True)
class RecordedUsage:
    """One settled usage row, as the port reports it.

    Owned by the port rather than taken from the ORM row, so an adapter never
    reads the table's shape and only these facts cross the seam.
    """

    usage_id: str
    request_group_id: str | None
    timestamp: datetime
    workspace_id: uuid.UUID
    user_id: str | None
    api_key_id: str | None
    provider: str | None
    model: str
    # The usage-log label of what was called (``/v1/chat/completions``, the
    # Playground's own label, ...): an identifier, not the URL that served it.
    endpoint: str
    # ``gateway`` for a request this deployment served itself.
    source: str
    # ``success``, ``error`` or ``absorbed`` (a failed attempt a routing policy
    # recovered from on a later candidate).
    status: str
    status_code: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    # USD, exact; ``None`` when nothing priced the row.
    cost: Decimal | None
    latency_ms: int | None
    # Served here from somebody's own code rather than from the dashboard's
    # Playground: the traffic the setup guide counts as integrating Otari.
    integration_traffic: bool


class UsageSignalPort(Protocol):
    """What a build does when a request settles a usage row, if anything."""

    async def usage_recorded(self, usage: RecordedUsage) -> None:
        """Notify that a usage row was handed to the writer.

        Must return promptly and never raise; see the module docstring.
        """
        ...
