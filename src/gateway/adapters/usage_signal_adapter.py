"""Null Object usage-signal adapter, for a build that reports its traffic to nobody.

Satisfies :class:`gateway.ports.usage_signal_port.UsageSignalPort` with the
honest answer for a deployment whose usage table is the whole record: nothing
is scheduled, so no HTTP call, background task, or vendor credential is ever
reached for want of an analytics integration.
"""

from gateway.ports.usage_signal_port import RecordedUsage, UsageSignalPort


class NullUsageSignalAdapter(UsageSignalPort):
    """``UsageSignalPort`` adapter for a build with no analytics vendor."""

    async def usage_recorded(self, usage: RecordedUsage) -> None:
        return None
