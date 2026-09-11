"""The plugin's one Prometheus counter, on Otari's registry.

Registered on ``gateway.metrics.REGISTRY`` rather than a registry of its own,
because ``/metrics`` serves that one and an operator scraping it should see the
alert counters beside everything else. The counter is defined here rather than
in core for the reason the tables are: a metric nobody can increment is noise
on a deployment that has not installed this plugin.
"""

from prometheus_client import Counter

from gateway.metrics import REGISTRY

# Labeled by outcome as well as kind, because "the alert fired" and "the alert
# arrived" are the two different things an operator needs to tell apart: a
# destination that has quietly stopped accepting deliveries looks identical to a
# budget that never crossed a threshold unless the failures are counted
# separately.
ALERT_DELIVERIES = Counter(
    "gateway_alert_deliveries",
    "Total number of budget alerts dispatched, by kind and outcome",
    ["kind", "outcome"],
    registry=REGISTRY,
)


def record_alert_delivery(*, kind: str, delivered: bool) -> None:
    """Record one dispatched budget alert and whether it reached its destination."""
    ALERT_DELIVERIES.labels(kind=kind, outcome="delivered" if delivered else "failed").inc()
