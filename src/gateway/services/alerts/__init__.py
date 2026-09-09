"""Outbound alerts: what an organization is told about its budgets, and where.

Three modules, split by what they each know:

- :mod:`gateway.services.alerts.dispatcher` knows how to deliver a message to an
  Apprise URL, and nothing about budgets.
- :mod:`gateway.services.alerts.evaluator` knows which ceilings have crossed a
  threshold, claims the right to alert on each one, and asks the dispatcher to
  send. It is the periodic worker ``gateway.main`` starts.
- ``gateway.services.tenancy.organization_alert_service`` is the CRUD surface a
  tenant configures rules through, and lives with the other tenancy services
  because it shares their role gate and their error family.

**Why a periodic evaluator rather than a hook on the request path.** The
budget-refusal sites in ``services/budget_service.py`` are inside the
reserve/settle lifecycle, which ``AGENTS.md`` and the backend standards both
single out as load-bearing: a reservation that never settles leaks and
permanently shrinks a user's budget. Alerting from there would add a query and a
fan-out to the one path in the codebase least able to absorb either, to buy
latency that a budget warning does not need. Reading the state on a timer costs
the request path nothing, treats "warning" and "exceeded" as the same
computation, and additionally catches a ceiling that crossed while the
destination was misconfigured, which an event hook fires once and loses.
"""

from gateway.services.alerts.dispatcher import (
    AlertDispatchResult,
    UnsupportedAlertDestinationError,
    parse_destination,
    send_alert,
)
from gateway.services.alerts.evaluator import (
    ALERT_EVALUATION_INTERVAL_SECONDS,
    evaluate_budget_alerts,
    run_budget_alert_evaluator,
)

__all__ = [
    "ALERT_EVALUATION_INTERVAL_SECONDS",
    "AlertDispatchResult",
    "UnsupportedAlertDestinationError",
    "evaluate_budget_alerts",
    "parse_destination",
    "run_budget_alert_evaluator",
    "send_alert",
]
