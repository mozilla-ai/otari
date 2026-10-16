"""The rate-limits domain owns the ``rate_limits`` rules an operator manages from the dashboard.

Enforcement stays in ``gateway.rate_limit``, which reads ``config.rate_limits``
on every request; this package keeps the stored rules in that list.
"""

from gateway.services.rate_limits._overlay import load_rate_limit_rules_at_startup, run_rate_limit_refresher
from gateway.services.rate_limits._service import RateLimitService

__all__ = ["RateLimitService", "load_rate_limit_rules_at_startup", "run_rate_limit_refresher"]
