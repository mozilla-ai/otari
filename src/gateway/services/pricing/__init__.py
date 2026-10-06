"""The pricing domain owns the deployment price list and the organization rate overrides.

This package holds the deployment price list's service. The ladder a request
settles on, the organization override surface and the upstream price snapshots
are still in the flat modules beside `services/`, and move here when the
domain's migration reaches them (`docs/domains.md`).
"""

from gateway.services.pricing._deployment_pricing_service import DeploymentPricingService

__all__ = ["DeploymentPricingService"]
