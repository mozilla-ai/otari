"""The providers domain owns a deployment's provider credentials.

This package holds the organization-scoped side: the models a BYO key offers,
which of them the runtime serves, and the rate each is charged at. The
deployment's own instances are still in the flat modules beside `services/`,
and move here when the domain's migration reaches them (`docs/domains.md`).
"""

from gateway.services.providers._org_provider_model_service import OrgProviderModelService

__all__ = ["OrgProviderModelService"]
