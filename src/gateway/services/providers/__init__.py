"""The providers domain owns a deployment's provider credentials.

This package holds the organization-scoped side (the models a BYO key offers,
which of them the runtime serves, and the rate each is charged at) and the
endpoints a workspace or one of its users owns. The deployment's own instances
are still in the flat modules beside `services/`, and move here when the
domain's migration reaches them (`docs/domains.md`).
"""

from gateway.services.providers._org_provider_model_service import OrgProviderModelService
from gateway.services.providers._owned_endpoint_network import owned_endpoint_http_client
from gateway.services.providers._provider_endpoint_cache import (
    OwnedEndpoint,
    cached_owned_endpoint,
    load_provider_endpoints_at_startup,
    refresh_provider_endpoint_cache,
    reset_provider_endpoint_cache,
    run_provider_endpoint_refresher,
)
from gateway.services.providers._provider_endpoint_service import ProviderEndpointService

__all__ = [
    "OrgProviderModelService",
    "OwnedEndpoint",
    "ProviderEndpointService",
    "cached_owned_endpoint",
    "load_provider_endpoints_at_startup",
    "owned_endpoint_http_client",
    "refresh_provider_endpoint_cache",
    "reset_provider_endpoint_cache",
    "run_provider_endpoint_refresher",
]
