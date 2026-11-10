"""The providers domain owns a deployment's provider credentials.

This package holds the organization-scoped side (the models a BYO key offers,
which of them the runtime serves, and the rate each is charged at) and the
endpoints a workspace or one of its users owns, and the hosted providers the
deployment serves on its own credentials. The deployment's config instances
are still in the flat modules beside `services/`, and move here when the
domain's migration reaches them (`DOMAINS.md`).
"""

from gateway.services.providers._country_gate import (
    CountryGate,
    CountryNotServedError,
    ensure_hosted_country,
    ensure_provider_country,
)
from gateway.services.providers._hosted_catalog_update import (
    HostedCatalogUpdate,
    PriceUpdates,
    ScheduledRun,
    run_catalog_updater,
)
from gateway.services.providers._hosted_credentials import last4, normalize_api_base, public_api_base
from gateway.services.providers._hosted_name_index import (
    hosted_implementation,
    load_hosted_names_at_startup,
    reset_hosted_name_index,
    run_hosted_name_refresher,
)
from gateway.services.providers._hosted_provider_service import (
    HostedProviderService,
    LiveByoPairs,
    OrganizationExists,
    ResolvedHostedProvider,
)
from gateway.services.providers._hosted_runtime import (
    UnknownHostedUpstreamError,
    dispatch_on_hosted_credential,
    hosted_credential_for,
    hosted_models_for,
    organization_to_serve,
    serve_on_hosted_provider,
)
from gateway.services.providers._hosted_sources import HostedSources, refuse_unsafe_configured_endpoints
from gateway.services.providers._missing_credential import (
    missing_provider_credential,
    unpriced_model_refusal,
)
from gateway.services.providers._org_provider_model_service import OrgProviderModelService
from gateway.services.providers._owned_endpoint_network import owned_endpoint_http_client, owned_endpoint_sdk_client
from gateway.services.providers._platform_connect import (
    PLATFORM_PROVIDER_NAME,
    PlatformClient,
    PlatformConnectService,
    new_platform_connect_rate_limiter,
)
from gateway.services.providers._provider_clients import (
    acompletion,
    amessages,
    aresponses,
    provider_for,
)
from gateway.services.providers._provider_endpoint_cache import (
    OwnedEndpoint,
    cached_owned_endpoint,
    load_provider_endpoints_at_startup,
    refresh_provider_endpoint_cache,
    reset_provider_endpoint_cache,
    run_provider_endpoint_refresher,
)
from gateway.services.providers._provider_endpoint_service import ProviderEndpointService
from gateway.services.providers._workspace_hosted_access import (
    WorkspaceHostedAccessDeniedError,
    WorkspaceHostedAccessService,
    load_workspace_hosted_access_at_startup,
    narrow_hosted_models,
    refresh_workspace_hosted_access_cache,
    reset_workspace_hosted_access_cache,
    run_workspace_hosted_access_refresher,
    workspace_may_use_hosted,
    workspace_requires_zero_data_retention,
)

__all__ = [
    "CountryGate",
    "CountryNotServedError",
    "HostedCatalogUpdate",
    "HostedProviderService",
    "HostedSources",
    "LiveByoPairs",
    "OrganizationExists",
    "OrgProviderModelService",
    "OwnedEndpoint",
    "PLATFORM_PROVIDER_NAME",
    "PlatformClient",
    "PlatformConnectService",
    "PriceUpdates",
    "ProviderEndpointService",
    "ResolvedHostedProvider",
    "ScheduledRun",
    "UnknownHostedUpstreamError",
    "WorkspaceHostedAccessDeniedError",
    "WorkspaceHostedAccessService",
    "acompletion",
    "amessages",
    "aresponses",
    "cached_owned_endpoint",
    "dispatch_on_hosted_credential",
    "ensure_hosted_country",
    "ensure_provider_country",
    "hosted_credential_for",
    "hosted_implementation",
    "last4",
    "hosted_models_for",
    "load_hosted_names_at_startup",
    "load_provider_endpoints_at_startup",
    "load_workspace_hosted_access_at_startup",
    "missing_provider_credential",
    "narrow_hosted_models",
    "new_platform_connect_rate_limiter",
    "normalize_api_base",
    "organization_to_serve",
    "owned_endpoint_http_client",
    "owned_endpoint_sdk_client",
    "provider_for",
    "public_api_base",
    "refresh_provider_endpoint_cache",
    "refresh_workspace_hosted_access_cache",
    "refuse_unsafe_configured_endpoints",
    "reset_hosted_name_index",
    "reset_provider_endpoint_cache",
    "reset_workspace_hosted_access_cache",
    "run_catalog_updater",
    "run_hosted_name_refresher",
    "run_provider_endpoint_refresher",
    "run_workspace_hosted_access_refresher",
    "serve_on_hosted_provider",
    "unpriced_model_refusal",
    "workspace_may_use_hosted",
    "workspace_requires_zero_data_retention",
]
