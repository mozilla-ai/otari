from fastapi import APIRouter, Depends, FastAPI

from gateway.api.deps import require_capability
from gateway.api.routes import (
    admin,
    agent_recommenders,
    agent_routing,
    agent_telemetry,
    aliases,
    audio,
    auth_oauth,
    auth_password,
    auth_password_reset,
    auth_profile,
    auth_session,
    auth_signup,
    auth_webauthn,
    batches,
    bootstrap,
    budgets,
    catalog,
    chat,
    country_database,
    decisions,
    embeddings,
    end_users,
    files,
    health,
    hook_guardrails,
    hooks,
    hosted_providers,
    identity,
    images,
    invitations,
    keys,
    mail,
    maintenance_mode,
    mcp,
    messages,
    models,
    moderations,
    org_provider_keys,
    organization_budgets,
    organization_guardrail_definitions,
    organization_guardrails,
    organization_keys,
    organization_pricing,
    organization_routing,
    organization_usage,
    organizations,
    otlp,
    overview,
    playground,
    pricing,
    provider_endpoints,
    providers,
    rate_limits,
    rerank,
    responses,
    retired_root,
    routing,
    routing_memory,
    scoped_budgets,
    search,
    search_tools,
    settings,
    tool_settings,
    tools,
    usage,
    users,
    web_search_keys,
    workspace_activation,
    workspace_code_execution_policy,
    workspace_mcp_servers,
    workspace_member_budget_policies,
    workspace_web_search,
    workspaces,
)
from gateway.container import Container
from gateway.core.config import API_ROOT, OTLP_ROOT, GatewayConfig
from gateway.core.feature import CoreFeature


def register_routers(app: FastAPI, config: GatewayConfig) -> None:
    """Mount Otari's own routers, then whatever the bootstrap contributed.

    One aggregate router carries the API, so its mount prefix has one owner.
    """
    api = APIRouter(prefix=API_ROOT)
    _register_core_routers(api, config, app.state.enabled_features)
    _register_contributed_routers(api, app.state.container)
    app.include_router(api)
    # OTLP is a sibling namespace, not a child of the API root: OTel owns the
    # /v1/{traces,logs,metrics} tail, so an exporter pointed at `{origin}/otlp`
    # appends it unaided.
    app.include_router(otlp.router, prefix=OTLP_ROOT)
    # The pre-/api root, refused with a pointer to its successor.
    app.include_router(retired_root.router)


def _register_contributed_routers(api: APIRouter, container: Container) -> None:
    """Mount the routers this build's bootstrap contributed, each behind its gate.

    The additive half of the extension seam: an overlay records a router on the
    container and Otari mounts it, gated on the capability it names.

    A contribution inherits the aggregate's prefix, so it declares its resource
    only.
    """
    for contribution in container.router_contributions():
        api.include_router(
            contribution.router,
            dependencies=[Depends(require_capability(contribution.capability))],
        )


# Every core router, in mount order. Order is load-bearing in places: where two
# routers share a prefix, the one whose path could sit behind the other's
# catch-all goes first, and the comments below say which.
_CORE_ROUTERS: tuple[APIRouter, ...] = (
    chat.router,
    health.router,
    bootstrap.router,
    messages.router,
    responses.router,
    mcp.router,
    # Evaluates only the policy and evidence the caller sent in the same
    # request, so it needs no tenancy, no provider and no database.
    hooks.router,
    hook_guardrails.router,
    admin.router,
    auth_session.router,
    auth_password.router,
    auth_profile.router,
    auth_signup.router,
    auth_password_reset.router,
    identity.router,
    # Absent rather than refusing where the deployment turned passkeys off: a
    # ceremony route that answers 503 still says passkeys exist here.
    auth_webauthn.router,
    auth_oauth.router,
    embeddings.router,
    images.router,
    audio.router,
    rerank.router,
    moderations.router,
    files.router,
    decisions.router,
    # One decision call, billed here like a decisions call. Open to any key,
    # since the harness dispatches the subagent to its own provider.
    agent_routing.router,
    search.router,
    batches.router,
    # Discovery rather than dispatch, so these are not operator-gated. The
    # operator router goes first: its /discoverable and /metadata would
    # otherwise sit behind the catalog router's {model_id:path} catch-all.
    models.operator_router,
    models.catalog_router,
    # The same merged catalog, folded by model for a chooser rather than
    # listed flat for an SDK.
    catalog.router,
    catalog.operator_router,
    playground.router,
    # Split from the operator router because the organization provider-key
    # form is read by a tenant's owners and admins, who operate nothing.
    providers.catalog_router,
    providers.router,
    # The deployment's own upstream credentials and the models it offers on
    # them. Operator-only, and answered 404 to anyone else like /admin.
    hosted_providers.router,
    # The decision models that pick a subagent's model, and their prices.
    # Operator-only, and answered 404 to anyone else like /admin.
    agent_recommenders.router,
    keys.router,
    # A service key managing its own end users, rather than the operator gate
    # ``keys.router`` holds over every key's.
    end_users.router,
    end_users.budgets_router,
    users.router,
    organizations.router,
    organization_budgets.budgets_router,
    organization_budgets.ceilings_router,
    organization_pricing.router,
    # Definitions first: a mandate can point at a definition, never the other
    # way round. Two prefixes because what a check is and where it runs have
    # different keys.
    organization_guardrail_definitions.router,
    organization_guardrails.router,
    # The tenant-scoped read over the rows the operator usage router serves.
    # Grouped by what it is scoped to, which is what decides who may call it.
    organization_usage.router,
    # The tenant-scoped half of the routing and alias tables, grouped for the
    # same reason.
    organization_routing.policies_router,
    organization_routing.aliases_router,
    organization_routing.fallbacks_router,
    # The caller's own keys, in workspaces they may see. The deployment-wide
    # ``keys.router`` keeps its operator gate.
    organization_keys.router,
    workspaces.router,
    invitations.router,
    workspace_member_budget_policies.router,
    workspace_activation.router,
    workspace_mcp_servers.router,
    workspace_code_execution_policy.router,
    workspace_web_search.router,
    web_search_keys.org_router,
    web_search_keys.workspace_router,
    org_provider_keys.org_router,
    org_provider_keys.workspace_router,
    budgets.router,
    scoped_budgets.router,
    overview.router,
    aliases.router,
    provider_endpoints.router,
    routing.router,
    routing_memory.router,
    # Both prefixed /pricing, split by who may call them; operator first, so
    # its DELETE /{model_key:path} does not sit behind the catalog catch-all.
    pricing.operator_router,
    pricing.catalog_router,
    # Both prefixed /usage. POST /external-events and GET /requests/{id}
    # authenticate with an API key rather than operator standing, so they are
    # mounted on their own router.
    usage.operator_router,
    usage.key_router,
    agent_telemetry.router,
    settings.router,
    rate_limits.router,
    mail.router,
    maintenance_mode.router,
    country_database.router,
    # All three prefixed /tool-settings, split by who may call them. Operator
    # first, matching the pairs above, though none ends with a catch-all.
    tool_settings.operator_router,
    tool_settings.reader_router,
    tool_settings.catalog_router,
    # Both prefixed /search-tools. The catalog first, because its one route is a
    # fixed path and the operator's end in a path parameter.
    search_tools.catalog_router,
    search_tools.router,
    tools.router,
)


def _register_core_routers(api: APIRouter, config: GatewayConfig, enabled_features: tuple[CoreFeature, ...]) -> None:
    """Mount every router in ``_CORE_ROUTERS``, then each enabled feature's."""
    for router in _CORE_ROUTERS:
        if router is auth_webauthn.router and not config.passkeys_enabled:
            continue
        api.include_router(router)
    # No capability gate: a listed feature is part of this build.
    for feature in enabled_features:
        for feature_router in feature.routers(config):
            api.include_router(feature_router)
