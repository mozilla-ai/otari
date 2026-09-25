from fastapi import APIRouter, Depends, FastAPI

from gateway.api.deps import require_capability
from gateway.api.routes import (
    admin,
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
    embeddings,
    files,
    health,
    hooks,
    hosted_mode,
    hybrid_mode,
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
    providers,
    rerank,
    responses,
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
    web_search_backend,
    workspace_activation,
    workspace_code_execution_policy,
    workspace_mcp_servers,
    workspace_member_budget_policies,
    workspace_web_search,
    workspaces,
)
from gateway.container import Container
from gateway.core.config import API_ROOT, OTLP_ROOT, GatewayConfig
from gateway.core.deployment import Plane, RouterMount, deployment_for
from gateway.core.feature import CoreFeature


def register_routers(app: FastAPI, config: GatewayConfig) -> None:
    """Mount Otari's own routers, then whatever the bootstrap contributed.

    One aggregate router carries the API, so its mount prefix has one owner.
    Mount order on that router matters. The hybrid and hosted stubs are
    ``{path:path}`` catch-alls and the first matching route wins, so the
    stubs go last. A stub mounted earlier would answer a path that a
    contributed router serves.
    """
    api = APIRouter(prefix=API_ROOT)
    deployment = deployment_for(config)
    _register_core_routers(api, config, app.state.enabled_features)
    _register_contributed_routers(api, app.state.container)
    if not deployment.supports(Plane.CONTROL):
        api.include_router(hybrid_mode.router)
    elif not deployment.supports(Plane.DATA):
        # A hosted control plane holds no data plane, so the inference prefixes
        # get catch-all stubs. A contributed route still wins: an overlay that
        # adds one has made a choice, and a fallback does not overrule it.
        api.include_router(hosted_mode.router)
    app.include_router(api)
    # OTLP is a sibling namespace, not a child of the API root: OTel owns the
    # /v1/{traces,logs,metrics} tail, so an exporter pointed at `{origin}/otlp`
    # appends it unaided. Standalone and hosted only, like the rest of the
    # management surface; a hybrid data plane stores no telemetry.
    if deployment.supports(Plane.CONTROL):
        app.include_router(otlp.router, prefix=OTLP_ROOT)


def _register_contributed_routers(api: APIRouter, container: Container) -> None:
    """Mount the routers this build's bootstrap contributed, each behind its gate.

    The additive half of the extension seam: an overlay records a router on the
    container and Otari mounts it, gated on the capability it names. RouterMount in
    both modes, because an overlay may extend the data plane as readily as the
    management plane.

    A contribution inherits the aggregate's prefix, so it declares its resource
    only.
    """
    for contribution in container.router_contributions():
        api.include_router(
            contribution.router,
            dependencies=[Depends(require_capability(contribution.capability))],
        )


# Every core router, in mount order, with the planes a deployment must serve
# for it to answer. Order is load-bearing in places: where two routers share a
# prefix, the one whose path could sit behind the other's catch-all goes first,
# and the comments below say which.
#
# ``Plane.DATA`` alone is a data-plane route a hybrid gateway answers.
# ``Plane.DATA | Plane.CONTROL`` is one that also reads a local control plane,
# so a hybrid gateway does not answer it.
_CORE_ROUTERS: tuple[RouterMount, ...] = (
    RouterMount(chat.router, Plane.DATA),
    RouterMount(health.router),
    # A browser learns which deployment it reached from this, so every
    # deployment answers it.
    RouterMount(bootstrap.router),
    # Absent rather than refusing where the deployment holds neither a search
    # credential nor a token to recognize its own gateway by.
    RouterMount(
        web_search_backend.router,
        when=lambda config: config.web_search_provider_configured() and bool(config.web_search_backend_token),
    ),
    RouterMount(messages.router, Plane.DATA),
    RouterMount(responses.router, Plane.DATA),
    # Stateless: a hybrid gateway authenticates through the platform's MCP
    # resolver rather than a local database.
    RouterMount(mcp.router, Plane.DATA),
    # Evaluates only the policy and evidence the caller sent in the same
    # request, so it needs no tenancy, no provider and no database.
    RouterMount(hooks.router),
    RouterMount(admin.router, Plane.CONTROL),
    RouterMount(auth_session.router, Plane.CONTROL),
    RouterMount(auth_password.router, Plane.CONTROL),
    RouterMount(auth_profile.router, Plane.CONTROL),
    RouterMount(auth_signup.router, Plane.CONTROL),
    RouterMount(auth_password_reset.router, Plane.CONTROL),
    RouterMount(auth_webauthn.router, Plane.CONTROL),
    RouterMount(auth_oauth.router, Plane.CONTROL),
    # Data-plane routes that also read a local control plane, because none of
    # them has hybrid handling. ``files`` is among them because an upload exists
    # to be referenced from a completion or a batch.
    RouterMount(embeddings.router, Plane.DATA | Plane.CONTROL),
    RouterMount(images.router, Plane.DATA | Plane.CONTROL),
    RouterMount(audio.router, Plane.DATA | Plane.CONTROL),
    RouterMount(files.router, Plane.DATA | Plane.CONTROL),
    RouterMount(rerank.router, Plane.DATA | Plane.CONTROL),
    RouterMount(search.router, Plane.DATA | Plane.CONTROL),
    RouterMount(batches.router, Plane.DATA | Plane.CONTROL),
    RouterMount(moderations.router, Plane.DATA | Plane.CONTROL),
    # Discovery rather than dispatch, so these are not operator-gated. The
    # operator router goes first: its /discoverable and /metadata would
    # otherwise sit behind the catalog router's {model_id:path} catch-all.
    RouterMount(models.operator_router, Plane.CONTROL),
    RouterMount(models.catalog_router, Plane.CONTROL),
    # The same merged catalog, folded by model for a chooser rather than
    # listed flat for an SDK.
    RouterMount(catalog.router, Plane.CONTROL),
    RouterMount(catalog.operator_router, Plane.CONTROL),
    # Reads the management surface and dispatches a completion, so it needs a
    # session a hybrid gateway cannot offer. A hosted control plane forwards
    # the completion to ``data_plane_url``, which is why this prefix is absent
    # from ``hosted_mode.DATA_PLANE_PREFIXES``: a stub there would shadow it.
    RouterMount(playground.router, Plane.CONTROL),
    # Split from the operator router because the organization provider-key
    # form is read by a tenant's owners and admins, who operate nothing.
    RouterMount(providers.catalog_router, Plane.CONTROL),
    RouterMount(providers.router, Plane.CONTROL),
    RouterMount(keys.router, Plane.CONTROL),
    RouterMount(users.router, Plane.CONTROL),
    RouterMount(organizations.router, Plane.CONTROL),
    RouterMount(organization_budgets.budgets_router, Plane.CONTROL),
    RouterMount(organization_budgets.ceilings_router, Plane.CONTROL),
    RouterMount(organization_pricing.router, Plane.CONTROL),
    # Definitions first: a mandate can point at a definition, never the other
    # way round. Two prefixes because what a check is and where it runs have
    # different keys.
    RouterMount(organization_guardrail_definitions.router, Plane.CONTROL),
    RouterMount(organization_guardrails.router, Plane.CONTROL),
    # The tenant-scoped read over the rows the operator usage router serves.
    # Grouped by what it is scoped to, which is what decides who may call it.
    RouterMount(organization_usage.router, Plane.CONTROL),
    # The tenant-scoped half of the routing and alias tables, grouped for the
    # same reason.
    RouterMount(organization_routing.policies_router, Plane.CONTROL),
    RouterMount(organization_routing.aliases_router, Plane.CONTROL),
    # The caller's own keys, in workspaces they may see. The deployment-wide
    # ``keys.router`` keeps its operator gate.
    RouterMount(organization_keys.router, Plane.CONTROL),
    RouterMount(workspaces.router, Plane.CONTROL),
    RouterMount(invitations.router, Plane.CONTROL),
    RouterMount(workspace_member_budget_policies.router, Plane.CONTROL),
    RouterMount(workspace_activation.router, Plane.CONTROL),
    RouterMount(workspace_mcp_servers.router, Plane.CONTROL),
    RouterMount(workspace_code_execution_policy.router, Plane.CONTROL),
    RouterMount(workspace_web_search.router, Plane.CONTROL),
    RouterMount(org_provider_keys.org_router, Plane.CONTROL),
    RouterMount(org_provider_keys.workspace_router, Plane.CONTROL),
    RouterMount(budgets.router, Plane.CONTROL),
    RouterMount(scoped_budgets.router, Plane.CONTROL),
    RouterMount(overview.router, Plane.CONTROL),
    RouterMount(aliases.router, Plane.CONTROL),
    RouterMount(routing.router, Plane.CONTROL),
    RouterMount(routing_memory.router, Plane.CONTROL),
    # Both prefixed /pricing, split by who may call them; operator first, so
    # its DELETE /{model_key:path} does not sit behind the catalog catch-all.
    RouterMount(pricing.operator_router, Plane.CONTROL),
    RouterMount(pricing.catalog_router, Plane.CONTROL),
    # Both prefixed /usage. POST /external-events authenticates with an API
    # key rather than operator standing, so it is mounted on its own router.
    RouterMount(usage.operator_router, Plane.CONTROL),
    RouterMount(usage.ingest_router, Plane.CONTROL),
    RouterMount(agent_telemetry.router, Plane.CONTROL),
    RouterMount(settings.router, Plane.CONTROL),
    RouterMount(mail.router, Plane.CONTROL),
    RouterMount(maintenance_mode.router, Plane.CONTROL),
    # All three prefixed /tool-settings, split by who may call them. Operator
    # first, matching the pairs above, though none ends with a catch-all.
    RouterMount(tool_settings.operator_router, Plane.CONTROL),
    RouterMount(tool_settings.reader_router, Plane.CONTROL),
    RouterMount(tool_settings.catalog_router, Plane.CONTROL),
    RouterMount(search_tools.router, Plane.CONTROL),
    RouterMount(tools.router, Plane.CONTROL),
)


def _register_core_routers(api: APIRouter, config: GatewayConfig, enabled_features: tuple[CoreFeature, ...]) -> None:
    """Mount every router in ``_CORE_ROUTERS`` that ``config``'s deployment can serve."""
    deployment = deployment_for(config)
    for mount in _CORE_ROUTERS:
        if mount.applies_to(deployment, config):
            api.include_router(mount.router)
    # No capability gate: a listed feature is part of this build. Control
    # plane only, because the registry has no shape for a data-plane feature.
    if deployment.supports(Plane.CONTROL):
        for feature in enabled_features:
            for router in feature.routers(config):
                api.include_router(router)
