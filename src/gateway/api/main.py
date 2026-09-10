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
    auth_session,
    auth_signup,
    auth_webauthn,
    batches,
    bootstrap,
    budgets,
    chat,
    embeddings,
    files,
    health,
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
    organization_guardrails,
    organization_keys,
    organization_pricing,
    organization_routing,
    organization_usage,
    organizations,
    otlp,
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


def register_routers(app: FastAPI, config: GatewayConfig) -> None:
    """Mount Otari's own routers, then whatever the bootstrap contributed.

    One aggregate router carries the API, so its mount prefix has one owner.
    Mount order on that router matters. The hybrid and hosted stubs are
    ``{path:path}`` catch-alls and the first matching route wins, so the
    stubs go last. A stub mounted earlier would answer a path that a
    contributed router serves.
    """
    api = APIRouter(prefix=API_ROOT)
    _register_core_routers(api, config)
    _register_contributed_routers(api, app.state.container)
    if config.is_hybrid_mode:
        api.include_router(hybrid_mode.router)
    elif config.is_hosted_mode:
        # A hosted control plane holds no data plane, so the inference prefixes
        # get catch-all stubs. A contributed route still wins: an overlay that
        # adds one has made a choice, and a fallback does not overrule it.
        api.include_router(hosted_mode.router)
    app.include_router(api)
    # OTLP is a sibling namespace, not a child of the API root: OTel owns the
    # /v1/{traces,logs,metrics} tail, so an exporter pointed at `{origin}/otlp`
    # appends it unaided. Standalone and hosted only, like the rest of the
    # management surface; a hybrid data plane stores no telemetry.
    if not config.is_hybrid_mode:
        app.include_router(otlp.router, prefix=OTLP_ROOT)


def _register_contributed_routers(api: APIRouter, container: Container) -> None:
    """Mount the routers this build's bootstrap contributed, each behind its gate.

    The additive half of the extension seam: an overlay records a router on the
    container and Otari mounts it, gated on the capability it names. Mounted in
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


def _register_core_routers(api: APIRouter, config: GatewayConfig) -> None:
    # Whether this deployment serves inference at all. False only for a hosted
    # control plane, which owns many tenants' wallets and credentials but runs
    # none of their traffic: that belongs on a hybrid data-plane gateway, whose
    # usage report is what debits the wallet. Serving a completion here would
    # skip that report and run unbilled (otari#822). Standalone stays true, and
    # legitimately serves both planes from the one process.
    serves_data_plane = not config.is_hosted_mode

    if serves_data_plane:
        api.include_router(chat.router)
    api.include_router(health.router)
    # Registered in every mode on purpose: the deployment bootstrap is how a
    # browser learns which mode it reached, so it is the one management-adjacent
    # route a hybrid gateway still answers.
    api.include_router(bootstrap.router)
    # The search backend a data-plane gateway calls, mounted only where it can
    # both authenticate one and answer it: this deployment holds a search
    # provider's credential and a token to recognize its own gateway by. Absent
    # otherwise rather than mounted and refusing, because a deployment that
    # configured neither is not offering this surface at all.
    if config.web_search_provider_configured() and config.web_search_backend_token:
        api.include_router(web_search_backend.router)
    # /api/v1/messages and /api/v1/responses now support hybrid mode (multi-attempt
    # fallback + usage reporting), so they're registered for hybrid too.
    if serves_data_plane:
        api.include_router(messages.router)
        api.include_router(responses.router)
        # Stateless MCP execution is a data-plane route. In hybrid mode it
        # authenticates through the platform's MCP resolver without opening a local
        # database; standalone uses the ordinary API/master-key path.
        api.include_router(mcp.router)

    if config.is_hybrid_mode:
        # The hybrid stub router is mounted by register_routers, after the
        # contributed routers; see the note there.
        return  # Remaining routers (including batches) are standalone-mode only

    api.include_router(admin.router)
    api.include_router(auth_session.router)
    api.include_router(auth_password.router)
    api.include_router(auth_signup.router)
    api.include_router(auth_password_reset.router)
    api.include_router(auth_webauthn.router)
    api.include_router(auth_oauth.router)
    if serves_data_plane:
        # The rest of the data plane. ``files`` sits here because an upload
        # exists to be referenced from a completion or a batch, so it follows
        # the traffic rather than the management API.
        api.include_router(embeddings.router)
        api.include_router(images.router)
        api.include_router(audio.router)
        api.include_router(files.router)
        api.include_router(rerank.router)
        api.include_router(search.router)
        api.include_router(batches.router)
        api.include_router(moderations.router)
    # The catalog reads are not operator-gated: /api/v1/models is discovery, not
    # dispatch. A control plane needs it to tell a tenant which models their
    # gateway could route to, and "models" is one of the surfaces bootstrap
    # publishes for a hosted deployment. The operator router goes first so
    # /api/v1/models/discoverable and /api/v1/models/metadata stay ahead of the
    # /api/v1/models/{model_id:path} catch-all the catalog router ends with.
    api.include_router(models.operator_router)
    api.include_router(models.catalog_router)
    api.include_router(providers.router)
    api.include_router(keys.router)
    api.include_router(users.router)
    api.include_router(organizations.router)
    api.include_router(organization_budgets.budgets_router)
    api.include_router(organization_budgets.ceilings_router)
    api.include_router(organization_pricing.router)
    api.include_router(organization_guardrails.router)
    # The tenant-scoped read over the same rows ``/api/v1/usage`` serves to an
    # operator. Mounted with the rest of the ``/api/v1/organizations/me`` surface
    # rather than beside the usage routers, because what it is scoped to is what
    # decides who may call it (otari#837).
    api.include_router(organization_usage.router)
    # The tenant-scoped reads and writes over the same tables ``/api/v1/routing/policies``
    # and ``/api/v1/aliases`` serve to an operator, mounted here for the same reason
    # (otari-ai#1942, otari-ai#1969).
    api.include_router(organization_routing.policies_router)
    api.include_router(organization_routing.aliases_router)
    # The member-scoped key surface: the caller's own keys, in workspaces they
    # may see. Mounted here for the reason the usage sibling above is; the
    # deployment-wide ``keys.router`` keeps its operator gate unchanged
    # (mozilla-ai/otari-ai#1941).
    api.include_router(organization_keys.router)
    api.include_router(workspaces.router)
    api.include_router(invitations.router)
    api.include_router(workspace_member_budget_policies.router)
    api.include_router(workspace_activation.router)
    api.include_router(workspace_mcp_servers.router)
    api.include_router(workspace_code_execution_policy.router)
    api.include_router(workspace_web_search.router)
    api.include_router(org_provider_keys.org_router)
    api.include_router(org_provider_keys.workspace_router)
    api.include_router(budgets.router)
    api.include_router(scoped_budgets.router)
    api.include_router(aliases.router)
    api.include_router(routing.router)
    api.include_router(routing_memory.router)
    # Both prefixed /pricing, split by who may call them; operator first, so
    # its DELETE /{model_key:path} does not sit behind the catalog catch-all.
    api.include_router(pricing.operator_router)
    api.include_router(pricing.catalog_router)
    # Both prefixed /usage. POST /external-events authenticates with an API
    # key rather than operator standing, so it is mounted on its own router.
    api.include_router(usage.operator_router)
    api.include_router(usage.ingest_router)
    api.include_router(agent_telemetry.router)
    api.include_router(settings.router)
    api.include_router(mail.router)
    api.include_router(maintenance_mode.router)
    # Both prefixed /tool-settings, split by who may call them: the reader is
    # the one route a tenant may reach, narrowed inside the handler
    # (otari-ai#1969). Operator first, matching the pair above, though neither
    # router here ends with a catch-all for the other to sit behind.
    api.include_router(tool_settings.operator_router)
    api.include_router(tool_settings.reader_router)
    api.include_router(search_tools.router)
    api.include_router(tools.router)
