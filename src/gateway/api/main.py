from fastapi import Depends, FastAPI

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
    messages,
    models,
    moderations,
    org_provider_keys,
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
from gateway.core.config import GatewayConfig


def register_routers(app: FastAPI, config: GatewayConfig) -> None:
    """Mount Otari's own routers, then whatever the bootstrap contributed."""
    _register_core_routers(app, config)
    _register_contributed_routers(app)
    if config.is_hybrid_mode:
        # Last, and after the contributed routers on purpose. These are
        # ``{path:path}`` catch-alls over whole management prefixes
        # (/v1/organizations, /v1/usage, ...), and FastAPI serves the first
        # route that matches, so registering them earlier would swallow an
        # overlay route under any of those prefixes and answer "manage this via
        # the platform UI" instead. They are a fallback for a path nothing else
        # serves, so they are mounted like one.
        app.include_router(hybrid_mode.router)
    elif config.is_hosted_mode:
        # The same treatment for the opposite plane, and last for the same
        # reason: a hosted control plane holds no data plane, so the inference
        # prefixes get catch-all stubs that a contributed router still wins
        # against. An overlay that deliberately contributes a data-plane route
        # to a control plane has made a choice, and a fallback does not overrule
        # one. See gateway.api.routes.hosted_mode.
        app.include_router(hosted_mode.router)


def _register_contributed_routers(app: FastAPI) -> None:
    """Mount the routers this build's bootstrap contributed, each behind its gate.

    The additive half of the extension seam: an overlay records a router on the
    container and Otari mounts it, gated on the capability it names. Mounted in
    both modes, because an overlay may extend the data plane as readily as the
    management plane. With no bootstrap configured there are none, so this is a
    no-op for the plain build.

    Mounted after Otari's own routers and before the hybrid stubs, so a
    contribution cannot take a path the core already serves and the hybrid
    stubs' catch-alls cannot take one the contribution serves.
    """
    container: Container = app.state.container
    for contribution in container.router_contributions():
        app.include_router(
            contribution.router,
            dependencies=[Depends(require_capability(contribution.capability))],
        )


def _register_core_routers(app: FastAPI, config: GatewayConfig) -> None:
    # Whether this deployment serves inference at all. False only for a hosted
    # control plane, which owns many tenants' wallets and credentials but runs
    # none of their traffic: that belongs on a hybrid data-plane gateway, whose
    # usage report is what debits the wallet. Serving a completion here would
    # skip that report and run unbilled (otari#822). Standalone stays true, and
    # legitimately serves both planes from the one process.
    serves_data_plane = not config.is_hosted_mode

    if serves_data_plane:
        app.include_router(chat.router)
    app.include_router(health.router)
    # Registered in every mode on purpose: the deployment bootstrap is how a
    # browser learns which mode it reached, so it is the one management-adjacent
    # route a hybrid gateway still answers.
    app.include_router(bootstrap.router)
    # The search backend a data-plane gateway calls, mounted only where it can
    # both authenticate one and answer it: this deployment holds a search
    # provider's credential and a token to recognize its own gateway by. Absent
    # otherwise rather than mounted and refusing, because a deployment that
    # configured neither is not offering this surface at all.
    if config.web_search_provider_configured() and config.web_search_backend_token:
        app.include_router(web_search_backend.router)
    # /v1/messages and /v1/responses now support hybrid mode (multi-attempt
    # fallback + usage reporting), so they're registered for hybrid too.
    if serves_data_plane:
        app.include_router(messages.router)
        app.include_router(responses.router)

    if config.is_hybrid_mode:
        # The hybrid stub router is mounted by register_routers, after the
        # contributed routers; see the note there.
        return  # Remaining routers (including batches) are standalone-mode only

    app.include_router(admin.router)
    app.include_router(auth_session.router)
    app.include_router(auth_password.router)
    app.include_router(auth_signup.router)
    app.include_router(auth_password_reset.router)
    app.include_router(auth_webauthn.router)
    app.include_router(auth_oauth.router)
    if serves_data_plane:
        # The rest of the data plane. ``files`` sits here because an upload
        # exists to be referenced from a completion or a batch, so it follows
        # the traffic rather than the management API.
        app.include_router(embeddings.router)
        app.include_router(images.router)
        app.include_router(audio.router)
        app.include_router(files.router)
        app.include_router(rerank.router)
        app.include_router(search.router)
        app.include_router(batches.router)
        app.include_router(moderations.router)
    # Not gated: /v1/models is discovery, not dispatch. A control plane needs it
    # to tell a tenant which models their gateway could route to, and "models"
    # is one of the surfaces bootstrap publishes for a hosted deployment.
    app.include_router(models.router)
    app.include_router(providers.router)
    app.include_router(keys.router)
    app.include_router(users.router)
    app.include_router(organizations.router)
    app.include_router(organization_pricing.router)
    app.include_router(organization_guardrails.router)
    # The tenant-scoped read over the same rows ``/v1/usage`` serves to an
    # operator. Mounted with the rest of the ``/v1/organizations/me`` surface
    # rather than beside ``usage.router``, because what it is scoped to is what
    # decides who may call it (otari#837).
    app.include_router(organization_usage.router)
    # The tenant-scoped read over the same table ``/v1/routing/policies`` serves
    # to an operator, mounted here for the same reason (otari-ai#1942).
    app.include_router(organization_routing.router)
    # The member-scoped key surface: the caller's own keys, in workspaces they
    # may see. Mounted here for the reason the usage sibling above is; the
    # deployment-wide ``keys.router`` keeps its operator gate unchanged
    # (mozilla-ai/otari-ai#1941).
    app.include_router(organization_keys.router)
    app.include_router(workspaces.router)
    app.include_router(invitations.router)
    app.include_router(workspace_member_budget_policies.router)
    app.include_router(workspace_activation.router)
    app.include_router(workspace_mcp_servers.router)
    app.include_router(workspace_code_execution_policy.router)
    app.include_router(workspace_web_search.router)
    app.include_router(org_provider_keys.org_router)
    app.include_router(org_provider_keys.workspace_router)
    app.include_router(budgets.router)
    app.include_router(scoped_budgets.router)
    app.include_router(aliases.router)
    app.include_router(routing.router)
    app.include_router(routing_memory.router)
    app.include_router(pricing.router)
    app.include_router(usage.router)
    app.include_router(agent_telemetry.router)
    app.include_router(otlp.router)
    app.include_router(settings.router)
    app.include_router(mail.router)
    app.include_router(maintenance_mode.router)
    app.include_router(tool_settings.router)
    app.include_router(search_tools.router)
    app.include_router(tools.router)
