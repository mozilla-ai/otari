from fastapi import FastAPI

from gateway.api.routes import (
    agent_telemetry,
    aliases,
    audio,
    auth_session,
    batches,
    bootstrap,
    budgets,
    chat,
    embeddings,
    files,
    health,
    hybrid_mode,
    images,
    keys,
    messages,
    models,
    moderations,
    organizations,
    otlp,
    pricing,
    providers,
    rerank,
    responses,
    routing,
    routing_memory,
    search,
    search_tools,
    settings,
    tool_settings,
    tools,
    usage,
    users,
    workspaces,
)
from gateway.core.config import GatewayConfig


def register_routers(app: FastAPI, config: GatewayConfig) -> None:
    app.include_router(chat.router)
    app.include_router(health.router)
    # Registered in both modes on purpose: the deployment bootstrap is how a
    # browser learns which mode it reached, so it is the one management-adjacent
    # route a hybrid gateway still answers.
    app.include_router(bootstrap.router)
    # /v1/messages and /v1/responses now support hybrid mode (multi-attempt
    # fallback + usage reporting), so they're registered in both modes.
    app.include_router(messages.router)
    app.include_router(responses.router)

    if config.is_hybrid_mode:
        app.include_router(hybrid_mode.router)
        return  # Remaining routers (including batches) are standalone-mode only

    app.include_router(auth_session.router)
    app.include_router(embeddings.router)
    app.include_router(images.router)
    app.include_router(audio.router)
    app.include_router(files.router)
    app.include_router(rerank.router)
    app.include_router(search.router)
    app.include_router(batches.router)
    app.include_router(moderations.router)
    app.include_router(models.router)
    app.include_router(providers.router)
    app.include_router(keys.router)
    app.include_router(users.router)
    app.include_router(organizations.router)
    app.include_router(workspaces.router)
    app.include_router(budgets.router)
    app.include_router(aliases.router)
    app.include_router(routing.router)
    app.include_router(routing_memory.router)
    app.include_router(pricing.router)
    app.include_router(usage.router)
    app.include_router(agent_telemetry.router)
    app.include_router(otlp.router)
    app.include_router(settings.router)
    app.include_router(tool_settings.router)
    app.include_router(search_tools.router)
    app.include_router(tools.router)
