from fastapi import FastAPI

from gateway.api.routes import (
    audio,
    batches,
    budgets,
    chat,
    embeddings,
    files,
    health,
    images,
    keys,
    messages,
    models,
    moderations,
    platform_mode,
    pricing,
    rerank,
    responses,
    usage,
    users,
)
from gateway.api.routes import (
    router as router_routes,
)
from gateway.core.config import GatewayConfig


def register_routers(app: FastAPI, config: GatewayConfig) -> None:
    app.include_router(chat.router)
    app.include_router(health.router)
    # /v1/messages and /v1/responses now support platform mode (multi-attempt
    # fallback + usage reporting), so they're registered in both modes.
    app.include_router(messages.router)
    app.include_router(responses.router)

    if config.is_platform_mode:
        app.include_router(platform_mode.router)
        return  # Remaining routers (including batches) are standalone-mode only

    app.include_router(embeddings.router)
    app.include_router(images.router)
    app.include_router(audio.router)
    app.include_router(files.router)
    app.include_router(rerank.router)
    app.include_router(batches.router)
    app.include_router(moderations.router)
    app.include_router(models.router)
    app.include_router(keys.router)
    app.include_router(users.router)
    app.include_router(budgets.router)
    app.include_router(pricing.router)
    app.include_router(usage.router)
    app.include_router(router_routes.router)
