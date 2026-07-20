import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing_extensions import override

from gateway.api.deps import set_config
from gateway.api.main import register_routers
from gateway.core.config import API_KEY_HEADER, LEGACY_API_KEY_HEADERS, X_API_KEY_HEADER, GatewayConfig
from gateway.core.database import create_session, init_db
from gateway.dashboard import get_dashboard_build_id, get_dashboard_dir
from gateway.rate_limit import RateLimiter
from gateway.root_page import FAVICON_SVG, ROOT_TUTORIAL_HTML
from gateway.services.alias_service import load_aliases_at_startup, reset_alias_cache, run_alias_refresher
from gateway.services.bootstrap_service import bootstrap_first_api_key
from gateway.services.file_store import build_file_store
from gateway.services.log_writer import LogWriter, NoopLogWriter, create_log_writer
from gateway.services.master_key_service import ensure_master_key
from gateway.services.pricing_init_service import (
    initialize_pricing_from_config,
    warn_if_require_pricing_without_pricing,
)
from gateway.services.pricing_service import configure_default_pricing
from gateway.services.provider_store_service import (
    load_providers_at_startup,
    reset_provider_cache,
    run_provider_refresher,
)
from gateway.services.runtime_settings_service import apply_overrides_from_db
from gateway.version import __version__

_PUBLIC_PREFIXES = ("/health",)
# Public, unauthenticated static assets that shared caches may keep. Paths here
# set their own Cache-Control at the route (favicon.svg), so the middleware only
# fills one in when it is missing.
_CACHEABLE_PATHS = ("/favicon.svg",)
# Vite stamps a content hash into every /assets filename, so a given URL is
# immutable; the middleware marks these public and cacheable for a year, since
# StaticFiles does not set Cache-Control on its own.
_CACHEABLE_PREFIXES = ("/assets/",)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses.

    Sets standard security headers on every response, plus cache-control
    headers on non-health endpoints to prevent CDN/proxy caches from
    storing authenticated responses.
    """

    @override
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        path = request.url.path
        if path.startswith(_PUBLIC_PREFIXES):
            return response
        if path.startswith(_CACHEABLE_PREFIXES):
            response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        elif path in _CACHEABLE_PATHS:
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        else:
            response.headers["Cache-Control"] = "private, no-store, no-cache"
            vary_values = {part.strip() for part in response.headers.get("Vary", "").split(",") if part.strip()}
            vary_values.add("Authorization")
            response.headers["Vary"] = ", ".join(sorted(vary_values))
        return response


def _validate_platform_config(config: GatewayConfig) -> None:
    config.validate_mode_selection()
    if not config.is_hybrid_mode:
        return
    if not config.platform.get("base_url"):
        msg = "platform.base_url is required when hybrid mode is active"
        raise ValueError(msg)
    if config.providers:
        msg = "Local provider credentials are not supported in hybrid mode"
        raise ValueError(msg)


def _create_lifespan(config: GatewayConfig) -> Callable[[FastAPI], Any]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        configure_default_pricing(config.default_pricing)
        log_writer: LogWriter
        alias_refresher: asyncio.Task[None] | None = None
        provider_refresher: asyncio.Task[None] | None = None
        if config.is_hybrid_mode:
            log_writer = NoopLogWriter()
        else:
            init_db(config)
            async with create_session() as session:
                # Persisted dashboard overrides win over config/env; apply them
                # before pricing init so default-pricing behavior is consistent.
                await apply_overrides_from_db(config, session)
                # Generate + persist a master key on first run when none is set,
                # so the dashboard is reachable without hand-editing config, and
                # the management API is never left unauthenticated.
                await ensure_master_key(config, session)
                # Overlay dashboard-stored providers before pricing init, so a
                # provider added at runtime is visible to everything that reads
                # config.providers (pricing seeding, discovery, dispatch).
                await load_providers_at_startup(session, config)
                await bootstrap_first_api_key(config, session)
                await initialize_pricing_from_config(config, session)
                await warn_if_require_pricing_without_pricing(config, session)
                await load_aliases_at_startup(session)
            log_writer = create_log_writer(config.log_writer_strategy)
            app.state.file_store = build_file_store(config)
            # Alias resolution is synchronous and reads a cache, so something has
            # to reload it. A write refreshes its own worker; this is what makes
            # every other worker and replica catch up.
            alias_refresher = asyncio.create_task(run_alias_refresher())
            # Provider credentials are the same shape: resolution reads
            # config.providers synchronously, so the overlay is reloaded on a TTL
            # to converge sibling workers and replicas after a dashboard write.
            provider_refresher = asyncio.create_task(run_provider_refresher(config))

        await log_writer.start()
        app.state.log_writer = log_writer

        try:
            yield
        finally:
            if alias_refresher is not None:
                alias_refresher.cancel()
                with suppress(asyncio.CancelledError):
                    await alias_refresher
                reset_alias_cache()
            if provider_refresher is not None:
                provider_refresher.cancel()
                with suppress(asyncio.CancelledError):
                    await provider_refresher
                reset_provider_cache()
            await log_writer.stop()

    return lifespan


def create_app(config: GatewayConfig) -> FastAPI:
    """Create and configure FastAPI application."""

    _validate_platform_config(config)
    set_config(config)

    app = FastAPI(
        title="otari",
        description="Otari, an OpenAI-compatible LLM gateway with API key management",
        version=__version__,
        docs_url="/docs" if config.enable_docs else None,
        redoc_url="/redoc" if config.enable_docs else None,
        openapi_url="/openapi.json" if config.enable_docs else None,
        lifespan=_create_lifespan(config),
    )

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        from fastapi.openapi.utils import get_openapi

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        if "components" not in openapi_schema:
            openapi_schema["components"] = {}
        if "securitySchemes" not in openapi_schema["components"]:
            openapi_schema["components"]["securitySchemes"] = {}

        openapi_schema["components"]["securitySchemes"]["ApiKeyAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_HEADER,
            "description": f"Enter your API key here (sent as {API_KEY_HEADER} header).",
        }
        openapi_schema["components"]["securitySchemes"]["XApiKeyAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": X_API_KEY_HEADER,
            "description": "Anthropic-native clients send credentials here (no Bearer prefix).",
        }

        for path, path_item in openapi_schema.get("paths", {}).items():
            if path.startswith(_PUBLIC_PREFIXES):
                continue
            for operation in path_item.values():
                if isinstance(operation, dict):
                    operation["security"] = [{"ApiKeyAuth": []}, {"XApiKeyAuth": []}]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    @app.get("/welcome", response_class=HTMLResponse, include_in_schema=False)
    async def root_tutorial() -> str:
        return ROOT_TUTORIAL_HTML

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon() -> Response:
        return Response(
            content=FAVICON_SVG,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # The admin dashboard drives the standalone-only management API (models,
    # pricing, aliases, settings); in hybrid mode there is none, so the root keeps
    # serving the tutorial. The dashboard is a single-page app, so a static mount
    # for /assets plus an index.html at / is all it needs (navigation is client-side).
    dashboard_dir = get_dashboard_dir() if not config.is_hybrid_mode else None
    if dashboard_dir is not None:
        index_file = dashboard_dir / "index.html"
        app.mount(
            "/assets",
            StaticFiles(directory=dashboard_dir / "assets"),
            name="dashboard-assets",
        )

        @app.get("/", include_in_schema=False)
        async def dashboard_index() -> FileResponse:
            return FileResponse(index_file, media_type="text/html")

        # Lets an open tab notice it is running code this server no longer
        # serves, and offer a reload, instead of sitting on a stale bundle until
        # someone thinks to clear their storage. Public like the page it
        # describes, and read per request so an in-place rebuild is picked up.
        # The security middleware marks it no-store, which the poll depends on.
        @app.get("/dashboard-build.json", include_in_schema=False)
        async def dashboard_build() -> dict[str, str]:
            return {"build": get_dashboard_build_id(dashboard_dir), "version": __version__}
    else:

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def root_index() -> str:
            return ROOT_TUTORIAL_HTML

    app.add_middleware(SecurityHeadersMiddleware)

    if config.cors_allow_origins:
        allow_credentials = "*" not in config.cors_allow_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=[
                "Content-Type",
                "Authorization",
                API_KEY_HEADER,
                X_API_KEY_HEADER,
                *LEGACY_API_KEY_HEADERS,
            ],
        )

    if config.enable_metrics:
        from gateway.metrics import MetricsMiddleware

        app.add_middleware(MetricsMiddleware)

    if config.rate_limit_rpm is not None:
        app.state.rate_limiter = RateLimiter(config.rate_limit_rpm)
    else:
        app.state.rate_limiter = None

    app.state.config = config
    app.state.gateway_mode = config.effective_mode

    register_routers(app, config)

    if config.enable_metrics:
        from gateway.metrics import metrics_endpoint

        app.add_route("/metrics", metrics_endpoint, methods=["GET"])

    return app
