from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing_extensions import override

from gateway.api.deps import set_config
from gateway.api.main import register_routers
from gateway.core.config import API_KEY_HEADER, LEGACY_API_KEY_HEADERS, GatewayConfig
from gateway.core.database import create_session, init_db
from gateway.rate_limit import RateLimiter
from gateway.root_page import FAVICON_SVG, ROOT_TUTORIAL_HTML
from gateway.services.bootstrap_service import bootstrap_first_api_key
from gateway.services.file_store import build_file_store
from gateway.services.log_writer import LogWriter, NoopLogWriter, create_log_writer
from gateway.services.pricing_init_service import (
    initialize_pricing_from_config,
    warn_if_require_pricing_without_pricing,
)
from gateway.services.pricing_service import configure_default_pricing
from gateway.version import __version__

_PUBLIC_PREFIXES = ("/health",)
# Public, unauthenticated static assets that are safe for shared caches and set
# their own Cache-Control; the middleware leaves their caching headers alone.
_CACHEABLE_PATHS = ("/favicon.svg",)


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
        if not request.url.path.startswith(_PUBLIC_PREFIXES) and request.url.path not in _CACHEABLE_PATHS:
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
        if config.is_hybrid_mode:
            log_writer = NoopLogWriter()
        else:
            init_db(config)
            async with create_session() as session:
                await bootstrap_first_api_key(config, session)
                await initialize_pricing_from_config(config, session)
                await warn_if_require_pricing_without_pricing(config, session)
            log_writer = create_log_writer(config.log_writer_strategy)
            app.state.file_store = build_file_store(config)

        await log_writer.start()
        app.state.log_writer = log_writer

        try:
            yield
        finally:
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

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root_tutorial() -> str:
        return ROOT_TUTORIAL_HTML

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon() -> Response:
        return Response(
            content=FAVICON_SVG,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )

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

    app.state.gateway_mode = config.effective_mode

    register_routers(app, config)

    if config.enable_metrics:
        from gateway.metrics import metrics_endpoint

        app.add_route("/metrics", metrics_endpoint, methods=["GET"])

    return app
