import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing_extensions import override

from gateway.api.deps import set_config
from gateway.api.main import register_routers
from gateway.core.config import API_KEY_HEADER, X_API_KEY_HEADER, GatewayConfig
from gateway.core.database import create_session, init_db
from gateway.dashboard import DASHBOARD_PACKAGE_PATH, get_dashboard_build_id, get_dashboard_dir
from gateway.inflight import InFlightMiddleware, InFlightRegistry
from gateway.log_config import logger
from gateway.rate_limit import RateLimiter
from gateway.root_page import FAVICON_SVG, ROOT_TUTORIAL_HTML
from gateway.services.alias_service import load_aliases_at_startup, reset_alias_cache, run_alias_refresher
from gateway.services.bootstrap_service import bootstrap_first_api_key
from gateway.services.dashboard_session_service import revoke_sessions_on_master_key_change
from gateway.services.file_store import build_file_store
from gateway.services.log_writer import LogWriter, NoopLogWriter, create_log_writer
from gateway.services.master_key_service import ensure_master_key
from gateway.services.model_catalog_service import (
    clear_catalog_cache,
    run_catalog_refresher,
)
from gateway.services.model_discovery_service import (
    reset_discovery_cache,
    run_discovery_refresher,
)
from gateway.services.policy_store import (
    load_policies_at_startup,
    reset_policy_cache,
    run_policy_refresher,
)
from gateway.services.pricing_init_service import (
    initialize_pricing_from_config,
    warn_if_gateway_tools_lack_pricing,
    warn_if_require_pricing_without_pricing,
    warn_if_router_candidates_lack_pricing,
    warn_if_search_tools_lack_flat_pricing,
)
from gateway.services.pricing_refresh_service import (
    load_persisted_price_snapshot,
    run_price_snapshot_refresher,
)
from gateway.services.pricing_service import configure_default_pricing, configure_provider_types
from gateway.services.provider_store_service import (
    load_providers_at_startup,
    reset_provider_cache,
    run_provider_refresher,
)
from gateway.services.runtime_settings_service import apply_overrides_from_db
from gateway.services.search_backend import close_search_client
from gateway.services.search_tool_store_service import (
    load_search_tools_at_startup,
    reset_search_tool_cache,
    run_search_tool_refresher,
)
from gateway.services.secret_box import validate_secret_key
from gateway.services.tool_settings_service import apply_overrides_from_db as apply_tool_overrides_from_db
from gateway.version import __version__

_PUBLIC_PREFIXES = ("/health",)
# Paths authenticated by the master key in the request body (sign-in) or the
# session cookie (sign-out) rather than the header schemes; the OpenAPI
# security stamp below skips them. They still get the no-store cache headers.
_COOKIE_AUTH_PREFIXES = ("/v1/auth/session",)
# Public, unauthenticated static assets that shared caches may keep. Paths here
# set their own Cache-Control at the route (favicon.svg), so the middleware only
# fills one in when it is missing.
_CACHEABLE_PATHS = ("/favicon.svg",)
# Vite stamps a content hash into every /assets filename, so a given URL is
# immutable; the middleware marks these public and cacheable for a year, since
# StaticFiles does not set Cache-Control on its own.
_CACHEABLE_PREFIXES = ("/assets/",)
# The install manifest and home-screen icons, public like the page that links
# them. Their names are stable rather than content-hashed, so they get a day of
# caching like the favicon, not the immutable year /assets gets.
_SHORT_CACHE_PREFIXES = ("/pwa/",)


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
        elif path in _CACHEABLE_PATHS or path.startswith(_SHORT_CACHE_PREFIXES):
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


# How long shutdown waits for refreshers to acknowledge cancellation.
#
# Cancelling a task is a request, not a guarantee. The CancelledError is
# delivered at whatever the task is awaiting, and a nested cancel scope there can
# absorb it: httpx and the provider SDKs implement their own timeouts as anyio
# cancel scopes, which call ``Task.uncancel`` when they decide the cancellation
# was theirs. The refresher loop then resumes, falls through to its ``sleep``,
# and naps for a whole interval (a day, for the models.dev catalog). An
# unbounded ``await task`` never returns, so the lifespan never finishes and
# uvicorn's shutdown hangs behind a background refresh. Bounding the wait and
# moving on is the right trade: the event loop is torn down immediately after,
# and no refresher owns state that a late tick could corrupt.
_REFRESHER_STOP_TIMEOUT_SECONDS = 5.0


def _log_abandoned_refresher(name: str) -> None:
    logger.warning(
        "%s refresher did not stop within %.0fs; abandoning it so shutdown can finish",
        name,
        _REFRESHER_STOP_TIMEOUT_SECONDS,
    )


def _log_refresher_stop(task: asyncio.Task[None], name: str) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.warning("%s refresher stopped with an unexpected error", name, exc_info=error)


async def _wait_for_refresher_stop(task: asyncio.Task[None], name: str) -> None:
    """Wait for a cancelled lifespan refresher, but never indefinitely.

    ``asyncio.wait`` rather than ``await task``: it takes a timeout, and it
    reports the outcome instead of re-raising it, so a refresher that died on an
    unexpected error is logged here rather than aborting the rest of shutdown
    (the log writer and the pooled search client still need closing).
    """
    done, _pending = await asyncio.wait({task}, timeout=_REFRESHER_STOP_TIMEOUT_SECONDS)
    if not done:
        _log_abandoned_refresher(name)
        return
    _log_refresher_stop(task, name)


async def _stop_refresher(task: asyncio.Task[None], name: str) -> None:
    """Cancel one lifespan refresher and wait for it, but never indefinitely."""
    task.cancel()
    await _wait_for_refresher_stop(task, name)


async def _stop_refreshers(refreshers: list[tuple[asyncio.Task[None], str]]) -> None:
    """Cancel all refreshers, then give the group one shared shutdown bound."""
    if not refreshers:
        return
    for task, _name in refreshers:
        task.cancel()
    done, pending = await asyncio.wait(
        {task for task, _name in refreshers},
        timeout=_REFRESHER_STOP_TIMEOUT_SECONDS,
    )
    for task, name in refreshers:
        if task in pending:
            _log_abandoned_refresher(name)
        elif task in done:
            _log_refresher_stop(task, name)


def _create_lifespan(config: GatewayConfig) -> Callable[[FastAPI], Any]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        configure_default_pricing(config.default_pricing)
        # Bound method, not a snapshot: it reads config.providers on every call, so
        # a provider added or re-typed in the dashboard is priced under the
        # implementation it actually dispatches to. Deliberately not
        # ``provider_instance_type``: that normalizes ``openai-compatible`` to
        # ``openai``, which names a wire protocol rather than a vendor.
        configure_provider_types(config.provider_pricing_implementation)
        log_writer: LogWriter
        alias_refresher: asyncio.Task[None] | None = None
        policy_refresher: asyncio.Task[None] | None = None
        provider_refresher: asyncio.Task[None] | None = None
        search_tool_refresher: asyncio.Task[None] | None = None
        price_refresher: asyncio.Task[None] | None = None
        discovery_refresher: asyncio.Task[None] | None = None
        catalog_refresher: asyncio.Task[None] | None = None
        if config.is_hybrid_mode:
            log_writer = NoopLogWriter()
        else:
            init_db(config)
            async with create_session() as session:
                # Persisted dashboard overrides win over config/env; apply them
                # before pricing init so default-pricing behavior is consistent.
                await apply_overrides_from_db(config, session)
                await load_persisted_price_snapshot(session)
                # Persisted tool/guardrail overrides (service URLs + web-search
                # knobs) win over config/env too; apply them so the running worker
                # reflects a dashboard change made in a prior run.
                await apply_tool_overrides_from_db(config, session)
                # Overlay dashboard-stored search tools before the checks below,
                # so a tool added through the dashboard is one the backend-URL
                # warning and the flat-pricing warning can see.
                await load_search_tools_at_startup(session, config)
                # After the overrides, not at config load: the web-search URL a
                # searxng search tool inherits can be the dashboard-stored one
                # applied just above, and that tool is only broken if nothing
                # supplied it at all.
                if missing_backend_url := config.search_tools_without_backend_url():
                    logger.warning(
                        "No backend URL for search tool(s): %s. POST /v1/search refuses them with a 400 until "
                        "each declares an 'api_base' or a web-search URL is set (web_search_url, "
                        "OTARI_WEB_SEARCH_URL, or the dashboard's Tools page).",
                        ", ".join(sorted(missing_backend_url)),
                    )
                # Generate + persist a master key on first run when none is set,
                # so the dashboard is reachable without hand-editing config, and
                # the management API is never left unauthenticated.
                await ensure_master_key(config, session)
                # Dashboard sessions must not outlive the key they were minted
                # under: revoke them all when the master key changed across a
                # restart (e.g. OTARI_MASTER_KEY was rotated).
                await revoke_sessions_on_master_key_change(config, session)
                # Overlay dashboard-stored providers before pricing init, so a
                # provider added at runtime is visible to everything that reads
                # config.providers (pricing seeding, discovery, dispatch).
                await load_providers_at_startup(session, config)
                await bootstrap_first_api_key(config, session)
                await initialize_pricing_from_config(config, session)
                await warn_if_require_pricing_without_pricing(config, session)
                await warn_if_search_tools_lack_flat_pricing(config, session)
                await warn_if_gateway_tools_lack_pricing(config, session)
                await warn_if_router_candidates_lack_pricing(config, session)
                await load_aliases_at_startup(session)
                await load_policies_at_startup(session)
            log_writer = create_log_writer(config.log_writer_strategy)
            app.state.file_store = build_file_store(config)
            # Alias resolution is synchronous and reads a cache, so something has
            # to reload it. A write refreshes its own worker; this is what makes
            # every other worker and replica catch up.
            alias_refresher = asyncio.create_task(run_alias_refresher())
            # Routing policies are the same shape as aliases: resolution reads a
            # process cache synchronously, so it is reloaded on a TTL to converge
            # sibling workers and replicas after a write.
            policy_refresher = asyncio.create_task(run_policy_refresher())
            # Provider credentials are the same shape: resolution reads
            # config.providers synchronously, so the overlay is reloaded on a TTL
            # to converge sibling workers and replicas after a dashboard write.
            provider_refresher = asyncio.create_task(run_provider_refresher(config))
            # Search tools are the provider overlay's twin: resolve_search_tool
            # reads config.search_tools synchronously, so the overlay is reloaded
            # on a TTL to converge sibling workers and replicas after a write.
            search_tool_refresher = asyncio.create_task(run_search_tool_refresher(config))
            # An accepted pricing snapshot is applied in-memory by the worker that
            # served the confirm; reload it on a TTL so sibling workers and replicas
            # converge, the same way aliases and provider credentials do.
            price_refresher = asyncio.create_task(run_price_snapshot_refresher())
            # Discovery is the one cache that used to be filled on the request
            # path, which put model_discovery_timeout_seconds (10s per
            # unreachable provider) on a dashboard page load and held that
            # request's database session open for the whole dial. The refresher
            # owns the dialing now and reads answer from the cache. Not awaited:
            # priming runs on its first tick so a slow provider cannot delay boot.
            #
            # Started unconditionally, and each loop re-checks whether caching is
            # on. Gating task creation on the setting here would strand the
            # gateway in the one state that combination must never reach:
            # model_cache_ttl_seconds and models_dev_cache_ttl_seconds are both
            # runtime-settable from the dashboard's Settings page, and raising
            # either from 0 flips every read onto the serve-from-cache path
            # immediately. With no refresher running, that cache is then filled
            # once per provider and never refreshed again for the life of the
            # worker, which is the "cache nothing refreshes" mode these knobs
            # deliberately do not offer.
            discovery_refresher = asyncio.create_task(run_discovery_refresher(config))
            # Same shape for the models.dev catalog, whose fetch is bounded at 15s.
            catalog_refresher = asyncio.create_task(run_catalog_refresher(config))

        # Start the writer inside the try so a failure here still runs the cleanup
        # below; the refresher tasks are already created and would otherwise leak.
        log_writer_started = False
        try:
            await log_writer.start()
            log_writer_started = True
            app.state.log_writer = log_writer
            yield
        finally:
            refreshers = [
                (alias_refresher, "alias"),
                (policy_refresher, "policy"),
                (provider_refresher, "provider"),
                (search_tool_refresher, "search tool"),
                (price_refresher, "price snapshot"),
                (discovery_refresher, "model discovery"),
                (catalog_refresher, "models.dev catalog"),
            ]
            await _stop_refreshers([(task, name) for task, name in refreshers if task is not None])
            if alias_refresher is not None:
                reset_alias_cache()
            if policy_refresher is not None:
                reset_policy_cache()
            if provider_refresher is not None:
                reset_provider_cache()
            if search_tool_refresher is not None:
                reset_search_tool_cache()
            if discovery_refresher is not None:
                reset_discovery_cache()
            if catalog_refresher is not None:
                clear_catalog_cache()
            # Only stop a writer that actually started; if start() raised there is
            # nothing to stop, but the refreshers above still needed cancelling.
            if log_writer_started:
                await log_writer.stop()
            # POST /v1/search dispatches on one pooled client for the process, so
            # shutdown owns closing it. A no-op when no search was ever served.
            await close_search_client()

    return lifespan


def create_app(config: GatewayConfig) -> FastAPI:
    """Create and configure FastAPI application."""

    _validate_platform_config(config)
    # A set-but-invalid OTARI_SECRET_KEY must not silently pass startup and then
    # break provider-credential storage at request time. Fail fast here instead.
    validate_secret_key()
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
            if path.startswith(_PUBLIC_PREFIXES) or path.startswith(_COOKIE_AUTH_PREFIXES):
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
        # Installing the dashboard to a phone home screen needs the manifest and
        # the PNG icons it points at (index.html links them from /pwa/). Only the
        # standalone dashboard is an app worth installing, so this rides along
        # with it rather than being served in hybrid mode next to the tutorial.
        pwa_dir = dashboard_dir / "pwa"
        if pwa_dir.is_dir():
            app.mount("/pwa", StaticFiles(directory=pwa_dir), name="dashboard-pwa")

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
        # Hybrid mode has no local management API, so the tutorial is the intended
        # root there and there is nothing to report. In standalone mode a missing
        # bundle means nobody built it, which is the ordinary state of a source
        # checkout now that the bundle is gitignored rather than committed. Say so
        # once at startup, so "/" serving the tutorial reads as a build step not
        # taken rather than as a broken dashboard.
        if not config.is_hybrid_mode:
            logger.info(
                'No dashboard bundle found at gateway/%s, so "/" serves the get-started tutorial '
                "(the API is unaffected). Run `make dashboard` to build it; the Docker image builds it for you.",
                DASHBOARD_PACKAGE_PATH,
            )

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
            ],
        )

    # Tracks what the gateway is serving right now, so the activity log can show
    # requests in progress and not only settled ones. Unconditional: the entry is
    # a dict insert and delete per request, and the middleware is what guarantees
    # an entry never outlives its response (see gateway.inflight).
    app.state.inflight = InFlightRegistry()
    app.add_middleware(InFlightMiddleware, registry=app.state.inflight)

    if config.enable_metrics:
        from gateway.metrics import MetricsMiddleware

        app.add_middleware(MetricsMiddleware)

    if config.rate_limit_rpm is not None:
        app.state.rate_limiter = RateLimiter(config.rate_limit_rpm)
    else:
        app.state.rate_limiter = None

    if config.dashboard_login_rate_limit_per_minute is not None:
        app.state.login_rate_limiter = RateLimiter(config.dashboard_login_rate_limit_per_minute)
    else:
        app.state.login_rate_limiter = None

    app.state.config = config
    app.state.gateway_mode = config.effective_mode

    register_routers(app, config)

    if config.enable_metrics:
        from gateway.metrics import metrics_endpoint

        app.add_route("/metrics", metrics_endpoint, methods=["GET"])

    return app
