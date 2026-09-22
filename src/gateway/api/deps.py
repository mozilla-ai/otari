import secrets
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.auth.models import hash_key
from gateway.container import Container
from gateway.core.config import API_KEY_HEADER, API_ROOT, X_API_KEY_HEADER, GatewayConfig
from gateway.core.database import DATABASE_ERRORS, create_session, get_db
from gateway.core.feature import CoreFeature
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.metrics import REGISTRY, Counter
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import User as TenancyUser
from gateway.ports.api_key_format_port import ApiKeyFormatPort, Malformed, Misdirected
from gateway.ports.billing_port import BillingPort
from gateway.ports.entitlement_port import EntitlementPort
from gateway.ports.growth_signal_port import GrowthSignalPort
from gateway.ports.identity_provider_port import IdentityProviderPort
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.ports.telemetry_storage_port import TelemetryStoragePort
from gateway.repositories.api_keys import ApiKeyRepository
from gateway.repositories.budgets import BudgetRepositories
from gateway.repositories.overview.overview_repository import OverviewRepository
from gateway.repositories.providers import OrgProviderKeyModelRepository
from gateway.repositories.tenancy import OrgProviderKeyRepository
from gateway.services.api_keys import ApiKeyService
from gateway.services.budgets import BudgetService, WorkspaceBudgetDefaultService
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, resolve_dashboard_session
from gateway.services.file_service import StagedFile
from gateway.services.file_store import FileStore
from gateway.services.files import SandboxFileBridge
from gateway.services.log_writer import LogWriter
from gateway.services.master_key_service import hash_master_key, is_generated_master_key, load_master_key_hash
from gateway.services.organization_pricing_service import OrganizationPricingService
from gateway.services.overview.overview_service import OverviewService
from gateway.services.providers import OrgProviderModelService
from gateway.services.routing import clear_router_backend_cache
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.deployment_user_service import DeploymentUserService
from gateway.services.tenancy.org_provider_key_service import OrgProviderKeyService, refresh_org_provider_cache
from gateway.services.tenancy.provisioning_service import ensure_bootstrap_identity
from gateway.services.tenancy.workspace_service import WorkspaceService

# Legacy module-level fallback. Config now lives on ``app.state.config`` (set in
# ``create_app``); ``get_config`` reads from the request's app state and only
# falls back to this shim for callers that set it directly (see ``set_config``).
_config: GatewayConfig | None = None
_LAST_USED_UPDATE_INTERVAL_SECONDS = 300

AUTH_FAILURES = Counter(
    "gateway_auth_failures",
    "Total number of authentication failures",
    ["reason"],
    registry=REGISTRY,
)


def record_auth_failure(reason: str) -> None:
    """Record an authentication failure."""
    AUTH_FAILURES.labels(reason=reason).inc()


def _as_utc(value: datetime | None) -> datetime | None:
    """Return ``value`` as a timezone-aware datetime in UTC.

    SQLite stores ``DateTime(timezone=True)`` columns as naive strings and
    returns them naive on read. PostgreSQL returns them as aware. Normalising
    here keeps the subtraction/comparison call sites identical across both
    backends — a naive value is *assumed* to be UTC, which matches how the
    gateway writes them (always ``datetime.now(UTC)``).
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def set_config(config: GatewayConfig) -> None:
    """Set the legacy module-level config fallback.

    Compatibility shim. Config is stored on ``app.state.config`` by
    ``create_app``; this only updates the module-level fallback that
    ``get_config`` consults when no request-scoped app state is available.
    """
    global _config  # noqa: PLW0603
    _config = config


def get_config(request: Request) -> GatewayConfig:
    """Return the config for the current app from ``request.app.state``.

    Every shared resource (rate limiter, log writer, file store) lives on
    ``app.state``; config does too, so two apps in one process no longer share
    a single instance. Falls back to the legacy module-level shim only when the
    app state has no config attached.
    """
    config: GatewayConfig | None = getattr(request.app.state, "config", None)
    if config is None:
        config = _config
    if config is None:
        msg = "Config not initialized"
        raise RuntimeError(msg)
    return config


def reset_config() -> None:
    """Reset the legacy module-level config fallback. Intended for testing only."""
    global _config  # noqa: PLW0603
    _config = None
    # Router backends are cached per config signature and hold a per-process
    # decision cache, so a test that swaps config must not inherit the previous
    # one's trace stickiness.
    clear_router_backend_cache()


def get_enabled_features(request: Request) -> tuple[CoreFeature, ...]:
    """Return the core features this app enabled when it was built."""
    enabled: tuple[CoreFeature, ...] | None = getattr(request.app.state, "enabled_features", None)
    if enabled is None:
        msg = "Enabled features not initialized"
        raise RuntimeError(msg)
    return enabled


def extract_credential_token(request: Request) -> str:
    """Extract the caller's credential token from the request headers.

    Every mode reads the same headers through this one helper, so which
    deployment a caller talks to never changes how their key is presented;
    only who verifies the token differs (hybrid forwards it to the platform,
    the other modes check the local database).

    The canonical Otari-Key header carries the token directly. A ``Bearer ``
    prefix is accepted and stripped for back-compat, but is not required: a header
    named for the key holds the raw token, matching the ``x-api-key`` convention
    and the snippet the dashboard hands out. The standard Authorization header
    still requires the Bearer scheme. Finally the raw x-api-key header is honored
    (Anthropic-native clients). Surrounding whitespace is stripped, and a
    credential that is only whitespace is answered as missing rather than sent
    on to fail verification as a token of spaces.
    """
    token: str | None = None
    value = request.headers.get(API_KEY_HEADER)
    if value:
        # Leading whitespace comes off before the scheme check, so a padded
        # value still has its Bearer prefix recognized rather than kept as
        # part of the token.
        value = value.lstrip()
        token = value[7:] if value.startswith("Bearer ") else value
    else:
        auth_header = request.headers.get("Authorization")
        if auth_header:
            auth_header = auth_header.lstrip()
            # A blank value is a missing credential, not a scheme violation;
            # only a non-empty non-Bearer value is an invalid format.
            if auth_header and not auth_header.startswith("Bearer "):
                record_auth_failure("invalid_format")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid header format. Expected 'Bearer <token>'",
                )
            token = auth_header[7:]
        else:
            token = request.headers.get(X_API_KEY_HEADER)

    token = token.strip() if token else ""
    if token:
        return token

    record_auth_failure("missing_credentials")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=f"Missing {API_KEY_HEADER}, Authorization, or {X_API_KEY_HEADER} header",
    )


def misdirected_key_detail(host: str) -> str:
    """The body of a 421, naming where the presented key is served."""
    return f"This API key belongs to {host}. Send the request there instead."


async def _verify_and_update_api_key(db: AsyncSession, token: str, key_format: ApiKeyFormatPort) -> APIKey:
    """Verify API key token and update last_used_at.

    The bound key format says where the token is checked before anything is
    looked up. A key another deployment minted is answered 421 naming that
    deployment, and a key that claims this build's format and fails it is
    answered 401; neither costs a database round trip. Everything else, which
    for the open-source format is every key, is hashed and looked up whatever
    its shape, so a key minted elsewhere (a migrated platform key) authenticates
    on its hash and an unrecognized one gets the ordinary "Invalid API key" 401.
    """
    match key_format.route(token):
        case Misdirected(host=host):
            record_auth_failure("misdirected_key")
            raise HTTPException(
                status_code=status.HTTP_421_MISDIRECTED_REQUEST,
                detail=misdirected_key_detail(host),
            )
        case Malformed():
            record_auth_failure("invalid_format")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

    key_hash = hash_key(token)

    try:
        result = await db.execute(select(APIKey).where(APIKey.key_hash == key_hash))
    except DATABASE_ERRORS as e:
        record_auth_failure("db_error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        ) from e
    api_key = result.scalar_one_or_none()

    if not api_key:
        record_auth_failure("invalid_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    if not api_key.is_active:
        record_auth_failure("inactive_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is inactive",
        )

    expires_at = _as_utc(api_key.expires_at)
    if expires_at is not None and expires_at < datetime.now(UTC):
        record_auth_failure("expired_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
        )

    now = datetime.now(UTC)
    last_used_at = _as_utc(api_key.last_used_at)
    should_update_last_used = (
        last_used_at is None or (now - last_used_at).total_seconds() >= _LAST_USED_UPDATE_INTERVAL_SECONDS
    )

    if should_update_last_used:
        await _bump_last_used_at(api_key.id, now)

    return api_key


async def _bump_last_used_at(api_key_id: str, now: datetime) -> None:
    """Record an API key's last use on a short-lived, separate session.

    The bump runs outside the request's transaction so it never commits the
    caller's session or leaves it in a dirty state, and a failure is logged
    rather than swallowed. It is best-effort: throttled by
    ``_LAST_USED_UPDATE_INTERVAL_SECONDS`` and never fails the request.
    """
    try:
        async with create_session() as session:
            await session.execute(update(APIKey).where(APIKey.id == api_key_id).values(last_used_at=now))
            await session.commit()
    except DATABASE_ERRORS:
        # Widened past SQLAlchemyError for the same reason as the arms above:
        # this one promises never to fail the request, and a connect timeout
        # raises a bare TimeoutError.
        logger.warning("Failed to update last_used_at for API key %s", api_key_id, exc_info=True)


def _header_credentials_present(request: Request) -> bool:
    """Whether the request carries any of the header credential forms.

    Header credentials always win over the dashboard session cookie: an API
    client that sends a key gets exactly today's behavior (including failures),
    and the cookie is only consulted for requests that present nothing else,
    i.e. the browser-driven dashboard. ``get_session_identity`` applies that
    rule, so a cookie is never resolved for a request that carries a header
    credential.
    """
    return bool(
        request.headers.get(API_KEY_HEADER)
        or request.headers.get("Authorization")
        or request.headers.get(X_API_KEY_HEADER)
    )


# Sec-Fetch-Site values under which a cookie may authenticate a request:
# same-origin fetches (the dashboard itself) and non-site-initiated requests
# ("none", e.g. a direct navigation). "same-site" is deliberately excluded, so a
# sibling-subdomain page cannot ride the cookie.
_COOKIE_SAFE_FETCH_SITES = ("same-origin", "none")


async def get_session_identity(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> TenancyUser | None:
    """The identity a valid dashboard session cookie authenticates, or None.

    A session cookie *authenticates* the management API and names who is using
    it, so this is both the credential check and the identity resolution: the
    callers below treat a non-None result as authenticated, and
    ``get_current_identity`` reuses the same resolved identity. It is not by
    itself an answer to *what* the caller may do: a deployment-wide route asks
    ``require_deployment_operator`` on top, and a tenant-scoped one asks a
    service about the organization or workspace named. Declared as a dependency
    rather than called directly so FastAPI's per-request cache means one lookup
    however many of them a route pulls in.

    ``SameSite=Strict`` on the cookie is the primary CSRF control; the
    Sec-Fetch-Site check is belt-and-braces for clients that send the header.
    Standalone-only: hybrid mode has no dashboard or management API.
    """
    if config.is_hybrid_mode or _header_credentials_present(request):
        return None
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    fetch_site = request.headers.get("Sec-Fetch-Site")
    if fetch_site is not None and fetch_site not in _COOKIE_SAFE_FETCH_SITES:
        record_auth_failure("cross_site_cookie")
        return None
    try:
        return await resolve_dashboard_session(db, token)
    except DATABASE_ERRORS as exc:
        record_auth_failure("db_error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        ) from exc


async def is_valid_master_key(token: str, config: GatewayConfig, db: AsyncSession) -> bool:
    """Check if token matches the configured key or the current generated key."""
    if config.master_key is not None and secrets.compare_digest(token, config.master_key):
        return True
    if config.master_key is not None or not is_generated_master_key(token):
        return False
    stored_hash = await _load_generated_master_key_hash(config, db)
    if stored_hash is None:
        return False
    return secrets.compare_digest(hash_master_key(token), stored_hash)


async def _load_generated_master_key_hash(config: GatewayConfig, db: AsyncSession) -> str | None:
    """Load the shared generated-key hash, treating DB failures as retryable auth outages."""
    try:
        stored_hash = await load_master_key_hash(db)
    except DATABASE_ERRORS as exc:
        record_auth_failure("db_error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        ) from exc
    config._master_key_hash = stored_hash
    return stored_hash


def _api_key_format(request: Request, db: AsyncSession) -> ApiKeyFormatPort:
    """Resolve the key format for a verify path, off the request rather than a dependency.

    ``verify_api_key_or_master_key`` is called directly from routes that already
    hold the session, not only as a dependency, so the container is read from the
    app the request reached instead of being one more positional argument.
    """
    return get_container(request).resolve(ApiKeyFormatPort, db)


async def verify_api_key(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> APIKey:
    """Verify API key from the credential headers.

    ``config`` is not consulted, but stays in the signature: like its sibling
    ``verify_api_key_or_master_key``, this is called directly (not only via
    ``Depends``) by downstream deployments, so the three-argument shape is API.

    Args:
        request: FastAPI request object
        db: Database session
        config: Gateway configuration

    Returns:
        APIKey object if valid

    Raises:
        HTTPException: If key is invalid, inactive, or expired

    """
    token = extract_credential_token(request)
    return await _verify_and_update_api_key(db, token, _api_key_format(request, db))


async def verify_master_key(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
) -> str | None:
    """Verify master key from Otari-Key header or the dashboard session cookie.

    Args:
        request: FastAPI request object
        db: Database session
        config: Gateway configuration
        session_identity: The identity behind a dashboard session cookie, if any

    Returns:
        The raw master key when header-authenticated, or None when a dashboard
        session cookie authenticated the request (the raw key is not available).
        Which identity that session speaks for is read with
        ``get_current_identity``, not from this return value.

    Raises:
        HTTPException: If master key is not configured or invalid

    """
    if session_identity is not None:
        return None
    token = extract_credential_token(request)

    if config.master_key is None:
        stored_hash = await _load_generated_master_key_hash(config, db)
        if stored_hash is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Master key not configured. Set OTARI_MASTER_KEY environment variable.",
            )
        if is_generated_master_key(token) and secrets.compare_digest(hash_master_key(token), stored_hash):
            return token
    elif secrets.compare_digest(token, config.master_key):
        return token

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid master key",
    )


async def require_deployment_operator(
    db: Annotated[AsyncSession, Depends(get_db)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
    _master_key: Annotated[str | None, Depends(verify_master_key)],
) -> None:
    """Refuse a deployment-wide management request from a non-operator identity.

    ``verify_master_key`` answers *authenticated*, not *authorized*: a dashboard
    session clears it for any identity still active. (Address verification is a
    condition of minting a session, not of resolving one, so it says nothing
    about authority either way.) That is right for the tenant-scoped routers
    (`organizations.py`, `workspaces.py`, `org_provider_keys.py`, `admin.py`),
    which declare it as their authentication gate and then re-check the caller's
    role against the organization, workspace or deployment they are acting on.
    It is wrong for the deployment-wide routers, where clearing it *is* the
    whole authorization: `/api/v1/keys` mints a key into any workspace,
    `/api/v1/provider-credentials` holds process-global provider secrets, and
    `POST /api/v1/settings/master-key/rotate` replaces the deployment credential. On
    a single-operator deployment every login is that operator and the
    distinction is invisible; once mutually-untrusting tenants sign in to one
    process, a member of one organization holding master-key authority is a
    cross-organization breach (otari-ai#1880).

    A **header master key** is the deployment credential itself, so it passes; it
    names nobody, and ``get_current_identity`` resolves it to the bootstrap
    operator. A **session** is put to
    ``DeploymentUserService.has_administration_access``, which `/api/v1/admin`
    already treats as the answer to "may this caller act deployment-wide": a
    superuser, or the bootstrap operator whatever its flag says. Reused rather
    than re-derived so the routers guarded here and the account administration
    that can grant the authority cannot come to disagree about who holds it.

    Declared on the router rather than per route, and as a dependency rather
    than a check inside each handler: a route added to one of those routers
    later inherits the gate instead of being reachable with no credential at
    all until someone notices the missing decorator. ``Depends`` caching means
    the master-key verification underneath still runs once per request however
    many of these a route pulls in. A module that holds an exception (a catalog
    read, the tool-settings reader, external-event ingestion) puts it on a router
    of its own, so admitting a non-operator is spelled at a router instead of
    hidden in one route's decorator.
    """
    if session_identity is not None and not await DeploymentUserService(db).has_administration_access(
        session_identity
    ):
        record_auth_failure("not_deployment_operator")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint requires deployment operator access.",
        )


async def verify_api_key_or_master_key(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> tuple[APIKey | None, bool]:
    """Verify either API key or master key from Otari-Key header.

    Deliberately does **not** consult the dashboard session cookie. It used to,
    on the same "a session grants master-key authority" premise
    ``require_deployment_operator`` exists to retire, and here that premise was
    worse than on the management plane: ``is_master_key`` is what makes
    ``resolve_request_context`` bill and resolve credentials through the
    deployment's *default* workspace, so a signed-in member of any organization
    could spend another organization's BYO provider credential on a completion,
    or file usage rows into a tenant they do not belong to, without holding a key
    at all (otari-ai#1880).

    A browser has no reason to reach this plane: the dashboard mentions these
    endpoints in its copy and calls none of them. The three catalog reads it does
    call take :func:`verify_catalog_reader` instead.

    Args:
        request: FastAPI request object
        db: Database session
        config: Gateway configuration

    Returns:
        Tuple of (APIKey object or None, is_master_key boolean)

    Raises:
        HTTPException: If key is invalid, inactive, or expired

    """
    token = extract_credential_token(request)

    if await is_valid_master_key(token, config, db):
        return None, True

    api_key = await _verify_and_update_api_key(db, token, _api_key_format(request, db))
    return api_key, False


async def verify_catalog_reader(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
) -> tuple[APIKey | None, bool]:
    """As :func:`verify_api_key_or_master_key`, and a dashboard session also reads.

    The narrow exception to the rule above, for the catalog reads that describe
    the deployment rather than act on it: ``GET /api/v1/models``, ``GET /api/v1/pricing``,
    ``GET /api/v1/tools`` and ``GET /api/v1/providers/catalog`` (with their by-id
    variants). The dashboard's Models and Pricing pages are built on these, so a
    session has to reach them; they call no provider, write nothing, and bill
    nothing, so reaching them deployment-wide costs a signed-in caller's own
    organization nothing.

    The provider catalog is the one of these a *tenant* rather than an operator
    needs: it names the providers any-llm knows, which the organization
    provider-key form offers as the BYO choices, so an owner or admin who reaches
    no operator route still has to read it.

    Split out rather than left as a branch inside the other dependency so that
    adding a route to this plane defaults to refusing the cookie. The four
    routers that serve these reads declare it on the router for the same reason,
    so admitting a session is spelled where the route is mounted rather than in
    one route's decorator.
    """
    if session_identity is not None:
        return None, True
    return await verify_api_key_or_master_key(request, db, config)


async def verify_catalog_reader_or_public(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
) -> tuple[APIKey | None, bool] | None:
    """As :func:`verify_catalog_reader`, and a visitor reads too while the catalog is public.

    ``None`` is the anonymous caller, admitted only while ``public_catalog`` is on
    and only when the request carries no credential at all: a credential that is
    present and wrong is refused as it always was, never downgraded to a visitor.
    The route is what narrows an anonymous read (the configured instances, the
    deployment price list, no tenant rows); this only decides who is asking.

    Throttled per client address on its own budget,
    ``public_catalog_rate_limit_per_minute``, the way the public auth routes
    are on theirs: ``rate_limit_rpm`` keys on an authenticated user and covers
    no anonymous path, so it is not what stands between an open catalog and a
    scraper.

    Two things that throttle is not, both documented beside the setting in
    ``docs/configuration.md``. The address is the socket's, and the CLI starts
    uvicorn without proxy headers, so behind a reverse proxy every visitor
    shares one bucket; a deployment that terminates TLS elsewhere throttles
    there. And the counter is per process, so N workers serve N times the
    configured number.
    """
    if session_identity is not None:
        return None, True
    if _header_credentials_present(request) or not config.public_catalog:
        return await verify_api_key_or_master_key(request, db, config)
    limiter = getattr(request.app.state, "public_catalog_rate_limiter", None)
    if limiter is not None:
        # The limiter raises its own 429; the key is the address, since a
        # visitor has no other identity.
        limiter.check(request.client.host if request.client is not None else "unknown")
    return None


async def get_db_if_needed(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> AsyncGenerator[AsyncSession | None, None]:
    """Get a database session in standalone mode, otherwise return None."""
    if config.is_hybrid_mode:
        yield None
        return

    # A bare ``async for`` leaves ``get_db`` open when an error or cancellation
    # is thrown in at teardown, so its session would hold a pooled connection
    # until garbage collection.
    async with aclosing(get_db()) as sessions:
        async for db in sessions:
            yield db


def build_sandbox_file_bridge(
    *,
    raw_request: Request,
    config: GatewayConfig,
    db: AsyncSession | None,
    user_id: str | None,
    workspace_id: uuid.UUID | None,
    inputs: list[StagedFile],
) -> SandboxFileBridge | None:
    """The file bridge a completion request's sandbox session gets, or ``None``.

    Built by the route once the billed user and workspace are resolved, over the
    request's own session. ``None`` in hybrid mode, which has no local database
    or file store to hold what a run produces, and when files are disabled.
    Produced files are announced under ``public_base_url`` where the deployment
    knows its address, and otherwise under the one the request arrived on.
    """
    file_store = getattr(raw_request.app.state, "file_store", None)
    if db is None or not config.files_enabled or file_store is None or user_id is None or workspace_id is None:
        return None
    base = (config.public_base_url or str(raw_request.base_url)).rstrip("/")
    return SandboxFileBridge(
        file_store=file_store,
        config=config,
        uow=UnitOfWork(db),
        user_id=user_id,
        workspace_id=workspace_id,
        inputs=inputs,
        base_url=f"{base}{API_ROOT}/files",
    )


def get_unit_of_work(db: Annotated[AsyncSession, Depends(get_db)]) -> UnitOfWork:
    """Return the request's Unit of Work over its session.

    Gotcha: the rest of the request writes to this same session.
    A block's commit also stores what that code staged outside a block, and its rollback discards it.
    Standalone and hosted only: a hybrid gateway has no local database, so it has no Unit of Work.
    """
    return UnitOfWork(db)


async def get_current_identity(
    db: Annotated[AsyncSession, Depends(get_db)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
    _master_key: Annotated[str | None, Depends(verify_master_key)],
) -> TenancyUser:
    """Resolve the tenancy identity acting on this request.

    A dashboard session names the identity it was minted for, so a
    cookie-authenticated request resolves that identity and, through its
    ``active_organization_id``, the organization it is acting in. No second
    lookup: ``verify_master_key`` already resolved it through the same cached
    dependency to decide the cookie authenticates at all.

    A header master key names nobody, so it falls back to the deployment's
    bootstrap operator, provisioned on first use (otari-ai#1716 option A, see
    `gateway.services.tenancy.provisioning_service`). That is also the identity
    master-key sign-in binds a session to, so both credentials resolve to the
    same operator on a standalone deployment; the difference matters once the
    per-user sign-in flows mint sessions for other identities.

    Depending on ``verify_master_key`` keeps the credential check exactly where
    the rest of the management API has it, and keeps a request with no credential
    at all from provisioning anything.
    """
    if session_identity is not None:
        return session_identity
    return await ensure_bootstrap_identity(db, membership_listener=WorkspaceBudgetDefaultService(db))


CurrentIdentity = Annotated[TenancyUser, Depends(get_current_identity)]


# =============================================================================
# Composition root
# =============================================================================
#
# Every port is resolved here and nowhere else: a dependency names the port and
# asks the container for whichever adapter this build bound to it, so no route
# or service ever names a concrete adapter (ARCHITECTURE.md, rule 5).


def get_container(request: Request) -> Container:
    """Return the composition-root container this app was built with."""
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:
        msg = "Composition root not initialized"
        raise RuntimeError(msg)
    return container


ContainerDep = Annotated[Container, Depends(get_container)]
# ``get_db_if_needed`` and not ``get_db``: hybrid mode runs with no local
# database at all, so a port resolved on a hybrid request gets ``None`` rather
# than a session that cannot be opened. Every core adapter ignores it.
#
# The consequence to know before an adapter writes anything: FastAPI caches a
# dependency per callable, so this shares one session with a route that also
# takes ``get_db_if_needed`` (the data-plane routes do) and opens a *second,
# independent* one for a route that takes ``get_db`` (the management-plane
# routes do). A port's "joins the caller's unit of work" therefore holds only
# for the first kind. A route that means to commit a port's writes with its own
# must take its session from ``get_db_if_needed`` too.
PortSessionDep = Annotated[AsyncSession | None, Depends(get_db_if_needed)]


def get_api_key_format_port(db: PortSessionDep, container: ContainerDep) -> ApiKeyFormatPort:
    """Resolve the key-format adapter this build bound at startup."""
    return container.resolve(ApiKeyFormatPort, db)


def get_billing_port(db: PortSessionDep, container: ContainerDep) -> BillingPort:
    """Resolve the billing adapter this build bound at startup."""
    return container.resolve(BillingPort, db)


def get_entitlement_port(db: PortSessionDep, container: ContainerDep) -> EntitlementPort:
    """Resolve the entitlement adapter this build bound at startup."""
    return container.resolve(EntitlementPort, db)


# ``get_db`` rather than ``PortSessionDep``, for the reason
# ``get_telemetry_storage_port`` below gives: both surfaces resolving this port
# (signup, and a member minting their own key) are standalone-only and already
# hold a session from ``get_db``, and FastAPI caches a dependency per callable.
# Naming the same one hands the adapter the caller's session instead of opening
# a second, independent one against the same database for the same request,
# which is what an adapter writing an outbox row or a "notified" stamp would
# need to land its write in a transaction someone commits.
def get_growth_signal_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    container: ContainerDep,
) -> GrowthSignalPort:
    """Resolve the growth-signal adapter this build bound at startup."""
    return container.resolve(GrowthSignalPort, db)


# ``get_db`` rather than ``PortSessionDep``, for the reason
# ``get_telemetry_storage_port`` below gives: the only surface resolving this is
# the OAuth sign-in route, which is standalone-only and already holds a session
# from ``get_db``, and FastAPI caches a dependency per callable. Naming the same
# one hands the adapter the caller's session instead of opening a second,
# independent one against the same database for the same request. That sharing
# is load-bearing here and not just tidy: the adapter writes (it links a
# provider, and may stamp a verification) and deliberately does not commit, so
# its writes have to be in the transaction the route commits or they are in one
# nobody does.
def get_identity_provider_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    container: ContainerDep,
) -> IdentityProviderPort:
    """Resolve the identity adapter this build bound at startup."""
    return container.resolve(IdentityProviderPort, db)


def get_model_provider_port(db: PortSessionDep, container: ContainerDep) -> ModelProviderPort:
    """Resolve the model-provider adapter this build bound at startup."""
    return container.resolve(ModelProviderPort, db)


# Deliberately ``get_db`` and not ``PortSessionDep``: every surface that
# resolves this port (the OTLP receiver, the telemetry read and purge
# endpoints, user deletion) is standalone-only and already holds a session from
# ``get_db``, and FastAPI caches a dependency per callable. Naming the same one
# hands the adapter the caller's session instead of opening a second,
# independent one against the same database for the same request.
def get_telemetry_storage_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    container: ContainerDep,
) -> TelemetryStoragePort:
    """Resolve the telemetry-storage adapter this build bound at startup."""
    return container.resolve(TelemetryStoragePort, db)


def get_overview_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OverviewService:
    """Build the dashboard overview's summary service on the request's session.

    Assembled here rather than in the route, because a route does not name a
    session type (``scripts/check_architecture.py``, rule 12).
    """
    return OverviewService(
        OverviewRepository(db),
        OrganizationService(db, membership_listener=None),
        DeploymentUserService(db),
        # The listener is for writes; this service only reads, and the same
        # pairing is what `routes/workspaces.py` builds.
        WorkspaceService(db, membership_listener=WorkspaceBudgetDefaultService(db)),
    )


OverviewServiceDep = Annotated[OverviewService, Depends(get_overview_service)]


def get_budget_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetService:
    """Build the request's budget service on the request's Unit of Work."""
    return BudgetService(
        uow,
        BudgetRepositories.on(uow),
        OrganizationService(db, membership_listener=None),
        ApiKeyService(ApiKeyRepository(uow)),
    )


BudgetServiceDep = Annotated[BudgetService, Depends(get_budget_service)]

ApiKeyFormatPortDep = Annotated[ApiKeyFormatPort, Depends(get_api_key_format_port)]
BillingPortDep = Annotated[BillingPort, Depends(get_billing_port)]
EntitlementPortDep = Annotated[EntitlementPort, Depends(get_entitlement_port)]
GrowthSignalPortDep = Annotated[GrowthSignalPort, Depends(get_growth_signal_port)]
IdentityProviderPortDep = Annotated[IdentityProviderPort, Depends(get_identity_provider_port)]
ModelProviderPortDep = Annotated[ModelProviderPort, Depends(get_model_provider_port)]


def get_org_provider_model_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    model_provider: ModelProviderPortDep,
) -> OrgProviderModelService:
    """Build the offered-models service on the request's session and unit of work.

    The service itself names neither the session nor SQLAlchemy, so its
    repositories and its cache-refresh callable are assembled here. The unit of
    work and the services built on the session are over the *same* session (see
    ``deps.get_unit_of_work``), so a block's commit also settles what they staged.
    """
    return OrgProviderModelService(
        uow,
        config=config,
        organizations=OrganizationService(db, membership_listener=None),
        provider_keys=OrgProviderKeyService(db),
        org_pricing=OrganizationPricingService(db, config, model_provider=model_provider),
        models=OrgProviderKeyModelRepository(uow),
        keys=OrgProviderKeyRepository(db),
        refresh_overlay=lambda: refresh_org_provider_cache(db),
    )


OrgProviderModelServiceDep = Annotated[OrgProviderModelService, Depends(get_org_provider_model_service)]
TelemetryStoragePortDep = Annotated[TelemetryStoragePort, Depends(get_telemetry_storage_port)]


def require_capability(capability: str) -> Callable[[EntitlementPort], Awaitable[None]]:
    """Build a dependency that refuses a request unless the deployment is entitled.

    The gate every contributed router is mounted behind. A refusal carries the
    same status and body as a request for a path nothing serves (404 with the
    framework's wording), so the response alone does not reveal whether the
    surface exists. The surface itself stays mounted: it appears in the OpenAPI
    document, and a request with an unsupported method meets the router's 405
    before this gate. The refusal is logged, so an operator can tell "not
    entitled" from "not mounted" where the client cannot.
    """

    async def _require(entitlements: EntitlementPortDep) -> None:
        if capability not in await entitlements.entitlements():
            logger.warning("Refused a request to a surface gated on capability %r: not entitled", capability)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    return _require


def get_log_writer(request: Request) -> LogWriter:
    writer: LogWriter = request.app.state.log_writer
    return writer


def get_file_store(request: Request) -> FileStore:
    """Return the configured blob store for uploaded files (standalone mode)."""
    store: FileStore = request.app.state.file_store
    return store


async def _caller_organization_id(
    db: Annotated[AsyncSession, Depends(get_db)],
    identity: CurrentIdentity,
) -> uuid.UUID:
    """The organization this request acts in.

    A key is minted, listed and revoked inside one organization, and so is a
    spend identity read, so every deployment-wide route that touches a tenant's
    rows resolves the caller's organization before it does. A dashboard session
    names the identity behind it and resolves that identity's active
    organization, which is what ``POST /api/v1/organizations/me/switch`` moves; a
    header master key names nobody, resolves the bootstrap operator, and
    therefore acts in the default organization. That is the same rule
    ``services/workspace_scope`` already documents for a deployment-wide write,
    so an operator running several organizations behind one gateway works in the
    one they are currently in rather than across all of them (otari#817).
    """
    return (await OrganizationService(db, membership_listener=None).get_active_organization_for_user(identity)).id


CallerOrganization = Annotated[uuid.UUID, Depends(_caller_organization_id)]


__all__ = [
    "BillingPortDep",
    "ContainerDep",
    "CallerOrganization",
    "CurrentIdentity",
    "EntitlementPortDep",
    "OverviewServiceDep",
    "GrowthSignalPortDep",
    "IdentityProviderPortDep",
    "ModelProviderPortDep",
    "OrgProviderModelServiceDep",
    "TelemetryStoragePortDep",
    "get_config",
    "get_container",
    "get_telemetry_storage_port",
    "get_current_identity",
    "get_db",
    "get_session_identity",
    "reset_config",
    "set_config",
    "get_db_if_needed",
    "get_file_store",
    "get_log_writer",
    "is_valid_master_key",
    "require_capability",
    "extract_credential_token",
    "verify_api_key",
    "verify_api_key_or_master_key",
    "verify_catalog_reader",
    "verify_master_key",
]
