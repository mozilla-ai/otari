import secrets
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.auth.models import hash_key
from gateway.container import Container
from gateway.core.config import API_KEY_HEADER, X_API_KEY_HEADER, GatewayConfig
from gateway.core.database import create_session, get_db
from gateway.log_config import logger
from gateway.metrics import record_auth_failure
from gateway.models.entities import APIKey
from gateway.models.tenancy import User as TenancyUser
from gateway.ports.billing_port import BillingPort
from gateway.ports.entitlement_port import EntitlementPort
from gateway.ports.growth_signal_port import GrowthSignalPort
from gateway.ports.identity_provider_port import IdentityProviderPort
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.ports.telemetry_storage_port import TelemetryStoragePort
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, resolve_dashboard_session
from gateway.services.file_store import FileStore
from gateway.services.log_writer import LogWriter
from gateway.services.master_key_service import hash_master_key, is_generated_master_key, load_master_key_hash
from gateway.services.routing import clear_router_backend_cache
from gateway.services.tenancy.deployment_user_service import DeploymentUserService
from gateway.services.tenancy.provisioning_service import ensure_bootstrap_identity

# Legacy module-level fallback. Config now lives on ``app.state.config`` (set in
# ``create_app``); ``get_config`` reads from the request's app state and only
# falls back to this shim for callers that set it directly (see ``set_config``).
_config: GatewayConfig | None = None
_LAST_USED_UPDATE_INTERVAL_SECONDS = 300


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


def _extract_bearer_token(request: Request, config: GatewayConfig) -> str:
    """Extract the API token from the request headers.

    The canonical Otari-Key header carries the token directly. A ``Bearer ``
    prefix is accepted and stripped for back-compat, but is not required: a header
    named for the key holds the raw token, matching the ``x-api-key`` convention
    and the snippet the dashboard hands out. The standard Authorization header
    still requires the Bearer scheme. Finally the raw x-api-key header is honored
    (Anthropic-native clients).
    """
    value = request.headers.get(API_KEY_HEADER)
    if value:
        return value[7:] if value.startswith("Bearer ") else value

    auth_header = request.headers.get("Authorization")
    if auth_header:
        if not auth_header.startswith("Bearer "):
            record_auth_failure("invalid_format")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid header format. Expected 'Bearer <token>'",
            )
        return auth_header[7:]

    raw_token = request.headers.get(X_API_KEY_HEADER)
    if raw_token:
        return raw_token

    record_auth_failure("missing_credentials")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=f"Missing {API_KEY_HEADER}, Authorization, or {X_API_KEY_HEADER} header",
    )


async def _verify_and_update_api_key(db: AsyncSession, token: str) -> APIKey:
    """Verify API key token and update last_used_at.

    The token's shape is not checked: any presented token is hashed and looked
    up, so a key minted elsewhere (a migrated platform key) authenticates on its
    hash and an unrecognized one gets the ordinary "Invalid API key" 401.
    """
    key_hash = hash_key(token)

    try:
        result = await db.execute(select(APIKey).where(APIKey.key_hash == key_hash))
    except SQLAlchemyError as e:
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
    except SQLAlchemyError:
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
    except SQLAlchemyError as exc:
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
    except SQLAlchemyError as exc:
        record_auth_failure("db_error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        ) from exc
    config._master_key_hash = stored_hash
    return stored_hash


async def verify_api_key(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> APIKey:
    """Verify API key from Otari-Key header.

    Args:
        request: FastAPI request object
        db: Database session
        config: Gateway configuration

    Returns:
        APIKey object if valid

    Raises:
        HTTPException: If key is invalid, inactive, or expired

    """
    token = _extract_bearer_token(request, config)
    return await _verify_and_update_api_key(db, token)


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
    token = _extract_bearer_token(request, config)

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
    session clears it for any active, email-verified identity. That is right for
    the tenant-scoped routers (`organizations.py`, `workspaces.py`,
    `org_provider_keys.py`, `admin.py`), which declare it as their authentication
    gate and then re-check the caller's role against the organization, workspace
    or deployment they are acting on. It is wrong for the deployment-wide
    routers, where clearing it *is* the whole authorization: `/v1/keys` mints a
    key into any workspace, `/v1/provider-credentials` holds process-global
    provider secrets, and `POST /v1/settings/master-key/rotate` replaces the
    deployment credential. On a single-operator deployment every login is that
    operator and the distinction is invisible; once mutually-untrusting tenants
    sign in to one process, a member of one organization holding master-key
    authority is a cross-organization breach (otari-ai#1880).

    A **header master key** is the deployment credential itself, so it passes; it
    names nobody, and ``get_current_identity`` resolves it to the bootstrap
    operator. A **session** is put to
    ``DeploymentUserService.has_administration_access``, which `/v1/admin`
    already treats as the answer to "may this caller act deployment-wide": a
    superuser, or the bootstrap operator whatever its flag says. Reused rather
    than re-derived so the routers guarded here and the account administration
    that can grant the authority cannot come to disagree about who holds it.

    Declared as a dependency and not a check inside each handler so it composes
    the way the router it guards already declares auth, and so ``Depends``
    caching means the master-key verification underneath runs once per request
    however many of these a route pulls in.
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
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
) -> tuple[APIKey | None, bool]:
    """Verify either API key or master key from Otari-Key header.

    A valid dashboard session cookie also grants master-key authority, but only
    when the request carries no header credentials at all.

    Args:
        request: FastAPI request object
        db: Database session
        config: Gateway configuration
        session_identity: The identity behind a dashboard session cookie, if any

    Returns:
        Tuple of (APIKey object or None, is_master_key boolean)

    Raises:
        HTTPException: If key is invalid, inactive, or expired

    """
    if session_identity is not None:
        return None, True

    token = _extract_bearer_token(request, config)

    if await is_valid_master_key(token, config, db):
        return None, True

    api_key = await _verify_and_update_api_key(db, token)
    return api_key, False


async def get_db_if_needed(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> AsyncGenerator[AsyncSession | None, None]:
    """Get a database session in standalone mode, otherwise return None."""
    if config.is_hybrid_mode:
        yield None
        return

    async for db in get_db():
        yield db


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
    return await ensure_bootstrap_identity(db)


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


def get_billing_port(db: PortSessionDep, container: ContainerDep) -> BillingPort:
    """Resolve the billing adapter this build bound at startup."""
    return container.resolve(BillingPort, db)


def get_entitlement_port(db: PortSessionDep, container: ContainerDep) -> EntitlementPort:
    """Resolve the entitlement adapter this build bound at startup."""
    return container.resolve(EntitlementPort, db)


def get_growth_signal_port(db: PortSessionDep, container: ContainerDep) -> GrowthSignalPort:
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


BillingPortDep = Annotated[BillingPort, Depends(get_billing_port)]
EntitlementPortDep = Annotated[EntitlementPort, Depends(get_entitlement_port)]
GrowthSignalPortDep = Annotated[GrowthSignalPort, Depends(get_growth_signal_port)]
IdentityProviderPortDep = Annotated[IdentityProviderPort, Depends(get_identity_provider_port)]
ModelProviderPortDep = Annotated[ModelProviderPort, Depends(get_model_provider_port)]
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


__all__ = [
    "BillingPortDep",
    "ContainerDep",
    "CurrentIdentity",
    "EntitlementPortDep",
    "GrowthSignalPortDep",
    "IdentityProviderPortDep",
    "ModelProviderPortDep",
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
    "verify_api_key",
    "verify_api_key_or_master_key",
    "verify_master_key",
]
