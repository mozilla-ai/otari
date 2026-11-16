"""Verifying the credential a request presents, and what it is then allowed to reach."""

import secrets
import time
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import (
    get_config,
    get_container,
)
from gateway.api.key_scopes import UndeclaredKeyScope, declared_key_scope
from gateway.auth.models import hash_key
from gateway.core.config import (
    API_KEY_HEADER,
    X_API_KEY_HEADER,
    GatewayConfig,
)
from gateway.core.database import (
    DATABASE_ERRORS,
    create_session,
    describe_database_error,
    get_db,
    release_session,
)
from gateway.exceptions.api_keys_exceptions import KeyScopeMissingError
from gateway.log_config import logger
from gateway.metrics import REGISTRY, Counter
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import User as TenancyUser
from gateway.ports.api_key_format_port import ApiKeyFormatPort, Malformed, Misdirected
from gateway.repositories.api_keys import ApiKeyRepository, FederationRuleRepository
from gateway.services.api_keys import (
    FederatedTokenVerifier,
    FederationRefusedError,
    FederationUnavailableError,
    is_access_token,
    key_refusal,
    match_federation_rules,
    read_access_token,
    unverified_issuer,
)
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, resolve_dashboard_session
from gateway.services.master_key_service import hash_master_key, is_generated_master_key, load_master_key_hash
from gateway.services.secret_box import configured_secret_keys
from gateway.services.tenancy.deployment_user_service import DeploymentAdministrationAccess
from gateway.services.workspace_scope import (
    organization_for_key_id,
)
from gateway.types.key_scope import END_USER_SCOPES

_LAST_USED_UPDATE_INTERVAL_SECONDS = 300

# How a key that is not a service key is refused where only a service key may go.
SERVICE_KEY_REQUIRED_DETAIL = "This endpoint requires a service key."

AUTH_FAILURES = Counter(
    "gateway_auth_failures",
    "Total number of authentication failures",
    ["reason"],
    registry=REGISTRY,
)


def record_auth_failure(reason: str) -> None:
    """Record an authentication failure."""
    AUTH_FAILURES.labels(reason=reason).inc()


def _invalid_api_key(reason: str) -> HTTPException:
    """Every refusal answers alike, so a caller cannot tell which check failed; ``reason`` goes to the metric."""
    record_auth_failure(reason)
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _authentication_unavailable(reason: str) -> HTTPException:
    record_auth_failure(reason)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Authentication temporarily unavailable, please retry",
    )


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


def extract_credential_token(request: Request) -> str:
    """Extract the caller's credential token from the request headers.

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
            raise _invalid_api_key("invalid_format")

    key_hash = hash_key(token)

    try:
        api_key = await ApiKeyRepository(db).get_by_hash(key_hash)
    except DATABASE_ERRORS as e:
        raise _authentication_unavailable("db_error") from e

    if not api_key:
        raise _invalid_api_key("invalid_key")

    return await _admit_api_key(api_key)


async def _admit_api_key(api_key: APIKey) -> APIKey:
    """The checks a key passes once found, shared so a secret, an identity token and an access token admit alike."""
    now = datetime.now(UTC)
    refusal = key_refusal(api_key, now)
    if refusal is not None:
        reason, detail = refusal
        record_auth_failure(reason)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)

    last_used_at = _as_utc(api_key.last_used_at)
    should_update_last_used = (
        last_used_at is None or (now - last_used_at).total_seconds() >= _LAST_USED_UPDATE_INTERVAL_SECONDS
    )

    if should_update_last_used:
        await _bump_last_used_at(api_key.id, now)

    return api_key


def federated_token_verifier(request: Request, config: GatewayConfig) -> FederatedTokenVerifier:
    """The app's token verifier for the configured issuers, built on first use and when they change.

    Kept on the app rather than per request, because it caches each issuer's signing keys.
    """
    verifier: FederatedTokenVerifier | None = getattr(request.app.state, "federated_token_verifier", None)
    if (
        verifier is None
        or verifier.issuers != tuple(config.federation_issuers)
        or verifier.audiences != tuple(config.federation_audiences)
        or verifier.allow_private_hosts != config.federation_allow_private_hosts
        or verifier.member_issuers != config.federation_member_rules
    ):
        verifier = FederatedTokenVerifier(
            config.federation_issuers,
            audiences=config.federation_audiences,
            allow_private_hosts=config.federation_allow_private_hosts,
            member_issuers=config.federation_member_rules,
        )
        request.app.state.federated_token_verifier = verifier
    return verifier


async def _verify_federated_token(request: Request, db: AsyncSession, token: str, config: GatewayConfig) -> APIKey:
    """Authenticate an identity token as the key whose federation rule it matches.

    Every refusal answers the same 401 as an unknown key, so a caller cannot
    tell which check failed; the metric records which one did. Once a rule
    matches, the key is admitted exactly as when its secret is presented.
    """
    try:
        identity = await federated_token_verifier(request, config).verify(token)
        match = match_federation_rules(
            await FederationRuleRepository(db).candidates(identity.issuer, identity.audiences), identity
        )
        api_key = await ApiKeyRepository(db).get(match.api_key_id)
    except FederationUnavailableError as exc:
        raise _authentication_unavailable("federation_unavailable") from exc
    except FederationRefusedError as exc:
        raise _invalid_api_key(exc.reason) from exc
    except DATABASE_ERRORS as exc:
        raise _authentication_unavailable("db_error") from exc
    if api_key is None:
        raise _invalid_api_key("federation_no_rule")
    logger.debug(
        "Federated token from %s matched rule(s) %s of API key %s",
        identity.issuer,
        ",".join(match.rule_ids),
        api_key.id,
    )
    return await _admit_api_key(api_key)


async def _verify_access_token(db: AsyncSession, token: str, config: GatewayConfig) -> APIKey:
    """Authenticate an access token from ``POST /oauth/token`` as the key of the rule it names.

    The rule and the key are reloaded on every request, so deleting the rule, deactivating the key, dropping an
    operator's rule's issuer from ``federation_issuers`` or turning ``federation_member_rules`` off for a member's rule
    ends the token at once. Refusals answer the same 401 as an unknown key.
    """
    try:
        claims = read_access_token(token, configured_secret_keys(), now=time.time())
        rule = await FederationRuleRepository(db).get(claims.rule_id)
        if rule is None or rule.api_key_id != claims.api_key_id:
            raise FederationRefusedError("federation_rule_gone")
        if rule.member_managed:
            if not config.federation_member_rules:
                raise FederationRefusedError("federation_member_rules_off")
        elif all(entry.issuer != rule.issuer for entry in config.federation_issuers):
            raise FederationRefusedError("federation_unknown_issuer")
        api_key = await ApiKeyRepository(db).get(claims.api_key_id)
    except FederationRefusedError as exc:
        raise _invalid_api_key(exc.reason) from exc
    except DATABASE_ERRORS as exc:
        raise _authentication_unavailable("db_error") from exc
    if api_key is None:
        raise _invalid_api_key("federation_rule_gone")
    return await _admit_api_key(api_key)


async def _verify_credential(request: Request, db: AsyncSession, token: str, config: GatewayConfig) -> APIKey:
    """Verify a presented credential as an API key.

    With federation on, the prefix of an access token or the shape of a JWT routes it before anything is parsed:
    an access token, then a raw identity token, which only an operator's rule on an allow-listed issuer can match.
    Everything else is a key secret. A key that authenticates is then held to the scope the route declares.
    """
    if config.federation_enabled and is_access_token(token):
        api_key = await _verify_access_token(db, token, config)
    elif config.federation_enabled and config.federation_issuers and unverified_issuer(token) is not None:
        api_key = await _verify_federated_token(request, db, token, config)
    else:
        api_key = await _verify_and_update_api_key(db, token, _api_key_format(request, db))
    _require_route_scope(request, api_key)
    return api_key


def _require_route_scope(request: Request, api_key: APIKey) -> None:
    """Refuse ``api_key`` on the matched route unless it holds the scope the route declares.

    A route that declares none is refused to every key: it was added without
    saying which scope it needs, and failing closed is what keeps a new route
    from being reachable by every key until someone notices.
    """
    route = request.scope.get("route")
    required = declared_key_scope(route)
    if isinstance(required, UndeclaredKeyScope):
        record_auth_failure("undeclared_key_scope")
        logger.error("Refused an API key on %s, which declares no key scope", getattr(route, "path", "?"))
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This endpoint does not accept API keys.")
    if required in END_USER_SCOPES and not api_key.is_service_key:
        # Said this way because no scope would help: only a service key may hold this one.
        record_auth_failure("not_service_key")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=SERVICE_KEY_REQUIRED_DETAIL)
    if required is not None and required not in api_key.scopes:
        record_auth_failure("missing_key_scope")
        logger.info(
            "Refused API key %s on %s: it lacks the %s scope", api_key.id, getattr(route, "path", "?"), required
        )
        missing = KeyScopeMissingError(required, "this endpoint")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=missing.message, headers=missing.challenge)


async def _bump_last_used_at(api_key_id: str, now: datetime) -> None:
    """Record an API key's last use on a short-lived, separate session.

    The bump runs outside the request's transaction so it never commits the
    caller's session or leaves it in a dirty state, and a failure is logged
    rather than swallowed. It is best-effort: throttled by
    ``_LAST_USED_UPDATE_INTERVAL_SECONDS`` and never fails the request.
    """
    try:
        async with create_session() as session:
            await ApiKeyRepository(session).touch_last_used(api_key_id, now)
            await session.commit()
    except DATABASE_ERRORS as exc:
        # Widened past SQLAlchemyError for the same reason as the arms above:
        # this one promises never to fail the request, and a connect timeout
        # raises a bare TimeoutError.
        # codeql[py/clear-text-logging-sensitive-data]
        logger.warning("Failed to update last_used_at for API key %s (%s)", api_key_id, describe_database_error(exc))


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
# sibling-subdomain page cannot ride the cookie; ``cookie_may_authenticate``
# admits it only from an origin the deployment itself listed.
_COOKIE_SAFE_FETCH_SITES = ("same-origin", "none")


def cookie_may_authenticate(request: Request, config: GatewayConfig) -> bool:
    """Whether the session cookie on this request may authenticate it.

    A same-site request is admitted only when its ``Origin`` is one of
    ``cors_allow_origins``: that list is where an operator names the origin an
    edge serves the dashboard from, so a dashboard on a sibling host of this
    process can hold a session here while every other sibling stays refused.
    A ``*`` entry never matches, since a wildcard is not an origin and CORS
    sends no credentials under one either.
    """
    fetch_site = request.headers.get("Sec-Fetch-Site")
    if fetch_site is None or fetch_site in _COOKIE_SAFE_FETCH_SITES:
        return True
    if fetch_site != "same-site":
        return False
    origin = request.headers.get("Origin", "")
    return bool(origin) and origin != "*" and origin in config.cors_allow_origins


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
    """
    if _header_credentials_present(request):
        return None
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    if not cookie_may_authenticate(request, config):
        record_auth_failure("cross_site_cookie")
        return None
    try:
        return await resolve_dashboard_session(db, token)
    except DATABASE_ERRORS as exc:
        raise _authentication_unavailable("db_error") from exc


def refuse_ended_session(request: Request, config: GatewayConfig) -> None:
    """Refuse a request whose only credential is a session cookie that no longer authenticates.

    Called once ``get_session_identity`` has answered None. Without this the caller would be told it sent no
    API key header, which describes an API client rather than a browser whose session ended (it was revoked,
    expired, or its account deactivated). A cookie refused for being cross-site is left to the usual answer.
    """
    if (
        _header_credentials_present(request)
        or not request.cookies.get(SESSION_COOKIE_NAME)
        or not cookie_may_authenticate(request, config)
    ):
        return
    record_auth_failure("session_ended")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Your dashboard session has ended. Sign in again.",
    )


async def is_valid_master_key(token: str, config: GatewayConfig, db: AsyncSession) -> bool:
    """Check if token matches the configured key or the current generated key."""
    if config.is_master_key(token):
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
        raise _authentication_unavailable("db_error") from exc
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

    ``config`` says whether an identity token may stand in for a key's secret
    (``federation_issuers``, ``federation_member_rules``). Like its sibling ``verify_api_key_or_master_key``,
    this is called directly (not only via ``Depends``) by downstream
    deployments, so the three-argument shape is API.

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
    return await _verify_credential(request, db, token, config)


async def verify_service_key(api_key: Annotated[APIKey, Depends(verify_api_key)]) -> APIKey:
    """Verify an API key that is a service key, for the routes where it manages its own end users.

    Neither the master key nor a dashboard session passes: these routes are
    scoped by the key they authenticate with, and either of those names no key.
    Any other valid key is refused with 403, since it is authenticated but has
    no end users to manage.
    """
    if not api_key.is_service_key or not api_key.user_id:
        record_auth_failure("not_service_key")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=SERVICE_KEY_REQUIRED_DETAIL)
    return api_key


ServiceKey = Annotated[APIKey, Depends(verify_service_key)]

# The API key a request authenticates with, for a route that answers for that key alone.
CallingKey = Annotated[APIKey, Depends(verify_api_key)]


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
    refuse_ended_session(request, config)
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
    elif config.is_master_key(token):
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
    `/api/v1/hosted-providers` holds process-global provider secrets, and
    `POST /api/v1/settings/master-key/rotate` replaces the deployment credential. On
    a single-operator deployment every login is that operator and the
    distinction is invisible; once mutually-untrusting tenants sign in to one
    process, a member of one organization holding master-key authority is a
    cross-organization breach (otari-ai#1880).

    A **header master key** is the deployment credential itself, so it passes; it
    names nobody, and ``get_current_identity`` resolves it to the bootstrap
    operator. A **session** is put to
    ``DeploymentAdministrationAccess.has_administration_access``, which `/api/v1/admin`
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
    if session_identity is not None and not await DeploymentAdministrationAccess(db).has_administration_access(
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

    api_key = await _verify_credential(request, db, token, config)
    return api_key, False


async def verify_api_key_or_master_key_for_upload(
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> tuple[APIKey | None, bool]:
    """:func:`verify_api_key_or_master_key`, then free its connection before the route reads the body.

    A route that parses its own upload does so after authentication, so a slow
    client would otherwise pin the pooled connection the key lookup checked out.
    """
    await release_session(db)
    return auth_result


async def verify_catalog_reader(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
) -> tuple[APIKey | None, bool]:
    """As :func:`verify_api_key_or_master_key`, and a dashboard session also reads.

    The narrow exception to the rule above, for the catalog reads that describe
    the deployment rather than act on it: ``GET /api/v1/models``, ``GET /api/v1/pricing``,
    ``GET /api/v1/tools``, ``GET /api/v1/providers/catalog`` (with their by-id
    variants), ``GET /api/v1/tool-settings/guardrails/catalog``,
    ``GET /api/v1/tool-settings/guardrails/profiles``,
    ``GET /api/v1/search-tools/providers``, ``GET /api/v1/search-tools/catalog`` and
    ``GET /api/v1/code/environments``. The
    dashboard's Models and Pricing pages are built on these, so a session has to
    reach them; they call no provider, write nothing, and bill nothing, so
    reaching them deployment-wide costs a signed-in caller's own organization
    nothing. The guardrail profiles read is the one that reaches a service, the
    deployment's own guardrails service, and it answers from a 30 second
    per-worker cache, so a worker sends that service at most one request in
    that time.

    The four catalogs are the reads a *tenant* rather than an operator needs. One
    names the providers any-llm knows, which the organization provider-key form
    offers as the BYO choices; one names the guardrails any-guardrail reaches
    over a hosted API, which the organization guardrail form offers the same
    way; one names the guardrail profiles the deployment serves, which any
    caller may put in a request; and one names the providers any-search and
    any-fetch serve, for the organization's own search keys. An owner or admin
    who reaches no operator route still has to read all four.

    Split out rather than left as a branch inside the other dependency so that
    adding a route to this plane defaults to refusing the cookie. The seven
    routers that serve these reads declare it on the router for the same reason,
    so admitting a session is spelled where the route is mounted rather than in
    one route's decorator.
    """
    if session_identity is not None:
        return None, True
    refuse_ended_session(request, config)
    return await verify_api_key_or_master_key(request, db, config)


async def catalog_reader_operates_deployment(
    db: Annotated[AsyncSession, Depends(get_db)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
    auth: Annotated[tuple[APIKey | None, bool], Depends(verify_catalog_reader)],
) -> bool:
    """Whether a caller :func:`verify_catalog_reader` admitted operates the deployment.

    The question :func:`require_deployment_operator` asks, answered rather than
    enforced, for a catalog read that shows its operator what it withholds from
    everyone else. A session is put to the same
    ``DeploymentAdministrationAccess.has_administration_access``. Without one, the caller
    holds the master key, which operates the deployment, or an API key, which
    does not.
    """
    if session_identity is not None:
        return await DeploymentAdministrationAccess(db).has_administration_access(session_identity)
    _, is_master_key = auth
    return is_master_key


async def catalog_reader_organization(
    db: Annotated[AsyncSession, Depends(get_db)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
    auth: Annotated[tuple[APIKey | None, bool], Depends(verify_catalog_reader)],
) -> uuid.UUID | None:
    """The organization whose prices a caller :func:`verify_catalog_reader` admitted is shown.

    Resolved as settlement resolves it: a session's active organization, an API
    key's workspace's, and the default workspace's for the master key.
    """
    if session_identity is not None:
        return session_identity.active_organization_id
    api_key, _ = auth
    return await organization_for_key_id(db, api_key.id if api_key is not None else None)


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
    The route is what narrows an anonymous read (the deployment's own
    offerings, the deployment price list, no tenant rows); this only decides who is asking.

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
