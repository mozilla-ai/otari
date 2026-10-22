import secrets
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import ColumnElement, Select, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.auth.models import hash_key
from gateway.container import Container
from gateway.core.config import API_KEY_HEADER, API_ROOT, X_API_KEY_HEADER, GatewayConfig
from gateway.core.database import DATABASE_ERRORS, create_session, get_db, release_session
from gateway.core.feature import CoreFeature
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.exceptions.identity_exceptions import DeploymentAdministrationUnavailableError
from gateway.log_config import logger
from gateway.metrics import REGISTRY, Counter
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import User as TenancyUser
from gateway.ports.agent_model_recommender_port import AgentModelRecommenderPort
from gateway.ports.api_key_format_port import ApiKeyFormatPort, Malformed, Misdirected
from gateway.ports.billing_port import BillingPort
from gateway.ports.code_execution_port import CodeExecutionPort
from gateway.ports.entitlement_port import EntitlementPort
from gateway.ports.file_storage_port import FileStoragePort
from gateway.ports.growth_signal_port import GrowthSignalPort
from gateway.ports.identity_provider_port import IdentityProviderPort
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.ports.provider_file_port import ProviderFilePort
from gateway.ports.rate_limit_store_port import RateLimitStorePort
from gateway.ports.telemetry_storage_port import TelemetryStoragePort
from gateway.repositories.api_keys import ApiKeyRepositories, ApiKeyRepository, FederationRuleRepository
from gateway.repositories.budgets import BudgetRepositories
from gateway.repositories.files import FileRepositories
from gateway.repositories.health_checks import (
    HealthCheckRecoveryRepository,
    HealthCheckResultRepository,
    HealthCheckRunRepository,
    HealthCheckSettingsRepository,
)
from gateway.repositories.identity import IdentityRepositories
from gateway.repositories.inference import InferenceRepositories
from gateway.repositories.organizations import OrganizationRepositories
from gateway.repositories.overview.overview_repository import OverviewRepository
from gateway.repositories.pricing import ModelPricingRepository, OrganizationModelPricingRepository
from gateway.repositories.providers import (
    HostedProviderModelRepository,
    HostedProviderRepository,
    OrgProviderKeyModelRepository,
    ProviderEndpointRepository,
)
from gateway.repositories.rate_limits import RateLimitRuleRepository
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationGuardrailDefinitionRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
    OrgProviderKeyRepository,
    WorkspaceRepository,
    ids_in_organization,
)
from gateway.repositories.tools import (
    OrgWebSearchKeyRepository,
    WorkspaceCodeExecutionPolicyRepository,
    WorkspaceWebSearchKeyOverrideRepository,
)
from gateway.repositories.traces import TracesRepositories
from gateway.repositories.usage_and_telemetry import UsageLogRepository
from gateway.repositories.users_repository import get_active_user
from gateway.services.api_keys import (
    ApiKeyService,
    FederatedTokenVerifier,
    FederationRefusedError,
    FederationTokenExchange,
    FederationUnavailableError,
    is_access_token,
    key_refusal,
    match_federation_rules,
    read_access_token,
    unverified_issuer,
)
from gateway.services.budgets import BudgetMembershipListener, BudgetService
from gateway.services.code_execution import SandboxContainerRegistry
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, resolve_dashboard_session
from gateway.services.feedback import FeedbackService
from gateway.services.files import FileBackends, FileService, SandboxFileBridge, StagedFile
from gateway.services.health_checks import HealthCheckService
from gateway.services.identity import IdentityService
from gateway.services.inference import InferenceService
from gateway.services.log_writer import LogWriter
from gateway.services.mail import Mailer
from gateway.services.master_key_service import hash_master_key, is_generated_master_key, load_master_key_hash
from gateway.services.model_discovery_service import DiscoveryDial, HostedDiscovery
from gateway.services.oauth_service import provider_label
from gateway.services.organization_pricing_service import OrganizationPricingService
from gateway.services.organizations import InvitationMailLimit, UserAccess
from gateway.services.overview import OverviewService
from gateway.services.playground_service import LocalToolOffer, PlaygroundToolOffer
from gateway.services.pricing import DeploymentPricingService
from gateway.services.providers import (
    HostedProviderService,
    HostedSources,
    OrgProviderModelService,
    ProviderEndpointService,
    refresh_provider_endpoint_cache,
)
from gateway.services.rate_limits import RateLimitService
from gateway.services.routing import RoutingService, clear_router_backend_cache
from gateway.services.secret_box import configured_secret_keys
from gateway.services.tenancy import (
    DomainAutoJoinListener,
    InvitationClaimSettings,
    OrganizationDomainService,
    OrganizationService,
    organization_guardrail_runner,
)
from gateway.services.tenancy.authorization import WorkspaceAccess, resolve_visible_workspace_scope
from gateway.services.tenancy.deployment_user_service import DeploymentUserService
from gateway.services.tenancy.membership_listener import MembershipListener
from gateway.services.tenancy.org_provider_key_service import OrgProviderKeyService, refresh_org_provider_cache
from gateway.services.tenancy.organization_guardrail_definition_service import (
    OrganizationGuardrailDefinitionService,
)
from gateway.services.tenancy.provisioning_service import ensure_bootstrap_identity
from gateway.services.tenancy.workspace_mcp_server_service import WorkspaceMcpServers
from gateway.services.tenancy.workspace_service import WorkspaceService
from gateway.services.tenancy.workspace_web_search_service import WorkspaceWebSearchPolicies, WorkspaceWebSearchService
from gateway.services.tools import (
    WebSearchKeyService,
    WorkspaceCodeExecutionPolicies,
    WorkspaceCodeExecutionPolicyService,
    WorkspaceSearchKeys,
    web_search_configured,
)
from gateway.services.traces import TraceService
from gateway.services.usage_admin_service import usage_in
from gateway.services.usage_and_telemetry import UsageService
from gateway.services.workspace_scope import default_workspace_id
from gateway.types.traces import TraceScope

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


def _federated_token_verifier(request: Request, config: GatewayConfig) -> FederatedTokenVerifier:
    """The app's token verifier for the configured issuers, built on first use and when they change.

    Kept on the app rather than per request, because it caches each issuer's signing keys.
    """
    verifier: FederatedTokenVerifier | None = getattr(request.app.state, "federated_token_verifier", None)
    if (
        verifier is None
        or verifier.issuers != tuple(config.federation_issuers)
        or verifier.allow_private_hosts != config.federation_allow_private_hosts
        or verifier.member_issuers != config.federation_member_rules
    ):
        verifier = FederatedTokenVerifier(
            config.federation_issuers,
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
        identity = await _federated_token_verifier(request, config).verify(token)
        match = match_federation_rules(
            await FederationRuleRepository(db).candidates(identity.issuer, identity.audiences), identity
        )
        api_key = await ApiKeyRepository(db).get(match.api_key_id)
    except FederationUnavailableError as exc:
        record_auth_failure("federation_unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        ) from exc
    except FederationRefusedError as exc:
        record_auth_failure(exc.reason)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key") from exc
    except DATABASE_ERRORS as exc:
        record_auth_failure("db_error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        ) from exc
    if api_key is None:
        record_auth_failure("federation_no_rule")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
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
        record_auth_failure(exc.reason)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key") from exc
    except DATABASE_ERRORS as exc:
        record_auth_failure("db_error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        ) from exc
    if api_key is None:
        record_auth_failure("federation_rule_gone")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return await _admit_api_key(api_key)


async def _verify_credential(request: Request, db: AsyncSession, token: str, config: GatewayConfig) -> APIKey:
    """Verify a presented credential as an API key.

    With federation on, the prefix of an access token or the shape of a JWT routes it before anything is parsed:
    an access token, then a raw identity token, which only an operator's rule on an allow-listed issuer can match.
    Everything else is a key secret.
    """
    if config.federation_enabled:
        if is_access_token(token):
            return await _verify_access_token(db, token, config)
        if config.federation_issuers and unverified_issuer(token) is not None:
            return await _verify_federated_token(request, db, token, config)
    return await _verify_and_update_api_key(db, token, _api_key_format(request, db))


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
        # codeql[py/clear-text-logging-sensitive-data]
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
        record_auth_failure("db_error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication temporarily unavailable, please retry",
        ) from exc


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
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint requires a service key.",
        )
    return api_key


ServiceKey = Annotated[APIKey, Depends(verify_service_key)]


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
    `/api/v1/hosted-providers` holds process-global provider secrets, and
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
    if session_identity is not None and not await DeploymentUserService(db).has_administration_access(session_identity):
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
    variants), ``GET /api/v1/tool-settings/guardrails/catalog`` and
    ``GET /api/v1/search-tools/providers``. The
    dashboard's Models and Pricing pages are built on these, so a session has to
    reach them; they call no provider, write nothing, and bill nothing, so
    reaching them deployment-wide costs a signed-in caller's own organization
    nothing.

    The three catalogs are the reads a *tenant* rather than an operator needs. One
    names the providers any-llm knows, which the organization provider-key form
    offers as the BYO choices; one names the guardrails any-guardrail reaches
    over a hosted API, which the organization guardrail form offers the same
    way; and one names the providers any-search and any-fetch serve, for the
    organization's own search keys. An owner or admin who reaches no operator
    route still has to read all three.

    Split out rather than left as a branch inside the other dependency so that
    adding a route to this plane defaults to refusing the cookie. The six
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
    ``DeploymentUserService.has_administration_access``. Without one, the caller
    holds the master key, which operates the deployment, or an API key, which
    does not.
    """
    if session_identity is not None:
        return await DeploymentUserService(db).has_administration_access(session_identity)
    _, is_master_key = auth
    return is_master_key


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


def build_file_service(uow: UnitOfWork, backends: FileBackends, config: GatewayConfig) -> FileService:
    """Build Files operations for a scoped output request or cleanup job."""

    async def reject_unscoped_upload() -> uuid.UUID:
        raise RuntimeError("Unscoped uploads are not supported in this context; specify a workspace.")

    return FileService(uow, FileRepositories.on(uow), backends, config, reject_unscoped_upload)


def build_trace_service(uow: UnitOfWork) -> TraceService:
    """Build the traces operations for a request or a worker job."""
    return TraceService(uow, TracesRepositories.on(uow))


def _file_backends(request: Request) -> FileBackends | None:
    """The backends this build bound for files, or ``None`` where it has none.

    A deployment with files off starts without a store it could not build.
    """
    storage: FileStoragePort | None = getattr(request.app.state, "file_store", None)
    provider_files: ProviderFilePort | None = getattr(request.app.state, "provider_files", None)
    if storage is None or provider_files is None:
        return None
    return FileBackends(storage=storage, provider_files=provider_files)


def build_inference_service(uow: UnitOfWork, config: GatewayConfig) -> InferenceService:
    """Build the inference service for a request or a worker job, such as the idempotency sweep.

    It receives the Unit of Work and its own domain's repositories over it, never the session.
    """
    return InferenceService(uow, InferenceRepositories.on(uow), config)


def build_sandbox_file_bridge(
    *,
    raw_request: Request,
    config: GatewayConfig,
    uow: UnitOfWork,
    user_id: str | None,
    workspace_id: uuid.UUID | None,
    inputs: list[StagedFile],
) -> SandboxFileBridge | None:
    """The file bridge a completion request's sandbox session gets, or ``None``.

    Built by the route once the billed user and workspace are resolved, on the request's own Unit of Work.
    It is ``None`` when files are disabled.
    Produced files are announced under ``public_base_url`` where the deployment knows its address,
    and otherwise under the one the request arrived on.
    """
    backends = _file_backends(raw_request)
    if not config.files_enabled or backends is None or user_id is None or workspace_id is None:
        return None
    base = (config.public_base_url or str(raw_request.base_url)).rstrip("/")
    return SandboxFileBridge(
        backends=backends,
        config=config,
        files=build_file_service(uow, backends, config),
        user_id=user_id,
        workspace_id=workspace_id,
        inputs=inputs,
        base_url=f"{base}{API_ROOT}/files",
    )


def build_sandbox_container_registry(
    *,
    config: GatewayConfig,
    uow: UnitOfWork,
    user_id: str | None,
    workspace_id: uuid.UUID | None,
    port: CodeExecutionPort | None,
) -> SandboxContainerRegistry | None:
    """The registry a completion request's sandbox session is held in, or ``None``.

    ``None`` is every case in which a sandbox cannot outlive its request: a
    deployment with no sandbox at all; and one that turned reuse off with a zero idle TTL. The
    lease is scoped to the billed user and workspace, and to the adapter this
    build runs, so a deployment that changes providers starts fresh.
    """
    if user_id is None or workspace_id is None or port is None or config.sandbox_container_idle_ttl_sec <= 0:
        return None
    return SandboxContainerRegistry(
        uow=uow,
        user_id=user_id,
        workspace_id=workspace_id,
        provider=port.label,
        idle_ttl_s=config.sandbox_container_idle_ttl_sec,
        max_lifetime_s=config.sandbox_container_max_lifetime_sec,
    )


def get_unit_of_work(db: Annotated[AsyncSession, Depends(get_db)]) -> UnitOfWork:
    """Return the request's Unit of Work over its session.

    Gotcha: the rest of the request writes to this same session.
    A block's commit also stores what that code staged outside a block, and its rollback discards it.
    """
    return UnitOfWork(db)


UnitOfWorkDep = Annotated[UnitOfWork, Depends(get_unit_of_work)]


def get_membership_listener(uow: UnitOfWorkDep) -> MembershipListener:
    """Return the listener that keeps members' budget ceilings in step with their memberships.

    It writes through the request's Unit of Work, so a service that changes membership must be built on the same one.
    """
    return BudgetMembershipListener(BudgetRepositories.on(uow))


MembershipListenerDep = Annotated[MembershipListener, Depends(get_membership_listener)]


def get_workspace_code_execution_policy_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceCodeExecutionPolicyService:
    """Build the workspace code execution policy service on the request's Unit of Work.

    ``sandbox_configured`` is the check ``GET /api/v1/tools`` makes before it advertises code
    execution, so the page and discovery agree. ``default_enabled`` is what the request path gives a
    workspace with no policy. ``allowed_images`` is the operator's curated list. All are properties
    of the running gateway, handed in rather than read by the service.
    """
    return WorkspaceCodeExecutionPolicyService(
        uow,
        WorkspaceCodeExecutionPolicyRepository(uow),
        WorkspaceAccess(db, OrganizationService(db, membership_listener=None)),
        sandbox_configured=config.sandbox_configured(),
        default_enabled=config.code_execution_workspace_default,
        allowed_images=config.pinnable_sandbox_images(),
    )


WorkspaceCodeExecutionPolicyServiceDep = Annotated[
    WorkspaceCodeExecutionPolicyService, Depends(get_workspace_code_execution_policy_service)
]


def get_workspace_web_search_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceWebSearchService:
    """Build the service on the request's session.

    The backend-presence check, with the workspace's organization key, is the
    one admission makes when it decides whether a request may declare
    ``otari_web_search``, and the default is the one the request path applies
    to a workspace with no row, so the page and the request path agree.
    """
    return WorkspaceWebSearchService(
        db,
        web_search_configured=web_search_configured(config),
        default_enabled=config.web_search_workspace_default,
        has_search_key=get_workspace_search_keys(db).has_search_key,
    )


WorkspaceWebSearchServiceDep = Annotated[WorkspaceWebSearchService, Depends(get_workspace_web_search_service)]


def get_workspace_code_execution_policies(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceCodeExecutionPolicies:
    """Build the request path's reader of a workspace's code execution policy, on the request's session."""
    return WorkspaceCodeExecutionPolicies(
        WorkspaceCodeExecutionPolicyRepository(db), on_without_policy=config.code_execution_workspace_default
    )


WorkspaceCodeExecutionPoliciesDep = Annotated[
    WorkspaceCodeExecutionPolicies, Depends(get_workspace_code_execution_policies)
]


def get_playground_tool_offer(config: Annotated[GatewayConfig, Depends(get_config)]) -> PlaygroundToolOffer:
    """Return what this deployment offers a Playground message."""
    return LocalToolOffer(config)


PlaygroundToolOfferDep = Annotated[PlaygroundToolOffer, Depends(get_playground_tool_offer)]


async def get_current_identity(
    db: Annotated[AsyncSession, Depends(get_db)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
    _master_key: Annotated[str | None, Depends(verify_master_key)],
    uow: UnitOfWorkDep,
    membership_listener: MembershipListenerDep,
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
    same operator; the difference matters once the per-user sign-in flows mint
    sessions for other identities.

    Depending on ``verify_master_key`` keeps the credential check exactly where
    the rest of the management API has it, and keeps a request with no credential
    at all from provisioning anything.
    """
    if session_identity is not None:
        return session_identity
    return await ensure_bootstrap_identity(
        db,
        uow=uow,
        membership_listener=membership_listener,
    )


CurrentIdentity = Annotated[TenancyUser, Depends(get_current_identity)]


async def require_deployment_operator_or_absent(
    db: Annotated[AsyncSession, Depends(get_db)],
    identity: Annotated[TenancyUser, Depends(get_current_identity)],
) -> None:
    """Refuse a non-operator with the 404 ``/api/v1/admin`` answers, so the surface does not confirm it exists.

    The same predicate as :func:`require_deployment_operator`, with the other
    status: a deployment-wide surface that holds the deployment's own
    credentials is one worth not confirming to a signed-in member, who gets
    the answer an unentitled deployment gets. A header master key is the
    deployment credential itself, resolved to the bootstrap operator, so it
    passes. Declared on a router rather than per route, for the reason the 403
    gate gives.
    """
    if not await DeploymentUserService(db).has_administration_access(identity):
        record_auth_failure("not_deployment_operator")
        raise DeploymentAdministrationUnavailableError


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
# The request's own session: FastAPI caches a dependency per callable, so a port
# resolved on this shares the session of the route that resolved it, and its
# writes land in the commit that settles the request.
PortSessionDep = Annotated[AsyncSession, Depends(get_db)]


def get_agent_model_recommender_port(container: ContainerDep) -> AgentModelRecommenderPort:
    """Resolve the subagent model recommender this build bound at startup.

    No session: the adapter asks a model, not this database, and the call is
    metered on the route's own session by the decisions scaffold.
    """
    return container.resolve(AgentModelRecommenderPort, None)


AgentModelRecommenderPortDep = Annotated[AgentModelRecommenderPort, Depends(get_agent_model_recommender_port)]


def get_api_key_format_port(db: PortSessionDep, container: ContainerDep) -> ApiKeyFormatPort:
    """Resolve the key-format adapter this build bound at startup."""
    return container.resolve(ApiKeyFormatPort, db)


def get_code_execution_port(
    config: Annotated[GatewayConfig, Depends(get_config)],
    container: ContainerDep,
) -> CodeExecutionPort | None:
    """Resolve the code-execution adapter this build bound at startup.

    ``None`` where the deployment has configured no sandbox, which is most of
    them: every completion request resolves this, and one that never asks for
    code execution must not be refused because there is nowhere to run it. The
    request that does ask is refused by name in ``prepare_gateway_tools``.

    No session either: the adapter reaches a sandbox, not this database, and
    which one it reaches is a deployment setting rather than a request fact.
    """
    if not config.sandbox_configured():
        return None
    return container.resolve(CodeExecutionPort, None)


CodeExecutionPortDep = Annotated[CodeExecutionPort | None, Depends(get_code_execution_port)]


def get_billing_port(db: PortSessionDep, container: ContainerDep) -> BillingPort:
    """Resolve the billing adapter this build bound at startup."""
    return container.resolve(BillingPort, db)


def get_request_session_billing_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    container: ContainerDep,
) -> BillingPort:
    """Resolve the billing adapter on the request's ``get_db`` session."""
    return container.resolve(BillingPort, db)


def get_entitlement_port(db: PortSessionDep, container: ContainerDep) -> EntitlementPort:
    """Resolve the entitlement adapter this build bound at startup."""
    return container.resolve(EntitlementPort, db)


def get_growth_signal_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    container: ContainerDep,
) -> GrowthSignalPort:
    """Resolve the growth-signal adapter this build bound at startup."""
    return container.resolve(GrowthSignalPort, db)


# The adapter writes and does not commit, so it shares the session of the Unit of Work whose block commits its writes.
def get_identity_provider_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    container: ContainerDep,
) -> IdentityProviderPort:
    """Resolve the identity adapter this build bound at startup, on the request's session and Unit of Work."""
    return container.resolve(IdentityProviderPort, db, uow=uow)


def get_workspace_mcp_servers(db: Annotated[AsyncSession, Depends(get_db)]) -> WorkspaceMcpServers:
    """Build the request path's reader of a workspace's MCP servers, on the request's session."""
    return WorkspaceMcpServers(db)


def get_model_provider_port(
    db: PortSessionDep,
    uow: UnitOfWorkDep,
    container: ContainerDep,
) -> ModelProviderPort:
    """Resolve the model-provider adapter this build bound at startup.

    On the request's session and Unit of Work: the core adapter reads the hosted-providers store through a block on it.
    """
    return container.resolve(ModelProviderPort, db, uow=uow)


def get_request_session_model_provider_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    container: ContainerDep,
) -> ModelProviderPort:
    """Resolve the model-provider adapter on the request's ``get_db`` session and its Unit of Work."""
    return container.resolve(ModelProviderPort, db, uow=uow)


def model_provider_port_on(request: Request, db: AsyncSession) -> ModelProviderPort:
    """Resolve the model-provider adapter on ``db``, for a path that resolves it only when it needs it."""
    return get_container(request).resolve(ModelProviderPort, db, uow=get_unit_of_work(db))


def billing_port_on(request: Request, db: AsyncSession) -> BillingPort:
    """Resolve the billing adapter on ``db``, whose commit lands the charge it writes."""
    return get_container(request).resolve(BillingPort, db)


def get_telemetry_storage_port(
    db: Annotated[AsyncSession, Depends(get_db)],
    container: ContainerDep,
) -> TelemetryStoragePort:
    """Resolve the telemetry-storage adapter this build bound at startup."""
    return container.resolve(TelemetryStoragePort, db)


def get_web_provider_client(request: Request) -> httpx.AsyncClient:
    """The client the web tools' providers are called through, which the app's lifespan builds and closes."""
    client: httpx.AsyncClient | None = getattr(request.app.state, "web_provider_client", None)
    if client is None:
        msg = "No web provider client: the app's lifespan has not started, or has ended"
        raise RuntimeError(msg)
    return client


def get_optional_web_provider_client(request: Request) -> httpx.AsyncClient | None:
    """The app's web provider client, or ``None`` outside the app's lifespan, where a provider opens its own.

    For ``ToolPorts``, which every completion route takes, so a request that runs
    no web tool never depends on the client.
    """
    client: httpx.AsyncClient | None = getattr(request.app.state, "web_provider_client", None)
    return client


def get_overview_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    membership_listener: MembershipListenerDep,
) -> OverviewService:
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
        WorkspaceService(db, uow=uow, membership_listener=membership_listener),
    )


OverviewServiceDep = Annotated[OverviewService, Depends(get_overview_service)]


def get_organization_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationService:
    """Build the request's organization service.

    It reads only. A membership write needs the listener this pairing leaves unset.
    """
    return OrganizationService(db, membership_listener=None)


OrganizationServiceDep = Annotated[OrganizationService, Depends(get_organization_service)]


def get_deployment_user_service(db: Annotated[AsyncSession, Depends(get_db)]) -> DeploymentUserService:
    """Build the request's deployment user service, which answers whether a caller operates the deployment."""
    return DeploymentUserService(db)


DeploymentUserServiceDep = Annotated[DeploymentUserService, Depends(get_deployment_user_service)]


def get_membership_organization_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    membership_listener: MembershipListenerDep,
) -> OrganizationService:
    """Build the request's organization service for writing memberships and creating workspaces."""
    return OrganizationService(db, membership_listener=membership_listener, uow=uow)


MembershipOrganizationServiceDep = Annotated[OrganizationService, Depends(get_membership_organization_service)]


def get_organization_domain_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationDomainService:
    """Build the email-domain service on the request's session."""
    return OrganizationDomainService(db)


OrganizationDomainServiceDep = Annotated[OrganizationDomainService, Depends(get_organization_domain_service)]


def get_invitation_claim_settings(config: Annotated[GatewayConfig, Depends(get_config)]) -> InvitationClaimSettings:
    """Return what this deployment asks of, and offers, an invitee who chooses a first password from an invitation link.

    The terms it publishes, and the ways in a refusal can offer instead: its sign-in providers by display name,
    and a reset link where mail can send one.
    """
    return InvitationClaimSettings(
        terms_url=config.terms_url,
        sign_in_providers=tuple(provider_label(name) for name in config.oauth_providers),
        can_send_links=Mailer(config).can_send_links,
    )


InvitationClaimSettingsDep = Annotated[InvitationClaimSettings, Depends(get_invitation_claim_settings)]


def get_invitation_mail_limit(
    config: Annotated[GatewayConfig, Depends(get_config)], container: ContainerDep
) -> InvitationMailLimit:
    """Return the invitation mail limits, counted in the rate-limit store this build bound."""
    return InvitationMailLimit(
        container.resolve(RateLimitStorePort, None),
        per_organization=config.invitation_mail_limit_per_organization,
        per_address=config.invitation_mail_limit_per_address,
    )


InvitationMailLimitDep = Annotated[InvitationMailLimit, Depends(get_invitation_mail_limit)]


def get_user_access(uow: Annotated[UnitOfWork, Depends(get_unit_of_work)]) -> UserAccess:
    """Build the organizations domain's read of users' access lists on the request's Unit of Work."""
    return UserAccess(OrganizationRepositories.on(uow))


def get_api_key_service(uow: Annotated[UnitOfWork, Depends(get_unit_of_work)]) -> ApiKeyService:
    """Build the request's API key service on the request's Unit of Work; key owners' lists come from organizations."""
    return ApiKeyService(uow, ApiKeyRepositories.on(uow), users=get_user_access(uow))


ApiKeyServiceDep = Annotated[ApiKeyService, Depends(get_api_key_service)]


def build_federation_token_exchange(
    uow: UnitOfWork, config: GatewayConfig, verifier: FederatedTokenVerifier
) -> FederationTokenExchange:
    """Build the token exchange on a Unit of Work; the key's organization is read through the same one."""
    workspaces = WorkspaceRepository(uow)

    async def organization_of_workspace(workspace_id: uuid.UUID) -> uuid.UUID | None:
        workspace = await workspaces.get(workspace_id)
        return None if workspace is None else workspace.organization_id

    return FederationTokenExchange(
        uow,
        ApiKeyRepositories.on(uow),
        verifier=verifier,
        organization_of_workspace=organization_of_workspace,
        ttl_seconds=config.federation_access_token_ttl_sec,
        secrets=configured_secret_keys(),
    )


def get_federation_token_exchange(
    request: Request,
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> FederationTokenExchange:
    """The request's token exchange."""
    return build_federation_token_exchange(uow, config, _federated_token_verifier(request, config))


FederationTokenExchangeDep = Annotated[FederationTokenExchange, Depends(get_federation_token_exchange)]


def get_budget_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetService:
    """Build the request's budget service on the request's Unit of Work."""
    organizations = OrganizationService(db, membership_listener=None)
    return BudgetService(
        uow,
        BudgetRepositories.on(uow),
        organizations,
        get_api_key_service(uow),
        WorkspaceAccess(db, organizations),
    )


BudgetServiceDep = Annotated[BudgetService, Depends(get_budget_service)]


def build_budget_service(session: AsyncSession) -> BudgetService:
    """Build the budget service for startup work, on a Unit of Work over a session of its own."""
    return get_budget_service(get_unit_of_work(session), session)


def get_organization_guardrail_definition_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrganizationGuardrailDefinitionService:
    """Build an organization's guardrail-definition service on the request's Unit of Work.

    Assembled here for the reason the overview service above is: a route does not
    name a session type (``scripts/check_architecture.py``, rule 12).

    The session is here only for the role gate. The service itself holds none,
    reaching the database through its repository and committing in the blocks it
    opens; `OrganizationService` is still in the old shape and takes a session,
    and both are the request's one session, so a block commits what the gate
    read.

    The runner is wired in here rather than imported by the service, because the
    runner imports the service for the function that undoes the secret split and
    the pair would otherwise form a cycle. This is the composition root, which
    is where that join belongs anyway.
    """
    return OrganizationGuardrailDefinitionService(
        definitions=OrganizationGuardrailDefinitionRepository(uow),
        organizations=OrganizationService(db, membership_listener=None),
        uow=uow,
        build_state=organization_guardrail_runner.build_state,
        rebuild=organization_guardrail_runner.rebuild_definition,
        handle=organization_guardrail_runner.handle,
    )


OrganizationGuardrailDefinitionServiceDep = Annotated[
    OrganizationGuardrailDefinitionService, Depends(get_organization_guardrail_definition_service)
]

ApiKeyFormatPortDep = Annotated[ApiKeyFormatPort, Depends(get_api_key_format_port)]
BillingPortDep = Annotated[BillingPort, Depends(get_billing_port)]
RequestSessionBillingPortDep = Annotated[BillingPort, Depends(get_request_session_billing_port)]
EntitlementPortDep = Annotated[EntitlementPort, Depends(get_entitlement_port)]
GrowthSignalPortDep = Annotated[GrowthSignalPort, Depends(get_growth_signal_port)]
IdentityProviderPortDep = Annotated[IdentityProviderPort, Depends(get_identity_provider_port)]
WorkspaceMcpServersDep = Annotated[WorkspaceMcpServers, Depends(get_workspace_mcp_servers)]
ModelProviderPortDep = Annotated[ModelProviderPort, Depends(get_model_provider_port)]
RequestSessionModelProviderPortDep = Annotated[ModelProviderPort, Depends(get_request_session_model_provider_port)]


def get_identity_service(
    uow: UnitOfWorkDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    identity_provider: IdentityProviderPortDep,
    organizations: OrganizationServiceDep,
) -> IdentityService:
    """Build the identity service on the request's Unit of Work."""
    return IdentityService(
        uow,
        IdentityRepositories.on(uow),
        identity_provider=identity_provider,
        organizations=organizations,
        sign_in_listener=DomainAutoJoinListener(OrganizationDomainRepository(uow), OrganizationMemberRepository(uow)),
        session_ttl_hours=config.dashboard_session_ttl_hours,
        terms_url=config.terms_url,
        mailer=Mailer(config),
        open_signup=config.open_signup,
        password_reset_expiry_hours=config.password_reset_expiry_hours,
    )


IdentityServiceDep = Annotated[IdentityService, Depends(get_identity_service)]


def get_org_provider_model_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    model_provider: RequestSessionModelProviderPortDep,
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


def get_provider_endpoint_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ProviderEndpointService:
    """Build the owned provider endpoint service on the request's session and unit of work."""

    async def resolve_workspace(workspace_id: uuid.UUID | None) -> uuid.UUID | None:
        if workspace_id is None:
            return await default_workspace_id(db)
        return workspace_id if await WorkspaceRepository(db).get(workspace_id) is not None else None

    async def user_is_active(user_id: str) -> bool:
        return await get_active_user(db, user_id) is not None

    return ProviderEndpointService(
        uow,
        config=config,
        endpoints=ProviderEndpointRepository(uow),
        resolve_workspace=resolve_workspace,
        user_is_active=user_is_active,
        refresh_cache=lambda: refresh_provider_endpoint_cache(uow),
    )


ProviderEndpointServiceDep = Annotated[ProviderEndpointService, Depends(get_provider_endpoint_service)]


def get_web_search_key_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WebSearchKeyService:
    """Build the organization web search key service on the request's Unit of Work.

    The config tells a workspace without a key of a kind what its calls run on.
    """
    organizations = OrganizationService(db, membership_listener=None)
    return WebSearchKeyService(
        uow,
        keys=OrgWebSearchKeyRepository(uow),
        overrides=WorkspaceWebSearchKeyOverrideRepository(uow),
        organizations=organizations,
        workspaces=WorkspaceAccess(db, organizations),
        lock_workspace=WorkspaceRepository(db).lock,
        config=config,
    )


WebSearchKeyServiceDep = Annotated[WebSearchKeyService, Depends(get_web_search_key_service)]


def get_workspace_search_keys(db: Annotated[AsyncSession, Depends(get_db)]) -> WorkspaceSearchKeys:
    """Build the resolver of the keys a workspace uses, on the request's session."""
    return WorkspaceSearchKeys(
        workspaces=WorkspaceRepository(db), overrides=WorkspaceWebSearchKeyOverrideRepository(db)
    )


def get_workspace_web_search_policies(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceWebSearchPolicies:
    """Build the request path's reader of a workspace's web search policy, on the request's session."""
    return WorkspaceWebSearchPolicies(
        db, search_keys=get_workspace_search_keys(db), on_without_policy=config.web_search_workspace_default
    )


def build_hosted_provider_service(uow: UnitOfWork, *, config: GatewayConfig) -> HostedProviderService:
    """Build the hosted-providers service on ``uow``.

    The one assembly, used by the request dependency below and handed to the
    composition root for the model provider adapter, the way the membership
    listener's builder is. The deployment price list is reached through the
    pricing domain's own service on the same unit of work, so a rate written
    while offering a model lands in the same transaction as the offer.
    """
    providers = HostedProviderRepository(uow)
    return HostedProviderService(
        uow,
        config=config,
        providers=providers,
        sources=HostedSources(config, providers),
        models=HostedProviderModelRepository(uow),
        endpoints=ProviderEndpointRepository(uow),
        pricing=DeploymentPricingService(
            uow, pricing=ModelPricingRepository(uow), overrides=OrganizationModelPricingRepository(uow)
        ),
        live_byo_pairs=OrgProviderKeyRepository(uow).live_provider_pairs,
        organization_exists=OrganizationRepository(uow).exists,
    )


def get_hosted_provider_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> HostedProviderService:
    """Build the hosted-providers service on the request's unit of work."""
    return build_hosted_provider_service(uow, config=config)


HostedProviderServiceDep = Annotated[HostedProviderService, Depends(get_hosted_provider_service)]


def build_hosted_discovery(config: GatewayConfig) -> HostedDiscovery:
    """Build the discovery worker's read of the hosted providers' dials.

    Each call reads in a unit of work of its own, which closes before any dial
    runs, so no session is held across an upstream call.
    """

    async def dials() -> dict[str, DiscoveryDial]:
        async with create_unit_of_work() as uow:
            return await build_hosted_provider_service(uow, config=config).discovery_dials()

    return dials


def get_rate_limit_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> RateLimitService:
    """Build the rate-limit rules service on the request's unit of work."""
    return RateLimitService(uow, RateLimitRuleRepository(uow), config)


RateLimitServiceDep = Annotated[RateLimitService, Depends(get_rate_limit_service)]


def build_health_check_service(uow: UnitOfWork, config: GatewayConfig) -> HealthCheckService:
    """Build the health checks service on a request's Unit of Work, or the worker's own.

    The service is handed the Unit of Work, never the session under it, with its own domain's repositories and
    the providers domain's service for the hosted roster, as DOMAINS.md's constructor injection asks.
    """
    return HealthCheckService(
        uow,
        HealthCheckSettingsRepository(uow),
        HealthCheckRunRepository(uow),
        HealthCheckResultRepository(uow),
        HealthCheckRecoveryRepository(uow),
        build_hosted_provider_service(uow, config=config),
        config,
    )


def get_health_check_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> HealthCheckService:
    """The request's service. The poll builds its own through ``build_health_check_service``."""
    return build_health_check_service(uow, config)


HealthCheckServiceDep = Annotated[HealthCheckService, Depends(get_health_check_service)]
TelemetryStoragePortDep = Annotated[TelemetryStoragePort, Depends(get_telemetry_storage_port)]


def get_trace_service(uow: UnitOfWorkDep) -> TraceService:
    """Build the request's traces service on the request's Unit of Work."""
    return build_trace_service(uow)


TraceServiceDep = Annotated[TraceService, Depends(get_trace_service)]


@dataclass(frozen=True)
class VisibleWorkspaces:
    """The caller's active organization, and the workspaces in it they may read."""

    organization_id: uuid.UUID
    workspace_ids: frozenset[uuid.UUID]


async def get_visible_workspaces(
    identity: CurrentIdentity, db: Annotated[AsyncSession, Depends(get_db)]
) -> VisibleWorkspaces:
    """The workspaces of their active organization this caller may read.

    The same rule the organization's usage reads use, resolved per request: an
    owner or an admin reads every workspace in the organization, anyone else the
    ones they actively belong to. Never taken from the request.
    """
    organizations = OrganizationService(db, membership_listener=None)
    scope = await resolve_visible_workspace_scope(db, user=identity, organizations=organizations)
    workspace_ids = (
        await WorkspaceRepository(db).get_ids_by_organization(scope.organization.id)
        if scope.workspace_ids is None
        else scope.workspace_ids
    )
    return VisibleWorkspaces(organization_id=scope.organization.id, workspace_ids=frozenset(workspace_ids))


VisibleWorkspacesDep = Annotated[VisibleWorkspaces, Depends(get_visible_workspaces)]


async def get_organization_trace_scope(visible: VisibleWorkspacesDep) -> TraceScope:
    """The workspaces of their active organization this caller may read traces in."""
    return TraceScope.workspaces(visible.workspace_ids)


def get_routing_service(uow: UnitOfWorkDep, config: Annotated[GatewayConfig, Depends(get_config)]) -> RoutingService:
    """Build the routing service, and the usage service it reads traffic through, on the request's Unit of Work."""
    return RoutingService(config, UsageService(uow, UsageLogRepository(uow)))


RoutingServiceDep = Annotated[RoutingService, Depends(get_routing_service)]


WorkspaceWebSearchPoliciesDep = Annotated[WorkspaceWebSearchPolicies, Depends(get_workspace_web_search_policies)]
WebProviderClientDep = Annotated[httpx.AsyncClient, Depends(get_web_provider_client)]
OptionalWebProviderClientDep = Annotated[httpx.AsyncClient | None, Depends(get_optional_web_provider_client)]


@dataclass(frozen=True, kw_only=True)
class ToolPorts:
    """The ports and policies a request's tools reach, and the client their web providers are called through.

    ``web_provider_client`` is the app's; ``None`` outside its lifespan, or where
    a test builds this by hand, and a provider then opens its own for the call.
    """

    code_execution: CodeExecutionPort | None
    code_execution_policy: WorkspaceCodeExecutionPolicies
    mcp_server: WorkspaceMcpServers
    web_search_policy: WorkspaceWebSearchPolicies
    web_provider_client: httpx.AsyncClient | None = None


def get_tool_ports(
    code_execution: CodeExecutionPortDep,
    code_execution_policy: WorkspaceCodeExecutionPoliciesDep,
    mcp_server: WorkspaceMcpServersDep,
    web_search_policy: WorkspaceWebSearchPoliciesDep,
    web_provider_client: OptionalWebProviderClientDep,
) -> ToolPorts:
    """Resolve the ports and policies a request's tools reach."""
    return ToolPorts(
        code_execution=code_execution,
        code_execution_policy=code_execution_policy,
        mcp_server=mcp_server,
        web_search_policy=web_search_policy,
        web_provider_client=web_provider_client,
    )


ToolPortsDep = Annotated[ToolPorts, Depends(get_tool_ports)]


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


def get_file_store(request: Request) -> FileStoragePort:
    """Return the configured blob store for uploaded files."""
    store: FileStoragePort = request.app.state.file_store
    return store


def get_file_service(
    request: Request,
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> FileService:
    """Build the request's files service on the request's Unit of Work."""
    backends = _file_backends(request)
    if backends is None:
        raise RuntimeError("Files are served only where this build binds a file store")
    return FileService(uow, FileRepositories.on(uow), backends, config, lambda: default_workspace_id(db))


FileServiceDep = Annotated[FileService, Depends(get_file_service)]


def get_optional_file_service(
    request: Request,
    config: Annotated[GatewayConfig, Depends(get_config)],
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
) -> FileService | None:
    """Return the request's files service, or ``None`` where files are unavailable.

    The counterpart of ``get_file_service``, for a completion route.
    """
    backends = _file_backends(request)
    if backends is None:
        return None
    return FileService(uow, FileRepositories.on(uow), backends, config, lambda: default_workspace_id(db))


OptionalFileServiceDep = Annotated[FileService | None, Depends(get_optional_file_service)]


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


async def _caller_workspace_ids(
    db: Annotated[AsyncSession, Depends(get_db)],
    organization_id: CallerOrganization,
) -> frozenset[uuid.UUID]:
    """Every workspace in the organization this request acts in.

    What a deployment-wide read of workspace-owned rows is confined to, so an
    operator sees their active organization's rows rather than every tenant's.
    """
    return frozenset(await WorkspaceRepository(db).get_ids_by_organization(organization_id))


CallerWorkspaces = Annotated[frozenset[uuid.UUID], Depends(_caller_workspace_ids)]


def _caller_workspace_query(organization_id: CallerOrganization) -> Select[tuple[uuid.UUID]]:
    """The same workspaces as a subquery, so a read of a growing table filters in the database."""
    return ids_in_organization(organization_id)


CallerWorkspaceQuery = Annotated[Select[tuple[uuid.UUID]], Depends(_caller_workspace_query)]


def _caller_usage_scope(workspaces: CallerWorkspaceQuery) -> ColumnElement[bool]:
    """The usage rows an operator's usage reads cover: those of the caller's organization."""
    return usage_in(workspaces)


CallerUsageScope = Annotated[ColumnElement[bool], Depends(_caller_usage_scope)]


def get_deployment_trace_scope(workspace_ids: CallerWorkspaces) -> TraceScope:
    """The scope of the deployment-operator trace routes: every workspace in the caller's organization."""
    return TraceScope.workspaces(workspace_ids)


async def get_feedback_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    _authenticated: Annotated[str | None, Depends(verify_master_key)],
) -> FeedbackService:
    """Finish authentication's database work before waiting for the receiver."""
    await release_session(db)
    return FeedbackService()


__all__ = [
    "AgentModelRecommenderPortDep",
    "BillingPortDep",
    "RequestSessionBillingPortDep",
    "ContainerDep",
    "CallerOrganization",
    "CallerUsageScope",
    "CallerWorkspaceQuery",
    "CallerWorkspaces",
    "CurrentIdentity",
    "EntitlementPortDep",
    "FileServiceDep",
    "OptionalFileServiceDep",
    "OverviewServiceDep",
    "RoutingServiceDep",
    "VisibleWorkspaces",
    "VisibleWorkspacesDep",
    "GrowthSignalPortDep",
    "IdentityProviderPortDep",
    "IdentityServiceDep",
    "ModelProviderPortDep",
    "HostedProviderServiceDep",
    "OrgProviderModelServiceDep",
    "RateLimitServiceDep",
    "HealthCheckServiceDep",
    "build_health_check_service",
    "RequestSessionModelProviderPortDep",
    "TelemetryStoragePortDep",
    "ToolPorts",
    "ToolPortsDep",
    "WebProviderClientDep",
    "WorkspaceMcpServersDep",
    "WorkspaceWebSearchPoliciesDep",
    "get_agent_model_recommender_port",
    "get_config",
    "get_container",
    "get_telemetry_storage_port",
    "get_current_identity",
    "get_db",
    "get_session_identity",
    "reset_config",
    "set_config",
    "get_feedback_service",
    "get_file_store",
    "get_log_writer",
    "is_valid_master_key",
    "model_provider_port_on",
    "require_capability",
    "extract_credential_token",
    "verify_api_key",
    "verify_api_key_or_master_key",
    "verify_catalog_reader",
    "verify_master_key",
]
