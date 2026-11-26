"""Which tenancy identity a request acts as, how it signs in, and whether it may act deployment-wide."""

from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, Request
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import (
    UnitOfWorkDep,
    get_config,
)
from gateway.api.deps._deployment_access import deployment_administration_access
from gateway.api.deps.api_keys import (
    ApiKeyFormatPortDep,
    get_api_key_service,
)
from gateway.api.deps.auth import (
    get_session_identity,
    record_auth_failure,
    verify_master_key,
)
from gateway.api.deps.budgets import BudgetServiceDep
from gateway.api.deps.organizations import (
    MembershipListenerDep,
    OrganizationServiceDep,
    WorkspaceServiceDep,
)
from gateway.api.deps.ports import (
    IdentityProviderPortDep,
)
from gateway.core.config import (
    OAUTH_PROVIDERS,
    GatewayConfig,
)
from gateway.core.database import get_db
from gateway.exceptions.identity_exceptions import DeploymentAdministrationUnavailableError
from gateway.models.tenancy import User as TenancyUser
from gateway.repositories.api_keys import ApiKeyRepositories
from gateway.repositories.identity import (
    DashboardSessionRepository,
    IdentityRepositories,
    OAuthPendingStateRepository,
)
from gateway.repositories.identity.oauth_app_grant_repository import OAuthAppGrantRepository
from gateway.repositories.organizations import BootstrapMarkerRepository, OrganizationRepositories
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationMemberRepository,
    UserRepository,
)
from gateway.services.api_keys import ApiKeyDeactivationListener
from gateway.services.identity import (
    DeploymentAdministrationAccess,
    DeploymentUserService,
    IdentityService,
    OAuthAppService,
    OIDCConnectionService,
)
from gateway.services.mail import Mailer
from gateway.services.oauth_service import redirect_uri
from gateway.services.organizations import BootstrapOperatorMarker, OrganizationClaimedDomains
from gateway.services.tenancy import (
    DomainAutoJoinListener,
    OrganizationService,
    VerifiedDomainLookup,
)
from gateway.services.tenancy.provisioning_service import ensure_bootstrap_identity


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
    identity = session_identity
    if identity is None:
        identity = await ensure_bootstrap_identity(
            db,
            uow=uow,
            membership_listener=membership_listener,
        )
    # Who made each change the request's span records (core/change_audit.py).
    trace.get_current_span().set_attribute("enduser.id", str(identity.id))
    return identity


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
    if not await deployment_administration_access(db).has_administration_access(identity):
        record_auth_failure("not_deployment_operator")
        raise DeploymentAdministrationUnavailableError


def get_deployment_user_service(
    db: Annotated[AsyncSession, Depends(get_db)], uow: UnitOfWorkDep
) -> DeploymentUserService:
    """Wire account changes and self-service key deactivation to the request's single transaction."""
    return DeploymentUserService(
        UserRepository(db),
        OrganizationMemberRepository(db),
        BootstrapOperatorMarker(BootstrapMarkerRepository(db)),
        uow=uow,
        sessions=DashboardSessionRepository(db),
        keys=ApiKeyDeactivationListener(ApiKeyRepositories.on(uow)),
    )


DeploymentUserServiceDep = Annotated[DeploymentUserService, Depends(get_deployment_user_service)]


def get_deployment_administration_access(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DeploymentAdministrationAccess:
    """Read deployment administration authority without constructing mutation collaborators."""
    return deployment_administration_access(db)


DeploymentAdministrationAccessDep = Annotated[
    DeploymentAdministrationAccess, Depends(get_deployment_administration_access)
]


def get_identity_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    identity_provider: IdentityProviderPortDep,
    membership_listener: MembershipListenerDep,
) -> IdentityService:
    """Build the identity service on the request's Unit of Work.

    Its own organization service is built on the same Unit of Work, not the deployment's shared
    read-only one, because provisioning an SSO sign-in's account writes through it. It carries
    the membership listener because the sign-in paths built here grant a workspace membership,
    which has to be announced.
    """
    organizations = OrganizationService(db, membership_listener=membership_listener, uow=uow)
    return IdentityService(
        uow,
        IdentityRepositories.on(uow),
        identity_provider=identity_provider,
        domain_claims=VerifiedDomainLookup(OrganizationDomainRepository(uow)),
        organizations=organizations,
        bootstrap_operator=BootstrapOperatorMarker(OrganizationRepositories.on(uow).bootstrap_marker),
        sign_in_listener=DomainAutoJoinListener(
            OrganizationDomainRepository(uow), OrganizationMemberRepository(uow), organizations
        ),
        session_ttl_hours=config.dashboard_session_ttl_hours,
        terms_url=config.terms_url,
        mailer=Mailer(config),
        open_signup=config.open_signup,
        password_reset_expiry_hours=config.password_reset_expiry_hours,
    )


IdentityServiceDep = Annotated[IdentityService, Depends(get_identity_service)]


def get_oidc_connection_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    uow: UnitOfWorkDep,
    organizations: OrganizationServiceDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> OIDCConnectionService:
    """Build the organization OIDC connection service on the request's Unit of Work.

    ``sign_in_states`` is the one repository built on the session instead, for the reason the
    service's own attribute gives: the sign-in client is used after the block that built it.

    ``sso_redirect_uri`` is None where this deployment has no public address,
    because a path alone is not registerable at a provider.
    """
    return OIDCConnectionService(
        uow,
        IdentityRepositories.on(uow),
        organizations=organizations,
        domains=OrganizationClaimedDomains(OrganizationDomainRepository(uow)),
        sign_in_states=OAuthPendingStateRepository(db),
        allow_private_hosts=config.oidc_allow_private_hosts,
        passkey_rp_id=config.webauthn_relying_party.rp_id if config.webauthn_relying_party else None,
        offered_providers=frozenset(
            provider for provider in OAUTH_PROVIDERS if config.oauth_client_credentials(provider) is not None
        ),
        sso_redirect_uri=redirect_uri(config, "sso") if config.public_base_url else None,
    )


OIDCConnectionServiceDep = Annotated[OIDCConnectionService, Depends(get_oidc_connection_service)]


def get_oauth_app_service(
    request: Request,
    uow: UnitOfWorkDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    key_format: ApiKeyFormatPortDep,
    organizations: OrganizationServiceDep,
    workspaces: WorkspaceServiceDep,
    budgets: BudgetServiceDep,
) -> OAuthAppService:
    """Build app authorization using the external authorization API origin as issuer."""
    return OAuthAppService(
        uow,
        OAuthAppGrantRepository(uow),
        get_api_key_service(uow),
        key_format,
        organizations,
        workspaces,
        budgets,
        apps=config.oauth_apps,
        open_apps=config.oauth_open_apps,
        spend_limits=config.budget_strategy.strip().lower() != "disabled",
        ui_link=config.ui_link,
        issuer=_oauth_app_issuer(request, config),
    )


def _oauth_app_issuer(request: Request, config: GatewayConfig) -> str:
    """The origin of ``public_base_url``, or of the request when it is unset.

    A proxy in front of the gateway can rewrite the host and terminate TLS, so
    the request alone would name the address behind it, which an app that
    checks ``iss`` against the address it called refuses. Serialized the way a
    browser writes an origin: lowercase, and no port where it is the scheme's
    default.
    """
    public = urlsplit(config.public_base_url or "")
    if not (public.scheme and public.hostname):
        return str(request.base_url.replace(path="", query="", fragment="")).rstrip("/")
    scheme = public.scheme.lower()
    host = f"[{public.hostname}]" if ":" in public.hostname else public.hostname
    port = "" if public.port in (None, {"http": 80, "https": 443}.get(scheme)) else f":{public.port}"
    return f"{scheme}://{host}{port}"


OAuthAppServiceDep = Annotated[OAuthAppService, Depends(get_oauth_app_service)]
