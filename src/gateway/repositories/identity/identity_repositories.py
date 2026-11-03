from dataclasses import dataclass
from typing import Self

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.identity.blocked_email_repository import BlockedEmailRepository
from gateway.repositories.identity.dashboard_session_repository import DashboardSessionRepository
from gateway.repositories.identity.oauth_identity_link_repository import OAuthIdentityLinkRepository
from gateway.repositories.identity.oidc_connection_repository import OIDCConnectionRepository
from gateway.repositories.identity.webauthn_credential_repository import WebAuthnCredentialRepository
from gateway.repositories.tenancy import UserRepository


@dataclass(frozen=True)
class IdentityRepositories:
    """The identity domain's repositories, all on one Unit of Work."""

    blocked_emails: BlockedEmailRepository
    connections: OIDCConnectionRepository
    links: OAuthIdentityLinkRepository
    passkeys: WebAuthnCredentialRepository
    sessions: DashboardSessionRepository
    users: UserRepository

    @classmethod
    def on(cls, uow: UnitOfWork) -> Self:
        """Build every repository on this Unit of Work."""
        return cls(
            blocked_emails=BlockedEmailRepository(uow),
            connections=OIDCConnectionRepository(uow),
            links=OAuthIdentityLinkRepository(uow),
            passkeys=WebAuthnCredentialRepository(uow),
            sessions=DashboardSessionRepository(uow),
            users=UserRepository(uow),
        )
