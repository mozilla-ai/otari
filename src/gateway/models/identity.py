"""Tables the identity domain owns."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Enum, ForeignKey, Index, String, Text, UniqueConstraint, Uuid, func, true
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime

MAX_EMAIL_LENGTH = 255
MAX_PROVIDER_LENGTH = 255
MAX_SUBJECT_LENGTH = 255
MAX_BLOCK_REASON_LENGTH = 500

OIDC_PROVIDER_PREFIX = "oidc:"


class EmailAssertion(StrEnum):
    """What a provider says about the email address it returns.

    NOTE: A provider that sends no claim is ``UNASSERTED``, not ``UNVERIFIED``.
    """

    UNASSERTED = "unasserted"
    UNVERIFIED = "unverified"
    VERIFIED = "verified"

    @classmethod
    def from_claim(cls, email_verified: bool | None) -> "EmailAssertion":
        """Return the assertion for a provider's tri-state ``email_verified`` claim."""
        if email_verified is None:
            return cls.UNASSERTED
        return cls.VERIFIED if email_verified else cls.UNVERIFIED


class MailedLinkStatus(StrEnum):
    """Whether a mailed verification or password reset link can still be used.

    NOTE: ``INVALID`` covers a link that was never issued and one already used, because a used link's hash is gone.
    """

    VALID = "valid"
    EXPIRED = "expired"
    INVALID = "invalid"


@dataclass(frozen=True)
class ProviderIdentity:
    """What a provider has proven about the person who signed in.

    ``(provider, subject)`` names the person at the provider.
    ``provider`` is an open string, because a customer's own OIDC connections are not enumerable.
    """

    provider: str
    subject: str
    email: str | None
    full_name: str | None
    email_verified: EmailAssertion

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("provider", self.provider, MAX_PROVIDER_LENGTH),
            ("subject", self.subject, MAX_SUBJECT_LENGTH),
        ):
            if not value or len(value) > limit:
                msg = f"A provider identity's {name} must hold 1 to {limit} characters"
                raise ValueError(msg)
        if self.email is not None and len(self.email) > MAX_EMAIL_LENGTH:
            msg = f"A provider identity's email must hold at most {MAX_EMAIL_LENGTH} characters"
            raise ValueError(msg)


class OAuthIdentityLink(Base):
    """A provider identity linked to one account.

    The primary key is the provider's ``(provider, subject)`` pair, which names one person at that provider.
    An account holds at most one link per provider.

    NOTE: ``email_at_link`` and ``email_assertion_at_link`` record what the provider said when the link was made.
    They must not be used to look an account up.
    """

    __tablename__ = "oauth_identity_link"
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_oauth_identity_link_user_provider"),)

    provider: Mapped[str] = mapped_column(String(MAX_PROVIDER_LENGTH), primary_key=True)
    subject: Mapped[str] = mapped_column(String(MAX_SUBJECT_LENGTH), primary_key=True)
    # The unique constraint above leads with this column, so it also serves as the column's index.
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    email_at_link: Mapped[str | None] = mapped_column(String(MAX_EMAIL_LENGTH))
    email_assertion_at_link: Mapped[EmailAssertion] = mapped_column(
        Enum(
            EmailAssertion,
            native_enum=False,
            length=16,
            values_callable=lambda members: [member.value for member in members],
            validate_strings=True,
        )
    )
    linked_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class BlockedEmail(Base):
    """An address or a domain the deployment's operator has blocked from holding an account.

    ``pattern`` is a whole address or a domain, normalized by ``core.email_domains.blocklist_entry``; a
    domain also blocks its subdomains. A blocked address cannot sign up, be added or invited, or sign in.
    """

    __tablename__ = "blocked_email"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    pattern: Mapped[str] = mapped_column(String(MAX_EMAIL_LENGTH), unique=True, nullable=False)
    reason: Mapped[str | None] = mapped_column(String(MAX_BLOCK_REASON_LENGTH))
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("user.id", ondelete="SET NULL"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )

    @property
    def is_domain(self) -> bool:
        """Whether the entry blocks a whole domain rather than one address."""
        return "@" not in self.pattern


def oidc_identity_provider(issuer: str) -> str:
    """Return the identity provider naming everyone this OIDC issuer vouches for.

    The same value for every organization whose connection resolves to this issuer: the
    issuer is who proved the person's identity, not which organization's connection reached it.

    Raises:
        ValueError: The provider value is too long to store.

    """
    provider = f"{OIDC_PROVIDER_PREFIX}{issuer}"
    if len(provider) > MAX_PROVIDER_LENGTH:
        msg = (
            f"An OIDC issuer identifies a provider too long to store: "
            f"{len(provider)} characters, at most {MAX_PROVIDER_LENGTH} allowed"
        )
        raise ValueError(msg)
    return provider


def oidc_flow_provider(connection_id: uuid.UUID) -> str:
    """Return the pending-flow provider value naming one organization's OIDC connection.

    Distinct per connection even where two organizations share one issuer, because only a
    connection carries the client credentials a sign-in attempt resolves against.
    """
    return f"{OIDC_PROVIDER_PREFIX}{connection_id}"


class OIDCConnection(Base):
    """One organization's connection to its own OIDC identity provider.

    An organization holds at most one connection, by the unique constraint on
    ``organization_id``. ``issuer`` is the organization's own claim; ``resolved_issuer`` is
    what discovery confirmed it to be.

    NOTE: ``resolved_issuer`` carries no unique constraint. Two organizations may configure
    one shared identity provider, and a sign-in attempt narrows among them by other means.
    """

    __tablename__ = "oidc_connection"
    __table_args__ = (
        UniqueConstraint("organization_id", name="uq_oidc_connection_organization_id"),
        Index("ix_oidc_connection_resolved_issuer", "resolved_issuer"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization.id", ondelete="CASCADE"), nullable=False
    )
    issuer: Mapped[str] = mapped_column(String(255), nullable=False)
    resolved_issuer: Mapped[str] = mapped_column(String(255), nullable=False)
    client_id: Mapped[str] = mapped_column(String(255), nullable=False)
    encrypted_client_secret: Mapped[str] = mapped_column(Text, nullable=False)
    client_secret_last4: Mapped[str] = mapped_column(String(4), nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    default_role: Mapped[str] = mapped_column(String(32), nullable=False, default="member", server_default="member")
    enabled: Mapped[bool] = mapped_column(nullable=False, default=True, server_default=true())
    discovery_document: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    discovered_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)
    first_sign_in_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=func.now(),
    )


__all__ = [
    "MAX_BLOCK_REASON_LENGTH",
    "BlockedEmail",
    "EmailAssertion",
    "MailedLinkStatus",
    "OAuthIdentityLink",
    "OIDCConnection",
    "ProviderIdentity",
    "oidc_flow_provider",
    "oidc_identity_provider",
]
