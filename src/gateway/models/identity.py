"""Tables the identity domain owns."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import Enum, ForeignKey, String, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime

MAX_EMAIL_LENGTH = 255
MAX_PROVIDER_LENGTH = 50
MAX_SUBJECT_LENGTH = 255


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


__all__ = ["EmailAssertion", "MailedLinkStatus", "OAuthIdentityLink", "ProviderIdentity"]
