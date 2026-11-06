"""Request and response models of the identity domain."""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from gateway.models.identity import MAX_BLOCK_REASON_LENGTH, MAX_EMAIL_LENGTH, BlockedEmail, MailedLinkStatus
from gateway.models.tenancy import OrganizationDomainRole

# A generated token is about 43 characters; this bounds the request body, it is not a format.
_MAX_SUBMITTED_TOKEN = 512


class PasswordLinkPublic(BaseModel):
    """A single-use link that sets an account's password, and whether it was mailed."""

    mail_sent: bool = Field(
        description=(
            "Whether the link was mailed to the account's address. False when this deployment cannot mail "
            "links, or the send failed."
        )
    )
    # A bearer link handed back on purpose, the way an invitation's accept_link is: only a deployment operator
    # can call the route, and only when mail did not go out, since the operator is then the one way to reach
    # the owner. It is never logged, and a mailed link is never returned.
    link: str | None = Field(
        default=None,
        description=(
            "The link to pass on to the account's owner yourself, set only when mail_sent is false. Choosing a "
            "password through it records the address as vouched for rather than proven, since the mailbox never "
            "saw the link."
        ),
    )
    expires_at: datetime = Field(description="When the link stops working.")


class MailedLinkRequest(BaseModel):
    """A token from a verification or password reset link."""

    token: str = Field(max_length=_MAX_SUBMITTED_TOKEN, description="The token from the link.")


class MailedLinkStatusResponse(BaseModel):
    """Whether the link can still be used. Checking it does not use it."""

    status: MailedLinkStatus = Field(
        description=(
            "valid, expired, or invalid. invalid covers a link that was never issued, one already used, "
            "and one whose account was deactivated."
        )
    )


class BlockedEmailCreateRequest(BaseModel):
    """An address or a domain to block."""

    value: str = Field(
        min_length=1,
        # Room for an ``@`` or ``*.`` prefix in front of the longest stored entry.
        max_length=MAX_EMAIL_LENGTH + 2,
        description=(
            "An address (person@example.com) or a domain (example.com, also accepted as @example.com or "
            "*.example.com). A domain also blocks its subdomains."
        ),
    )
    reason: str | None = Field(
        default=None,
        max_length=MAX_BLOCK_REASON_LENGTH,
        description="A note for other operators on why the entry exists. Never shown to the blocked person.",
    )


class BlockedEmailPublic(BaseModel):
    """One blocklist entry."""

    id: uuid.UUID
    pattern: str = Field(description="The normalized address or domain the entry blocks.")
    kind: str = Field(description="address or domain. A domain also blocks its subdomains.")
    reason: str | None
    created_by_user_id: uuid.UUID | None = Field(description="The operator who added it, or null once removed.")
    created_at: datetime

    @classmethod
    def of(cls, entry: BlockedEmail) -> "BlockedEmailPublic":
        """Render one entry."""
        return cls(
            id=entry.id,
            pattern=entry.pattern,
            kind="domain" if entry.is_domain else "address",
            reason=entry.reason,
            created_by_user_id=entry.created_by_user_id,
            created_at=entry.created_at,
        )


class BlockedEmailsPublic(BaseModel):
    data: list[BlockedEmailPublic]
    count: int


class BlockedEmailCreatedPublic(BlockedEmailPublic):
    """A new entry, and how many existing accounts it now blocks."""

    blocked_account_count: int = Field(
        description="How many existing accounts the entry matches. Their dashboard sessions were ended."
    )


class OIDCConnectionRequest(BaseModel):
    """What an organization administrator submits to configure its own sign-in connection."""

    issuer: str = Field(
        max_length=255, description="The identity provider's issuer URL, as the organization's own claim."
    )
    client_id: str = Field(max_length=255, description="The OAuth client ID this deployment authenticates with.")
    client_secret: str | None = Field(
        default=None,
        min_length=8,
        max_length=1024,
        description=(
            "The OAuth client secret. Stored encrypted and never served back. Leave it out to keep the "
            "one already stored, which is refused unless the issuer and the client ID are both unchanged."
        ),
    )
    display_name: str = Field(
        min_length=1, max_length=100, description="The identity provider's name, as shown to people signing in."
    )
    default_role: OrganizationDomainRole = Field(
        default="member", description="The role given to a person's first sign-in through this connection."
    )
    enabled: bool = Field(default=True, description="Whether this connection currently admits sign-in.")


class OIDCConnectionDomain(BaseModel):
    """One of the organization's email domains."""

    domain: str
    proof_expires_at: datetime | None = Field(
        description="When the domain's DNS proof stops being acted on, or None when it has never been verified."
    )
    enabled: bool = Field(description="Whether the domain's claim is turned on.")


class OIDCConnectionPublic(BaseModel):
    """An organization's own sign-in connection, as its admins see it.

    Carries ``client_secret_last4`` and never the secret itself.
    """

    id: uuid.UUID
    issuer: str = Field(description="The organization's own claim.")
    resolved_issuer: str = Field(description="What discovery confirmed the issuer to be.")
    client_id: str
    client_secret_last4: str
    display_name: str
    default_role: str
    enabled: bool
    discovered_at: datetime = Field(description="When the provider's configuration was last read.")
    domains: list[OIDCConnectionDomain] = Field(
        description=(
            "Every one of the organization's email domains, each with its DNS proof's expiry and whether its "
            "claim is enabled. Only a domain that is both enabled and has an unexpired proof admits a sign-in."
        )
    )


class OrganizationSsoPublic(BaseModel):
    """An organization's own OIDC connection, and what to register at the provider."""

    redirect_uri: str = Field(
        description=(
            "This deployment's fixed callback path. Register it at the identity provider, "
            "whether or not a connection is saved yet."
        )
    )
    connection: OIDCConnectionPublic | None = Field(
        description="The organization's own connection, or null until one is saved."
    )


__all__ = [
    "BlockedEmailCreateRequest",
    "BlockedEmailCreatedPublic",
    "BlockedEmailPublic",
    "BlockedEmailsPublic",
    "MailedLinkRequest",
    "MailedLinkStatusResponse",
    "OIDCConnectionDomain",
    "OIDCConnectionPublic",
    "OIDCConnectionRequest",
    "OrganizationSsoPublic",
    "PasswordLinkPublic",
]
