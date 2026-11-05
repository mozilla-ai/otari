"""Request and response models of the identity domain."""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from gateway.models.identity import MAX_BLOCK_REASON_LENGTH, MAX_EMAIL_LENGTH, BlockedEmail, MailedLinkStatus
from gateway.models.tenancy import OrganizationDomainRole

# A generated token is about 43 characters; this bounds the request body, it is not a format.
_MAX_SUBMITTED_TOKEN = 512
# A provider's authorization code is a few hundred characters at most; this is a sanity ceiling
# on an unauthenticated request body, not a format.
_MAX_SSO_CODE_LENGTH = 2048
# A state of the wrong length matches nothing stored; this bounds what an unauthenticated caller
# can make this process hash.
_MAX_SSO_STATE_LENGTH = 512
# A sanity ceiling on an issuer URL, not a format.
_MAX_SSO_ISS_LENGTH = 2048


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


class SsoLookupRequest(BaseModel):
    """An address to resolve an organization's own sign-in connection from."""

    email: str = Field(max_length=MAX_EMAIL_LENGTH, description="The address to resolve an organization from.")


class SsoLookupResponse(BaseModel):
    """Whether an organization's own connection admits a sign-in at this address.

    An unknown domain, one no organization has proven, and a proven one whose organization holds
    no enabled connection all answer alike: ``found`` is false and the other two fields are null,
    so the response cannot be read as a directory of which organizations use Otari.
    """

    found: bool = Field(description="Whether an organization's own connection admits a sign-in at this address.")
    organization_name: str | None = Field(default=None, description="The organization's own name, set only when found.")
    slug: str | None = Field(
        default=None, description="The organization's slug, to start the sign-in with, set only when found."
    )


class SsoAuthorizeResponse(BaseModel):
    """Where to send the browser, and the state to check when it comes back."""

    authorization_url: str = Field(description="The identity provider's consent screen to navigate to.")
    state: str = Field(
        description=(
            "An opaque CSRF value to keep for the length of the redirect and send back with the "
            "authorization code. The response also sets an HttpOnly cookie that the callback requires, "
            "so the exchange can only be completed from the browser this call was made from."
        )
    )


class SsoCallbackRequest(BaseModel):
    """The authorization code a provider handed the browser, for a sign-in through one organization's own connection.

    No ``slug``: the pending authorization the ``state`` names already resolves to one connection,
    so a browser cannot redeem a code against a connection other than the one that minted the state.
    """

    code: str = Field(
        max_length=_MAX_SSO_CODE_LENGTH, description="The authorization code from the provider's redirect."
    )
    state: str = Field(
        max_length=_MAX_SSO_STATE_LENGTH,
        description="The 'state' from the provider's redirect, as issued by /authorize.",
    )
    terms_accepted: bool = Field(
        default=False,
        description=(
            "Whether the person accepted this deployment's terms before going to the provider. Required where "
            "the bootstrap reports a terms_url and the sign-in is the first way in to an account nobody has "
            "claimed yet; ignored otherwise."
        ),
    )
    iss: str | None = Field(
        default=None,
        max_length=_MAX_SSO_ISS_LENGTH,
        description=(
            "The 'iss' from the provider's redirect (RFC 9207), where it sent one. Checked against this "
            "connection's issuer when its discovery document advertises the parameter; a provider that does "
            "not advertise it sends none."
        ),
    )


class SsoSessionPublic(BaseModel):
    """A dashboard session minted by signing in through an organization's own connection."""

    expires_at: datetime = Field(description="When the session cookie stops being accepted.")
    user_id: uuid.UUID = Field(description="The identity this session speaks for.")
    active_organization_id: uuid.UUID | None = Field(
        description=(
            "The organization that identity is acting in, which scopes every tenancy surface. Null when it "
            "is an active member of none."
        )
    )
    password_removed: bool = Field(
        default=False,
        description=(
            "Whether this sign-in removed a password the account held: the first provider sign-in on an "
            "address nobody had proven removes the password set before then."
        ),
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
    "SsoAuthorizeResponse",
    "SsoCallbackRequest",
    "SsoLookupRequest",
    "SsoLookupResponse",
    "SsoSessionPublic",
]
