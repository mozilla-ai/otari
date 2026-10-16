"""Request and response models of the identity domain."""

from datetime import datetime

from pydantic import BaseModel, Field

from gateway.models.identity import MailedLinkStatus

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


__all__ = ["MailedLinkRequest", "MailedLinkStatusResponse", "PasswordLinkPublic"]
