"""The fields a person explicitly supplies."""

from pydantic import BaseModel, ConfigDict, Field


class FeedbackSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)

    message: str = Field(min_length=1, max_length=4000)
    # The address the person saw when they agreed, never the one sent: the
    # gateway sends the session's own, and only when it is this one, so a page
    # still showing an earlier account cannot share the next one's.
    share_email: str | None = Field(
        default=None,
        max_length=255,
        description=(
            "The signed-in account's email as the form showed it, to send with the message so the Otari team can "
            "reply. Refused when the caller has no email the team can reply to, or when it is not this one."
        ),
    )
