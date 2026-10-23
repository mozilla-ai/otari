"""The fields a person explicitly supplies."""

from pydantic import BaseModel, ConfigDict, Field


class FeedbackSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)

    message: str = Field(min_length=1, max_length=4000)
    # A yes or no rather than an address, so nobody can put someone else's
    # email on a message: the gateway reads the address from the session.
    share_email: bool = Field(
        default=False,
        strict=True,
        description=(
            "Send the signed-in account's email with the message so the Otari team can reply. "
            "Refused when the caller has no email the team can reply to."
        ),
    )
