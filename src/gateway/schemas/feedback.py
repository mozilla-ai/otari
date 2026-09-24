"""The one field a person explicitly supplies."""

from pydantic import BaseModel, ConfigDict, Field


class FeedbackSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)

    message: str = Field(min_length=1, max_length=4000)
