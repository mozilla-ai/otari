"""The one field a person explicitly supplies."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeedbackSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)

    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def require_utf8(cls, value: str) -> str:
        value.encode("utf-8")
        return value
