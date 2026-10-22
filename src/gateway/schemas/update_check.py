"""Request and response models of the update-check domain."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

Severity = Literal["low", "medium", "high", "critical"]

MAX_SUMMARY = 500
MAX_ADVISORIES = 50


def _require_https(value: str) -> str:
    # The dashboard renders these as links, so a javascript: or data: URL would run in its origin.
    if not value.lower().startswith("https://"):
        msg = "link must start with https://"
        raise ValueError(msg)
    return value


HttpsLink = Annotated[str, AfterValidator(_require_https)]


class UpdateCheckAdvisory(BaseModel):
    """A published security advisory for otari that covers the version this deployment ran when it checked."""

    id: str = Field(max_length=64)
    summary: str = Field(max_length=MAX_SUMMARY)
    severity: Severity
    url: HttpsLink = Field(max_length=500)
    fixed_in: str | None = Field(default=None, max_length=64)

    @field_validator("summary", mode="before")
    @classmethod
    def _shorten(cls, value: Any) -> Any:
        # An advisory's summary is its author's (GitHub allows 1024 characters): cut, rather than refuse the reply.
        return value[: MAX_SUMMARY - 1] + "…" if isinstance(value, str) and len(value) > MAX_SUMMARY else value


class UpdateCheckReply(BaseModel):
    """What otari.ai answers a check with. Anything else in the reply is ignored."""

    latest_version: str | None = Field(default=None, max_length=64)
    release_url: HttpsLink | None = Field(default=None, max_length=500)
    advisories: list[UpdateCheckAdvisory] = Field(default_factory=list, max_length=MAX_ADVISORIES)

    @field_validator("advisories", mode="before")
    @classmethod
    def _first_advisories(cls, value: Any) -> Any:
        return value[:MAX_ADVISORIES] if isinstance(value, list) else value


class UpdateCheckStatus(BaseModel):
    """Whether this deployment checks for new releases, who decided, what it sends, and what it last heard."""

    enabled: bool | None = Field(description="Whether checks are sent. Null until an operator answers.")
    answered_by: Literal["config", "operator"] | None = Field(
        description=(
            "Who decided: config.yml or the environment (the dashboard cannot change it), an operator in the "
            "dashboard, or nobody yet."
        )
    )
    current_version: str
    checkable: bool = Field(
        description="Whether this build carries a release version. A development build is never checked."
    )
    install_id: str = Field(description="The random id sent with each check.")
    endpoint: str = Field(description="Where checks are sent.")
    last_checked_at: datetime | None
    next_check_at: datetime | None
    latest_version: str | None
    release_url: str | None
    update_available: bool
    advisories: list[UpdateCheckAdvisory] = Field(
        description="Security advisories that affect the version running now, as of the last check."
    )


class UpdateCheckAnswer(BaseModel):
    """An operator's answer to whether this deployment checks for new releases."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {"enabled": True}})

    enabled: bool
