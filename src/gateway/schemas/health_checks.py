"""Request and response models of the health-checks domain."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gateway.core.addresses import normalized_address

MAX_ALERT_EMAILS = 20


class HealthCheckSettingsPublic(BaseModel):
    """Whether the checks run, how often, and what they send."""

    schedule_enabled: bool
    interval_minutes: int
    prompt: str
    max_tokens: int
    reasoning_max_tokens: int
    reasoning_every_runs: int
    next_run_at: datetime | None
    run_requested: bool
    alert_emails: list[str]


class HealthCheckSettingsUpdate(BaseModel):
    """Settings to change. An omitted field keeps its value."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"example": {"schedule_enabled": True, "interval_minutes": 60}}
    )

    schedule_enabled: bool | None = Field(default=None, description="Run the checks on a schedule.")
    interval_minutes: int | None = Field(default=None, ge=15, le=1440, description="Minutes between scheduled runs.")
    prompt: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description="The user message every model is sent. A check passes on any answer, whatever it says.",
    )
    max_tokens: int | None = Field(
        default=None, ge=1, le=4096, description="Output cap for a model that does not reason."
    )
    reasoning_max_tokens: int | None = Field(
        default=None, ge=1, le=8192, description="Output cap for a reasoning model."
    )
    reasoning_every_runs: int | None = Field(
        default=None, ge=0, le=48, description="Include reasoning models every Nth run; 0 leaves them out."
    )
    alert_emails: list[str] | None = Field(
        default=None,
        max_length=MAX_ALERT_EMAILS,
        description=(
            "Who is emailed when a model stops answering and when it answers again. An empty list emails nobody."
        ),
        json_schema_extra={"items": {"type": "string", "format": "email"}},
    )

    @field_validator("alert_emails")
    @classmethod
    def _addresses(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        addresses: list[str] = []
        for raw in value:
            address = normalized_address(raw)
            if address is None:
                msg = f"Not a valid email address: {raw!r}"
                raise ValueError(msg)
            if address not in addresses:
                addresses.append(address)
        return addresses


class HealthCheckRunPublic(BaseModel):
    """One pass over every offering."""

    id: str
    started_at: datetime
    finished_at: datetime | None
    included_reasoning: bool
    total: int
    passed: int
    failed: int


class HealthCheckResultPublic(BaseModel):
    """What one offering answered in the latest finished run."""

    selector: str
    provider: str
    status: Literal["pass", "fail", "skip"]
    status_code: int | None
    error: str | None
    latency_ms: int
    checked_at: datetime
    failing_since: datetime | None = Field(
        default=None,
        description=(
            "When a failing model's unbroken run of failures began, as far back as results are kept (48 hours). "
            "Null when it answered."
        ),
    )
    failed_runs: int = Field(default=0, description="How many runs in a row it has failed.")


class ProviderHistoryPoint(BaseModel):
    """How one provider did in one run."""

    started_at: datetime
    answered: int
    total: int


class ProviderHealthSummary(BaseModel):
    """One provider's offerings in the latest finished run.

    ``down`` when none answered, ``degraded`` when some did not, ``ok`` when all did.
    """

    provider: str
    status: Literal["ok", "degraded", "down"]
    answered: int
    total: int
    top_error: str | None = Field(description="The most common failure, with the model id taken out.")
    top_error_count: int
    history: list[ProviderHistoryPoint] = Field(description="Every finished run in the last 24 hours, oldest first.")


class HealthCheckStatus(BaseModel):
    """The Health page: settings, the run in progress, and the latest finished run by provider."""

    settings: HealthCheckSettingsPublic
    running: HealthCheckRunPublic | None
    last_run: HealthCheckRunPublic | None
    providers: list[ProviderHealthSummary]
    results: list[HealthCheckResultPublic]


class HealthCheckDisabled(BaseModel):
    """What switching off the failing models did."""

    disabled: int = Field(description="How many models were switched off on their hosted provider.")


class HealthCheckRecoveryCreate(BaseModel):
    """One outage a run ended, as it is recorded."""

    selector: str
    provider: str
    failing_since: datetime
    recovered_at: datetime
    failed_runs: int
    status_code: int | None
    error: str | None
