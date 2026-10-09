"""Budget settings."""

import re
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gateway.core.settings_view import OMITTED, SettingsGroup, Shown
from gateway.models.budgets import BUDGET_ID_PATTERN, MAX_COUNT_LIMIT, MAX_MINUTE_LIMIT, ResetAlignment
from gateway.models.money import MAX_USD_LIMIT

STREAM_MISSING_USAGE_POLICIES = ("estimate", "fail", "allow_free")

_BUDGET_ID = re.compile(BUDGET_ID_PATTERN)


class BudgetConfig(BaseModel):
    """One budget config.yml declares, with the fields and bounds ``PUT /budgets/{budget_id}`` accepts.

    Unknown fields are refused rather than ignored, so a misspelled limit fails
    the start instead of leaving the budget uncapped on that axis.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    max_budget: float | None = Field(default=None, ge=0, le=MAX_USD_LIMIT)
    token_limit: int | None = Field(default=None, ge=0, le=MAX_COUNT_LIMIT)
    request_limit: int | None = Field(default=None, ge=0, le=MAX_COUNT_LIMIT)
    rpm_limit: int | None = Field(default=None, ge=1, le=MAX_MINUTE_LIMIT)
    tpm_limit: int | None = Field(default=None, ge=1, le=MAX_MINUTE_LIMIT)
    budget_duration_sec: int | None = Field(default=None, gt=0)
    reset_alignment: ResetAlignment | None = None

    @model_validator(mode="after")
    def _single_period_source(self) -> Self:
        if self.budget_duration_sec is not None and self.reset_alignment is not None:
            msg = "a budget resets on budget_duration_sec or on reset_alignment, not both"
            raise ValueError(msg)
        return self


class BudgetSettings(BaseModel):
    """How spend is attributed to a user, held against a budget, and settled."""

    reject_user_mismatch: Annotated[bool, Shown(SettingsGroup.METERING)] = Field(
        default=True,
        description=(
            "When True (default), a non-master key whose request names a 'user' other than its own "
            "is rejected with 403. When False, the client-supplied 'user' is still forwarded to the "
            "provider (OpenAI-style end-user tag) but spend is always bound to the key's own user; "
            "use this if clients send arbitrary 'user' values for abuse tracking. This setting "
            "is the deployment-wide default: an individual key can override it in either "
            "direction with its own reject_user_mismatch (null inherits this setting). The "
            "master key may always bill an arbitrary user regardless of this setting."
        ),
    )
    budget_reservation_ttl_sec: Annotated[int, OMITTED] = Field(
        default=900,
        gt=0,
        description=(
            "How long a budget reservation may stay in flight before the sweep treats it as "
            "leaked and returns the hold. It must comfortably exceed the slowest request this "
            "deployment serves, because reclaiming a hold that is still live would let a "
            "concurrent request past a cap the in-flight one is already spending against."
        ),
    )
    budget_reservation_sweep_interval_sec: Annotated[int, OMITTED] = Field(
        default=300,
        ge=0,
        description=(
            "How often to sweep for leaked budget reservations across all users. 0 disables the "
            "sweep, leaving the opportunistic per-user reclaim that runs when a user next "
            "reserves. Standalone mode only."
        ),
    )
    budget_reservation_sweep_batch: Annotated[int, OMITTED] = Field(
        default=500,
        gt=0,
        description="Maximum leaked budget reservations one sweep pass reclaims before yielding.",
    )
    budget_reservation_retention_sec: Annotated[int, OMITTED] = Field(
        default=604800,
        ge=0,
        description=(
            "How long a settled, released or reclaimed budget reservation is kept before the "
            "sweep deletes it. The row exists to make an in-flight hold reclaimable; what a "
            "request cost is recorded durably in usage_logs, so this is an audit window rather "
            "than an accounting record. 0 keeps every row forever. Standalone mode only."
        ),
    )
    stream_missing_usage_policy: Annotated[str, Shown(SettingsGroup.METERING)] = Field(
        default="estimate",
        description=(
            "How to bill a streamed response that completes without provider usage data: "
            "'estimate' (charge the pre-debit estimate, default), 'fail' (charge estimate and mark "
            "the request errored), or 'allow_free' (release the reservation, legacy behavior)."
        ),
    )
    budget_strategy: Annotated[str, Shown(SettingsGroup.METERING)] = Field(
        default="for_update",
        description="Budget validation strategy: 'for_update' (default), 'cas' (lock-free), or 'disabled'.",
    )
    budget_estimate_default_output_tokens: Annotated[int, Shown(SettingsGroup.METERING)] = Field(
        default=1024,
        ge=0,
        description=(
            "Output-token count assumed when reserving budget for a request whose max output is "
            "unbounded. Used by the pre-debit estimate; reconciled to actual usage on completion."
        ),
    )

    budgets: Annotated[dict[str, BudgetConfig], OMITTED] = Field(
        default_factory=dict,
        description=(
            "Budgets this deployment declares, keyed by budget id. Each start creates a missing one and writes the "
            "declared values over an existing one, so config.yml wins over a change made through the API. A budget "
            "removed from here is kept as it is. Standalone mode only."
        ),
    )

    @field_validator("budgets")
    @classmethod
    def _validate_budget_ids(cls, value: dict[str, BudgetConfig]) -> dict[str, BudgetConfig]:
        for budget_id in value:
            if not _BUDGET_ID.match(budget_id):
                msg = (
                    f"budget id '{budget_id}' must start with a letter or digit and hold only letters, digits, "
                    "'.', '_' and '-' (at most 128 characters)"
                )
                raise ValueError(msg)
        return value

    @field_validator("stream_missing_usage_policy")
    @classmethod
    def _validate_stream_missing_usage_policy(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in STREAM_MISSING_USAGE_POLICIES:
            msg = f"stream_missing_usage_policy must be one of {sorted(STREAM_MISSING_USAGE_POLICIES)}, got '{value}'"
            raise ValueError(msg)
        return normalized
