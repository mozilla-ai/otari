"""Request and response models of the rate-limits domain."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gateway.core.config import RateLimitRule


class RateLimitRuleCreate(RateLimitRule):
    """A rule to add. The same fields, limits and validation as a ``rate_limits`` entry in config.yml."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"name": "keys", "per": "key", "rpm": 600, "tpm": 200000}},
    )


class RateLimitRuleUpdate(BaseModel):
    """Fields to change on a stored rule. An omitted field keeps its value; ``null`` clears a limit.

    The merged rule must still set at least one of rpm, tpm or max_concurrent.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {"rpm": 1200}})

    per: Literal["deployment", "key", "user", "model"] | None = Field(
        default=None, description="What one count is shared by."
    )
    models: list[str] | None = Field(
        default=None, description="The instance:model names a per: model rule limits; null for any other rule."
    )
    rpm: int | None = Field(default=None, ge=1, description="Requests per minute.")
    tpm: int | None = Field(default=None, ge=1, description="Tokens per minute.")
    tpm_admission: Literal["estimate", "used"] | None = Field(
        default=None, description="'estimate' holds a request's estimate; 'used' counts only what it used."
    )
    max_concurrent: int | None = Field(default=None, ge=1, description="Requests in flight at once.")
    lease_sec: float | None = Field(default=None, gt=0, description="How long a max_concurrent slot is held at most.")


class RateLimitRulePublic(RateLimitRule):
    """One rule in effect, and where it is defined."""

    source: Literal["config", "dashboard"] = Field(
        description="'config' for a rule from config.yml, which is read-only here; 'dashboard' for a stored one."
    )
    updated_at: datetime | None = Field(default=None, description="When a stored rule last changed.")


class RateLimitRulesPublic(BaseModel):
    """Every rule in effect: config-file rules first, then stored ones, each in name order."""

    rules: list[RateLimitRulePublic]
