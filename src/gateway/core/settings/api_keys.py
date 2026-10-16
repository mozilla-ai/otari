"""API key settings: the keys a deployment declares in config.yml.

A declared key carries a secret the operator chose, so the service that
presents it and this gateway share one value that survives restarts and
replicas. The secret is held as ``SecretStr`` and never appears in a log line,
an error message or an API response: validation errors here name the key,
never the value.
"""

import re
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from gateway.auth.models import API_KEY_PREFIX, MIN_API_KEY_LENGTH
from gateway.core.settings_view import SECRET
from gateway.models.api_keys import MAX_END_USER_BUDGETS
from gateway.models.budgets import BUDGET_ID_PATTERN

# A declared key's name follows the budget id rule, so it reads the same in a
# config file, a log line and the dashboard.
_CONFIG_NAME = re.compile(BUDGET_ID_PATTERN)

# How to make a secret that passes, quoted in every refusal.
SECRET_RECIPE = "python -c \"import secrets; print('tk-' + secrets.token_urlsafe(48))\""


def _mask(value: Any) -> Any:
    """Hide a value that may be a secret from the errors pydantic builds, which quote their input."""
    return SecretStr(value) if isinstance(value, str) else value


class ApiKeyConfig(BaseModel):
    """One API key config.yml declares.

    Unknown fields are refused rather than ignored, so a misspelled setting
    fails the start instead of being dropped.
    """

    model_config = ConfigDict(extra="forbid")

    secret: SecretStr = Field(
        description=(
            "The key's secret, which requests present. Interpolate it from the environment "
            f"(${{VAR}}). It starts with '{API_KEY_PREFIX}', is at least {MIN_API_KEY_LENGTH} characters "
            f"long, and is random: {SECRET_RECIPE}"
        ),
    )
    key_name: str | None = Field(default=None, description="Display name; defaults to the key's config name")
    user_id: str | None = Field(
        default=None,
        description="The user the key bills to, created when missing. Unset uses the shared 'default' user.",
    )
    is_service_key: bool = Field(
        default=False,
        description="Lets a request name an end user in its 'user' field, created on first use under this key's user.",
    )
    end_user_budget_id: str | None = Field(
        default=None, description="Budget a new end user starts on when the request names none."
    )
    end_user_budget_ids: list[str] | None = Field(
        default=None,
        max_length=MAX_END_USER_BUDGETS,
        description="Budgets a request may start a new end user on with Otari-End-User-Budget.",
    )
    ceiling: str | None = Field(
        default=None,
        description=(
            "Budget that caps this key as a whole, across all of its end users (a scoped budget on the key). "
            "Unset leaves any ceiling the key has as it is."
        ),
    )
    exclude_from_budget: bool = Field(
        default=False, description="Log the key's cost without reserving or enforcing any budget."
    )
    reject_user_mismatch: bool | None = Field(
        default=None, description="Per-key override of reject_user_mismatch; null inherits the deployment setting."
    )

    @model_validator(mode="before")
    @classmethod
    def _mask_secret_inputs(cls, data: Any) -> Any:
        """Wrap every input that could be the secret before anything is validated.

        A validation error quotes its input, so a value written in place of the
        mapping, or under a misspelled field name, would otherwise reach the log
        of a gateway that refused to start.
        """
        if not isinstance(data, dict):
            return _mask(data)
        return {
            name: value if name in cls.model_fields and name != "secret" else _mask(value)
            for name, value in data.items()
        }

    @model_validator(mode="after")
    def _default_on_list(self) -> Self:
        listed = self.end_user_budget_ids
        if listed is not None and self.end_user_budget_id is not None and self.end_user_budget_id not in listed:
            msg = "end_user_budget_id must be one of end_user_budget_ids"
            raise ValueError(msg)
        return self


class ApiKeySettings(BaseModel):
    """API keys a deployment declares rather than mints."""

    api_keys: Annotated[dict[str, ApiKeyConfig], SECRET] = Field(
        default_factory=dict,
        description=(
            "API keys this deployment declares, keyed by a stable name. Each start creates a missing key and "
            "writes the declared secret and settings over an existing one, so config.yml wins over a change made "
            "through the API. A key removed from here is kept as it is. Standalone mode only."
        ),
    )

    @field_validator("api_keys", mode="before")
    @classmethod
    def _mask_non_mapping(cls, value: Any) -> Any:
        """Keep a list or a scalar written in place of the mapping out of the error that refuses it."""
        if isinstance(value, dict):
            return value
        return SecretStr(repr(value))

    @field_validator("api_keys")
    @classmethod
    def _validate_config_names(cls, value: dict[str, ApiKeyConfig]) -> dict[str, ApiKeyConfig]:
        for name in value:
            if not _CONFIG_NAME.match(name):
                msg = (
                    f"api key name '{name}' must start with a letter or digit and hold only letters, digits, "
                    "'.', '_' and '-' (at most 128 characters)"
                )
                raise ValueError(msg)
        return value
