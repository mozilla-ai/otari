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

_SECRET_BODY = re.compile(r"^[A-Za-z0-9_-]+$")

# Distinct characters a declared secret holds after its prefix. A random
# ``token_urlsafe(48)`` holds about forty of the 64 it draws from, so this only
# refuses a value somebody typed, such as a repeated character or a word.
_MIN_DISTINCT_SECRET_CHARACTERS = 16

# How to make a secret that passes, quoted in every refusal.
SECRET_RECIPE = "python -c \"import secrets; print('tk-' + secrets.token_urlsafe(48))\""


def declared_secret_problem(secret: str) -> str | None:
    """Say what is wrong with a declared secret, or None when it is usable.

    The open-source key format: the ``tk-`` prefix, at least
    ``MIN_API_KEY_LENGTH`` characters, URL-safe characters only. A key in that
    format is checked locally by every key format a build binds, because a
    format refuses only a key that claims its own shape and fails it. The answer
    never quotes the secret.
    """
    if not secret:
        return "is empty"
    if secret != secret.strip():
        return "has leading or trailing whitespace"
    if not secret.startswith(API_KEY_PREFIX):
        return f"must start with '{API_KEY_PREFIX}'"
    if len(secret) < MIN_API_KEY_LENGTH:
        return f"must be at least {MIN_API_KEY_LENGTH} characters long"
    body = secret[len(API_KEY_PREFIX) :]
    if not _SECRET_BODY.match(body):
        return f"may hold only letters, digits, '-' and '_' after '{API_KEY_PREFIX}'"
    if len(set(body)) < _MIN_DISTINCT_SECRET_CHARACTERS:
        return "looks like a placeholder rather than a random value"
    return None


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

    def declared_api_key_problems(self, master_key: str | None) -> list[str]:
        """Every reason a declared key cannot be used, each naming the key and never its secret."""
        problems: list[str] = []
        owners: dict[str, str] = {}
        for name, key in self.api_keys.items():
            secret = key.secret.get_secret_value()
            if (problem := declared_secret_problem(secret)) is not None:
                problems.append(f"api_keys.{name}.secret {problem}; generate one with {SECRET_RECIPE}")
                continue
            if master_key is not None and secret == master_key:
                problems.append(f"api_keys.{name}.secret must differ from the master key")
            if (other := owners.setdefault(secret, name)) != name:
                problems.append(f"api_keys.{name}.secret is the same as api_keys.{other}.secret")
        return problems
