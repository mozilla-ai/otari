"""API key settings: the keys a deployment declares in config.yml, and the issuers a key may federate with.

A declared key carries a secret the operator chose, so the service that
presents it and this gateway share one value that survives restarts and
replicas. The secret is held as ``SecretStr`` and never appears in a log line,
an error message or an API response: validation errors here name the key,
never the value.
"""

import hashlib
import re
from typing import Annotated, Any, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from gateway.auth.models import API_KEY_PREFIX, MIN_API_KEY_LENGTH
from gateway.core.settings.declarations import declaration_digest
from gateway.core.settings_view import OMITTED, SECRET, SettingsGroup, Shown
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
    max_end_users: int | None = Field(
        default=None,
        ge=0,
        description="Most live end users the key may have on the budgets it assigns, or on any budget when it "
        "assigns none; null is no cap.",
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

    def declaration_digest(self) -> str:
        """A digest of this entry, which changes when any field config.yml sets for it, or its secret, does.

        The secret enters as its own SHA-256, so the digest stored beside the key
        says no more about it than the key's hash already does.
        """
        entry = self.model_dump(mode="json", exclude={"secret"})
        entry["secret_sha256"] = hashlib.sha256(self.secret.get_secret_value().encode()).hexdigest()
        return declaration_digest(entry)


def _https_url_problem(value: str) -> str | None:
    """Why ``value`` is not an absolute https URL without credentials, a query or a fragment, or None."""
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname:
        return "must be an absolute https URL"
    if parts.username or parts.password or parts.query or parts.fragment:
        return "must not carry credentials, a query or a fragment"
    return None


class FederationIssuer(BaseModel):
    """An OpenID Connect issuer whose identity tokens may stand in for an API key's secret."""

    model_config = ConfigDict(extra="forbid")

    issuer: str = Field(
        description=(
            "The issuer exactly as tokens carry it in their 'iss' claim, an https URL; "
            "for Google, https://accounts.google.com."
        ),
    )
    jwks_uri: str | None = Field(
        default=None,
        description=(
            "Where the issuer publishes its signing keys. Unset discovers it from "
            "<issuer>/.well-known/openid-configuration."
        ),
    )

    @field_validator("issuer")
    @classmethod
    def _issuer_is_https(cls, value: str) -> str:
        problem = _https_url_problem(value)
        if problem is not None:
            msg = f"federation issuer '{value}' {problem}"
            raise ValueError(msg)
        return value

    @field_validator("jwks_uri")
    @classmethod
    def _jwks_uri_is_https(cls, value: str | None) -> str | None:
        if value is None:
            return None
        problem = _https_url_problem(value)
        if problem is not None:
            msg = f"federation jwks_uri '{value}' {problem}"
            raise ValueError(msg)
        return value


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

    federation_issuers: Annotated[list[FederationIssuer], OMITTED] = Field(
        default_factory=list,
        description=(
            "OpenID Connect issuers whose identity tokens may authenticate as an API key, in place of its secret, "
            "when they match one of the key's federation rules. Each entry is an https issuer URL or a mapping "
            "with 'issuer' and an optional 'jwks_uri'. Empty, the default, turns workload identity off. "
            "Standalone and hosted modes only."
        ),
    )

    federation_allow_private_hosts: Annotated[bool, Shown(SettingsGroup.TOOLS)] = Field(
        default=False,
        description=(
            "Let a federation issuer's discovery document and signing keys be fetched from a private, loopback or "
            "link-local address, for an issuer on the deployment's own network. Off by default."
        ),
    )
    federation_access_token_ttl_sec: Annotated[int, Shown(SettingsGroup.GENERAL)] = Field(
        default=600,
        ge=60,
        le=3600,
        description=(
            "How long an access token from POST /api/v1/oauth/token lasts, in seconds, and never past the identity "
            "token it was exchanged for. Between 60 and 3600."
        ),
    )

    @field_validator("federation_issuers", mode="before")
    @classmethod
    def _issuer_shorthand(cls, value: Any) -> Any:
        """Accept a bare issuer URL for an entry with no explicit ``jwks_uri``."""
        if isinstance(value, list):
            return [{"issuer": entry} if isinstance(entry, str) else entry for entry in value]
        return value

    @field_validator("federation_issuers")
    @classmethod
    def _issuers_unique(cls, value: list[FederationIssuer]) -> list[FederationIssuer]:
        seen: set[str] = set()
        for entry in value:
            if entry.issuer in seen:
                msg = f"federation issuer '{entry.issuer}' is listed more than once"
                raise ValueError(msg)
            seen.add(entry.issuer)
        return value

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
