"""Settings for identity: how a person signs in, and who their account belongs to."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gateway.core.oauth_apps import parse_oauth_redirect
from gateway.core.settings_view import OMITTED, SettingsGroup, Shown
from gateway.types.key_scope import DEFAULT_KEY_SCOPES, KeyScope, KeyScopeSet


class OAuthAppConfig(BaseModel):
    """An operator-registered public OAuth client. Only loopback ports may vary."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    redirect_uris: list[str] = Field(min_length=1, max_length=20)
    scopes: list[KeyScope] = Field(
        default_factory=lambda: list(DEFAULT_KEY_SCOPES),
        min_length=1,
        description="The scopes of each key the app is issued. An app key is never a service key.",
    )

    @field_validator("redirect_uris")
    @classmethod
    def validate_redirects(cls, values: list[str]) -> list[str]:
        for value in values:
            if parse_oauth_redirect(value) is None:
                raise ValueError(
                    "OAuth redirects must be valid HTTPS URLs (HTTP loopback allowed), "
                    "without credentials, query or fragment"
                )
        return values

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, values: list[KeyScope]) -> list[KeyScope]:
        return list(KeyScopeSet.checked(values, is_service_key=False))


class IdentitySettings(BaseModel):
    oauth_apps: Annotated[dict[Annotated[str, Field(min_length=1, max_length=128)], OAuthAppConfig], OMITTED] = Field(
        default_factory=dict,
        description="Registered OAuth apps, keyed by public client ID. Empty disables app authorization.",
    )

    oidc_allow_private_hosts: Annotated[bool, Shown(SettingsGroup.TOOLS)] = Field(
        default=False,
        description=(
            "Let an organization's own OIDC issuer be discovered at a private, loopback or link-local "
            "address, for an identity provider on the deployment's own network. Off by default, because "
            "an organization administrator types the issuer in rather than the operator choosing it."
        ),
    )
