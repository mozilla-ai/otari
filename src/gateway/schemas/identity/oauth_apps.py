"""RFC 6749 public-client authorization, token and revocation contracts."""

from collections.abc import Callable
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from gateway.core.oauth_apps import parse_oauth_redirect
from gateway.models.identity import MAX_DYNAMIC_CLIENT_NAME_LENGTH
from gateway.types.key_scope import KeyScope

OAuthClientId = Annotated[str, Field(min_length=1, max_length=128)]
OAuthRedirectUri = Annotated[str, Field(max_length=2048)]
OAuthState = Annotated[str, Field(min_length=16, max_length=256, pattern=r"^[A-Za-z0-9._~+/=-]+$")]
PKCECodeChallenge = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{43}$")]
AppKeyExpiration = Literal["1-hour", "1-day", "7-days", "30-days", "90-days", "180-days", "1-year", "never"]
AppKeyLimit = Annotated[Decimal, Field(gt=0, lt=1_000_000, decimal_places=2)]


def _suggestion(max_length: int) -> Callable[[object], object]:
    """Drop a suggestion too long to be valid, so the consent screen falls back as it does for any bad one."""
    return lambda value: None if isinstance(value, str) and len(value) > max_length else value


class AppAuthorizationParameters(BaseModel):
    """The public client's authorization URL parameters.

    Unrecognized parameters are ignored, as RFC 6749 section 3.1 requires.
    """

    client_id: OAuthClientId
    redirect_uri: OAuthRedirectUri
    state: OAuthState
    code_challenge: PKCECodeChallenge
    code_challenge_method: Literal["S256"]
    response_type: Literal["code"] = "code"
    limit: Annotated[str | None, BeforeValidator(_suggestion(32))] = Field(
        default=None,
        description="A spending limit in USD to suggest for the key. The person can change it on the consent "
        "screen, and one the screen cannot use is left blank.",
    )
    expiration: Annotated[str | None, BeforeValidator(_suggestion(16))] = Field(
        default=None,
        description="An expiry to suggest for the key: 1-hour, 1-day, 7-days, 30-days, 90-days, 180-days, "
        "1-year or never. The person can change it, and any other value leaves it at never.",
    )


class AppTokenRequest(BaseModel):
    """Unrecognized parameters, such as a public client's empty ``client_secret``, are ignored.

    An authorization code is redeemed with its callback and verifier; a refresh token, which
    only a client that registered itself holds, on its own.
    """

    grant_type: Literal["authorization_code", "refresh_token"]
    client_id: OAuthClientId
    redirect_uri: OAuthRedirectUri | None = None
    code: str | None = Field(default=None, min_length=43, max_length=128)
    code_verifier: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._~-]{43,128}$")
    refresh_token: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def require_grant_parameters(self) -> Self:
        if self.grant_type == "authorization_code":
            if self.redirect_uri is None or self.code is None or self.code_verifier is None:
                raise ValueError("An authorization code is redeemed with redirect_uri, code and code_verifier")
        elif self.refresh_token is None:
            raise ValueError("A refresh is made with refresh_token")
        return self


class AppDetails(BaseModel):
    """What the consent screen shows about an app before a person approves it."""

    name: str
    redirect_uri: str
    issuer: str
    scopes: list[KeyScope] = Field(
        description="The scopes each key this app is issued will hold, from its registration."
    )
    verified: bool = Field(
        description="Whether the operator registered this app. An unregistered app is named after "
        "its site, which nothing has checked beyond its callback being on it."
    )
    spend_limits: bool = Field(
        description="Whether a spending limit the person sets binds the key. It does not where the "
        "deployment has turned budget enforcement off, so the consent screen does not offer one."
    )
    credential: Literal["api_key", "access_token"] = Field(
        default="api_key",
        description="What the app is given: an API key billed to a workspace the person picks, or, for a "
        "client that registered itself such as an MCP client, the person's own read-only access token.",
    )


class AppAuthorizationResponse(BaseModel):
    redirect_uri: str


class AppTokenResponse(BaseModel):
    """A minted app key or access token, as RFC 6749 section 5.1 shapes it."""

    access_token: str
    token_type: Literal["Bearer"] = "Bearer"
    scope: str | None = Field(
        default=None,
        description="The scopes the key holds, space delimited. Fewer than the registration names "
        "where the person approved it before an operator narrowed it. Absent for an access token.",
    )
    expires_in: int | None = Field(default=None, description="Seconds until an access token expires, where it does.")
    refresh_token: str | None = Field(
        default=None,
        description="Renews an MCP client's access token at this endpoint, and is replaced on each use. Using one "
        "a second time ends the sign-in.",
    )
    manage_url: str = Field(description="The absolute URL of the page where the person sees this key, and revokes it.")


class AppRevocationRequest(BaseModel):
    token: str = Field(min_length=1, max_length=4096)
    token_type_hint: str | None = Field(default=None, max_length=128)


class ClientRegistrationRequest(BaseModel):
    """RFC 7591 client metadata. Only what Otari uses is read; anything else is ignored."""

    model_config = ConfigDict(extra="ignore")
    client_name: Annotated[str | None, BeforeValidator(_suggestion(MAX_DYNAMIC_CLIENT_NAME_LENGTH))] = Field(
        default=None, description="The name the consent screen shows, marked unverified."
    )
    redirect_uris: list[OAuthRedirectUri] = Field(min_length=1, max_length=10)

    @field_validator("redirect_uris")
    @classmethod
    def validate_redirects(cls, values: list[str]) -> list[str]:
        for value in values:
            if parse_oauth_redirect(value) is None:
                raise ValueError("Redirect URIs must be HTTPS (HTTP only on loopback), without query or fragment")
        return values


class ClientRegistrationResponse(BaseModel):
    """RFC 7591 section 3.2.1: the registration as recorded. Every client registered here is public."""

    client_id: str
    client_id_issued_at: int
    client_name: str
    redirect_uris: list[str]
    grant_types: list[str] = Field(default_factory=lambda: ["authorization_code", "refresh_token"])
    response_types: list[str] = Field(default_factory=lambda: ["code"])
    token_endpoint_auth_method: Literal["none"] = "none"
