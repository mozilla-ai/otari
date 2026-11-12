"""RFC 6749 public-client authorization, token and revocation contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

OAuthClientId = Annotated[str, Field(min_length=1, max_length=128)]
OAuthRedirectUri = Annotated[str, Field(max_length=2048)]
OAuthState = Annotated[str, Field(min_length=16, max_length=256, pattern=r"^[A-Za-z0-9._~+/=-]+$")]
PKCECodeChallenge = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{43}$")]


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


class AppTokenRequest(BaseModel):
    """Unrecognized parameters, such as a public client's empty ``client_secret``, are ignored."""

    grant_type: Literal["authorization_code"]
    client_id: OAuthClientId
    redirect_uri: OAuthRedirectUri
    code: str = Field(min_length=43, max_length=128)
    code_verifier: str = Field(pattern=r"^[A-Za-z0-9._~-]{43,128}$")


class AppDetails(BaseModel):
    name: str
    redirect_uri: str
    issuer: str


class AppAuthorizationResponse(BaseModel):
    redirect_uri: str


class AppTokenResponse(BaseModel):
    access_token: str
    token_type: Literal["Bearer"] = "Bearer"


class AppRevocationRequest(BaseModel):
    token: str = Field(min_length=1, max_length=4096)
    token_type_hint: str | None = Field(default=None, max_length=128)
