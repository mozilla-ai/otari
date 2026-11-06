"""Request and response models of the api-keys domain."""

import uuid
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gateway.models.api_keys import APIKeyFederationRule


class FederationRuleCreate(BaseModel):
    """An OpenID Connect identity that may authenticate as the key, in place of its secret.

    A token matches when it is signed by ``issuer``, names ``audience`` among its audiences, and carries each of
    ``subject`` and ``email`` that the rule sets; an email matches only when the token marks it verified. At least
    one of the two is required.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "issuer": "https://accounts.google.com",
                "audience": "https://otari.example.com",
                "email": "hackbot-api@my-project.iam.gserviceaccount.com",
            }
        },
    )

    issuer: str = Field(
        min_length=1,
        max_length=512,
        description=(
            "The token's 'iss'. On an operator's rule, one of the deployment's federation_issuers; on a member's own "
            "key, any https issuer, whose discovery document and signing keys are fetched only from a public host."
        ),
    )
    audience: str = Field(min_length=1, max_length=512, description="A value the token's 'aud' must contain.")
    subject: str | None = Field(
        default=None, min_length=1, max_length=255, description="The token's 'sub', when the rule names one."
    )
    email: str | None = Field(
        default=None,
        min_length=3,
        max_length=320,
        description="The token's verified 'email', when the rule names one. Compared without regard to case.",
    )

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        if "@" not in value:
            msg = "email must be an email address"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _names_a_principal(self) -> Self:
        if self.subject is None and self.email is None:
            msg = "a federation rule names a subject, an email, or both; an issuer and audience alone match anyone"
            raise ValueError(msg)
        return self


class FederationRulePublic(BaseModel):
    """A federation rule of an API key."""

    id: str
    api_key_id: str
    issuer: str
    audience: str
    subject: str | None
    email: str | None
    member_managed: bool = Field(
        description="Whether a member gave the rule to their own key: it then matches only in the token exchange."
    )
    created_at: datetime

    @classmethod
    def from_model(cls, row: APIKeyFederationRule) -> Self:
        """Map a stored rule."""
        return cls(
            id=row.id,
            api_key_id=row.api_key_id,
            issuer=row.issuer,
            audience=row.audience,
            subject=row.subject,
            email=row.email,
            member_managed=row.member_managed,
            created_at=row.created_at,
        )


class FederationTokenRequest(BaseModel):
    """An RFC 7523 ``jwt-bearer`` token request, as the Anthropic SDKs and Claude Code send it.

    Every field is optional here so that a missing one is answered in the RFC 6749 error shape rather than as a
    validation error. Fields the request does not use are ignored.
    """

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": "<identity token>",
                "federation_rule_id": "7d0c7a52-9a2e-4b55-8f3a-2f1f6c0d2a11",
                "organization_id": "5b3f8e0e-1c2d-4e7f-9a8b-0c1d2e3f4a5b",
            }
        },
    )

    grant_type: str | None = Field(default=None, description="urn:ietf:params:oauth:grant-type:jwt-bearer")
    assertion: str | None = Field(default=None, description="The identity token.")
    federation_rule_id: str | None = Field(
        default=None, description="The id of the federation rule the identity token must match."
    )
    organization_id: str | None = Field(
        default=None, description="When set, the UUID of the organization of the rule's key."
    )
    workspace_id: str | None = Field(
        default=None,
        description="Ignored. Accepted so that a client configured for Anthropic's workspace id is not refused.",
    )
    service_account_id: str | None = Field(
        default=None,
        description="Ignored. Accepted so that a client configured for Anthropic's service account id is not refused.",
    )


class FederationTokenResponse(BaseModel):
    """An access token, as RFC 6749 section 5.1 shapes it. Present it as a bearer token in place of the key."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int = Field(description="Seconds until the access token expires.")


class OAuthErrorResponse(BaseModel):
    """A refused token request, as RFC 6749 section 5.2 shapes it."""

    error: str = Field(description="invalid_request, unsupported_grant_type, invalid_grant or temporarily_unavailable")
    error_description: str


class AppAuthorizationParameters(BaseModel):
    """Parameters shared by the authorization URL and explicit dashboard approval.

    Unrecognized parameters are ignored, as RFC 6749 section 3.1 requires.
    """

    client_id: str = Field(min_length=1, max_length=128)
    redirect_uri: str = Field(max_length=2048)
    state: str = Field(min_length=16, max_length=256, pattern=r"^[A-Za-z0-9._~+/=-]+$")
    code_challenge: str = Field(pattern=r"^[A-Za-z0-9_-]{43}$")
    code_challenge_method: Literal["S256"]
    response_type: Literal["code"] = "code"


class AppAuthorizationRequest(AppAuthorizationParameters):
    model_config = ConfigDict(extra="forbid")
    workspace_id: uuid.UUID


class AppTokenRequest(BaseModel):
    """Unrecognized parameters, such as a public client's empty ``client_secret``, are ignored."""

    grant_type: Literal["authorization_code"]
    client_id: str = Field(min_length=1, max_length=128)
    redirect_uri: str = Field(max_length=2048)
    code: str = Field(min_length=43, max_length=128)
    code_verifier: str = Field(pattern=r"^[A-Za-z0-9._~-]{43,128}$")


class AppDetails(BaseModel):
    name: str
    redirect_uri: str


class AppAuthorizationResponse(BaseModel):
    redirect_uri: str


class AppTokenResponse(BaseModel):
    access_token: str
    token_type: Literal["Bearer"] = "Bearer"
