"""Request and response models of the api-keys domain."""

from datetime import datetime
from typing import Self

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
        min_length=1, max_length=512, description="The token's 'iss', one of the deployment's federation_issuers."
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
            created_at=row.created_at,
        )
