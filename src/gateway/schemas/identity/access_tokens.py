"""Request and response models for a person's own access tokens."""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from gateway.models.identity import MAX_ACCESS_TOKEN_NAME_LENGTH, PersonalAccessToken
from gateway.schemas.identity.oauth_apps import AppKeyExpiration


class AccessTokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_ACCESS_TOKEN_NAME_LENGTH, description="What the token is for.")
    expiration: AppKeyExpiration = Field(
        default="90-days", description="How long the token lives from now: 1-hour up to 1-year, or never."
    )


class AccessTokenPublic(BaseModel):
    id: uuid.UUID
    name: str
    token_suffix: str = Field(description="The token's last characters, to tell it apart from the account's others.")
    created_at: datetime
    last_used_at: datetime | None = Field(description="Accurate to about a minute. Null until first used.")
    expires_at: datetime | None = Field(description="Null when it does not expire.")

    @classmethod
    def of(cls, token: PersonalAccessToken) -> "AccessTokenPublic":
        return cls(
            id=token.id,
            name=token.name,
            token_suffix=token.token_suffix,
            created_at=token.created_at,
            last_used_at=token.last_used_at,
            expires_at=token.expires_at,
        )


class AccessTokenCreated(AccessTokenPublic):
    token: str = Field(description="The token itself. Shown once: only its hash is stored.")


class AccessTokensPublic(BaseModel):
    data: list[AccessTokenPublic]
