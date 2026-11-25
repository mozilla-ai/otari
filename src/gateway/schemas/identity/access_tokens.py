"""Request and response models for a person's own access tokens."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from gateway.models.identity import MAX_ACCESS_TOKEN_NAME_LENGTH, PersonalAccessToken

AccessTokenExpiration = Literal["1-hour", "1-day", "7-days", "30-days", "90-days", "180-days", "1-year"]


class AccessTokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_ACCESS_TOKEN_NAME_LENGTH, description="What the token is for.")
    expiration: AccessTokenExpiration = Field(
        default="90-days", description="How long the token lives from now, at most a year."
    )


class AccessTokenPublic(BaseModel):
    id: uuid.UUID
    name: str
    token_suffix: str = Field(description="The token's last characters, to tell it apart from the account's others.")
    created_at: datetime
    last_used_at: datetime | None = Field(description="Accurate to about a minute. Null until first used.")
    expires_at: datetime | None = Field(
        description="When it ends. For an MCP client's sign-in, when renewing its access token stops working."
    )
    organization_id: uuid.UUID | None = Field(
        description="The organization it reads, whichever one the dashboard has switched to since."
    )

    @classmethod
    def of(cls, token: PersonalAccessToken) -> "AccessTokenPublic":
        return cls(
            id=token.id,
            name=token.name,
            token_suffix=token.token_suffix,
            created_at=token.created_at,
            last_used_at=token.last_used_at,
            expires_at=token.expires_at,
            organization_id=token.organization_id,
        )


class AccessTokenCreated(AccessTokenPublic):
    token: str = Field(description="The token itself. Shown once: only its hash is stored.")


class AccessTokensPublic(BaseModel):
    data: list[AccessTokenPublic]
