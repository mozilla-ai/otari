"""What the api-keys domain answers about a key."""

from pydantic import BaseModel


class KeyIdentity(BaseModel):
    """The key a caller presented, who owns it, and the tenancy it belongs to, every id as a string.

    `user_id` is null for a key with no owner, which the schema allows although every mint path attaches one.
    `organization_id` is null only for a workspace that resolves to no organization, a state no write path
    produces; it is reported rather than refused, because the key itself is live.
    """

    api_key_id: str
    user_id: str | None
    workspace_id: str
    organization_id: str | None
