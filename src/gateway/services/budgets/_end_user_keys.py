"""Keys a service key mints, each bound to one of its end users.

The key bills its end user whatever a request names, so the holder reaches that
end user's budget and, through the service key's own ceiling, nothing past it.
"""

from datetime import UTC, datetime, timedelta

from gateway.auth.models import hash_key, key_suffix
from gateway.core.config import GatewayConfig
from gateway.exceptions.budget_exceptions import (
    EndUserKeyLifetimeInvalidError,
    EndUserKeyModelsNotAllowedError,
    EndUserKeyNotFoundError,
    EndUserKeysNotAllowedError,
)
from gateway.models.api_keys import APIKey
from gateway.ports.api_key_format_port import ApiKeyFormatPort
from gateway.repositories.budgets import BudgetRepositories
from gateway.schemas.budgets import EndUserKeyCreate, EndUserKeyCreated, EndUserKeyPublic
from gateway.services.api_keys import ApiKeyService
from gateway.services.budgets._end_users import _EndUsers, require_service_key
from gateway.services.model_access import is_allowlist_subset, validate_allowed_models


class _EndUserKeys:
    """Mint, list and revoke the keys a service key mints for its end users."""

    def __init__(self, repositories: BudgetRepositories, end_users: _EndUsers, api_keys: ApiKeyService) -> None:
        self._repositories = repositories
        self._end_users = end_users
        self._api_keys = api_keys

    async def mint(
        self, parent: APIKey, request: EndUserKeyCreate, *, key_format: ApiKeyFormatPort, config: GatewayConfig
    ) -> EndUserKeyCreated:
        """Mint a key for the end user ``request.external_id`` names, creating the end user on first use."""
        require_service_key(parent)
        if parent.max_child_ttl_sec is None:
            raise EndUserKeysNotAllowedError()
        if request.expires_in_sec > parent.max_child_ttl_sec:
            raise EndUserKeyLifetimeInvalidError(parent.max_child_ttl_sec)
        try:
            allowed_models = validate_allowed_models(config, request.allowed_models)
        except ValueError as exc:
            raise EndUserKeyModelsNotAllowedError(str(exc)) from exc
        owner = await self._end_users.available_owner(parent)
        parent_models = parent.allowed_models if parent.allowed_models is not None else owner.allowed_models
        if not is_allowlist_subset(allowed_models, parent_models):
            raise EndUserKeyModelsNotAllowedError()

        resolved = await self._end_users.resolve(parent, request.external_id, request.budget_id)
        secret = key_format.mint()
        key = await self._api_keys.add_end_user_key(
            parent=parent,
            user_id=resolved.user_id,
            key_hash=hash_key(secret),
            key_prefix=key_format.fingerprint(secret),
            key_suffix=key_suffix(secret),
            key_name=request.key_name or request.external_id,
            expires_at=datetime.now(UTC) + timedelta(seconds=request.expires_in_sec),
            allowed_models=allowed_models,
            tags=request.tags,
        )
        end_user = await self._repositories.end_users.get(resolved.user_id)
        if end_user is None:
            raise RuntimeError("A minted key's end user is not there")
        public = EndUserKeyPublic.from_model(key, end_user)
        return EndUserKeyCreated(**public.model_dump(), key=secret, budget_id=resolved.budget_id)

    async def page(
        self, parent: APIKey, *, external_id: str | None, include_inactive: bool, skip: int, limit: int
    ) -> list[EndUserKeyPublic]:
        """A page of the keys ``parent`` minted, oldest first."""
        require_service_key(parent)
        user_id: str | None = None
        if external_id is not None:
            end_user = await self._repositories.end_users.find(str(parent.user_id), external_id)
            if end_user is None:
                return []
            user_id = end_user.user_id
        keys = await self._api_keys.list_end_user_keys(
            parent.id, user_id=user_id, include_inactive=include_inactive, skip=skip, limit=limit
        )
        users = await self._repositories.end_users.get_many({key.user_id for key in keys if key.user_id})
        return [EndUserKeyPublic.from_model(key, users[key.user_id]) for key in keys if key.user_id in users]

    async def revoke(self, parent: APIKey, key_id: str) -> EndUserKeyPublic:
        """Deactivate a key ``parent`` minted. Revoking one already revoked changes nothing."""
        require_service_key(parent)
        key = await self._api_keys.deactivate_end_user_key(parent.id, key_id)
        end_user = None if key is None or key.user_id is None else await self._repositories.end_users.get(key.user_id)
        if key is None or end_user is None:
            raise EndUserKeyNotFoundError(key_id)
        return EndUserKeyPublic.from_model(key, end_user)
