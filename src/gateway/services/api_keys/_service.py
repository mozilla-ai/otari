import uuid
from collections.abc import Callable, Collection, Sequence
from datetime import UTC, datetime, timedelta

from gateway.core.settings.api_keys import ApiKeyConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.api_keys_exceptions import (
    EndUserTokensDisabledError,
    SigningKeyLimitError,
    SigningKeyNotFoundError,
    SigningKeysNotAllowedError,
)
from gateway.models.api_keys import MAX_SIGNING_KEYS_PER_KEY, APIKey, EndUserSigningKey, EndUserTokenRedemption
from gateway.repositories.api_keys import ApiKeyRepository, SigningKeyRepository
from gateway.services.api_keys._declared import apply_declared_key
from gateway.services.api_keys._signed_tokens import normalize_public_key


class ApiKeyService:
    """Answer questions about API keys and the workspaces that hold them.

    Each method runs in the caller's Unit of Work block and raises ``OutsideUnitOfWorkError`` outside one,
    except the sweep, which is a worker job's whole step and opens its own, and the signing-key methods a route
    calls, which are business steps of their own.
    """

    def __init__(self, keys: ApiKeyRepository, uow: UnitOfWork, signing_keys: SigningKeyRepository) -> None:
        self._keys = keys
        self._uow = uow
        self._signing_keys = signing_keys

    async def get_workspace_id_for_key(self, key_id: str) -> uuid.UUID | None:
        """Return the ID of the workspace that owns a key, or None when no key has that ID."""
        return await self._keys.get_workspace_id_for_key(key_id)

    async def get_key_ids_in_workspaces(self, workspace_ids: Sequence[uuid.UUID]) -> list[str]:
        """Return the IDs of the keys in these workspaces, in no particular order."""
        return await self._keys.get_key_ids_in_workspaces(workspace_ids)

    async def keys_defaulting_end_users_to(self, budget_id: str) -> list[str]:
        """Name the keys that start a new end user on this budget when a request names none."""
        return await self._keys.names_defaulting_end_users_to(budget_id)

    async def forget_end_user_budget(self, budget_id: str) -> None:
        """Take a budget off every key's list of end-user budgets."""
        await self._keys.remove_end_user_budget(budget_id)

    async def apply_declared_key(
        self,
        config_name: str,
        spec: ApiKeyConfig,
        *,
        owner: str,
        declared_names: Collection[str],
        end_user_budget_ids: list[str] | None,
        workspace_id: uuid.UUID,
        fingerprint: Callable[[str], str],
    ) -> str:
        """Create the key config.yml declares under ``config_name``, or write the declared values over it.

        Returns the key's id. ``owner`` is the id of the user it bills to; ``end_user_budget_ids`` is the declared list
        once checked; ``fingerprint`` is the bound key format's, and a new key lands in ``workspace_id``.
        """
        return await apply_declared_key(
            self._keys,
            config_name,
            spec,
            owner=owner,
            declared_names=declared_names,
            end_user_budget_ids=end_user_budget_ids,
            workspace_id=workspace_id,
            fingerprint=fingerprint,
        )

    async def release_undeclared_keys(self, declared_names: Collection[str]) -> list[str]:
        """Stop marking as declared every key config.yml no longer declares, and return their former names.

        The keys themselves stay, with their secrets and settings.
        """
        return await self._keys.release_config_names(declared_names)

    async def add_end_user_key(
        self,
        *,
        parent: APIKey,
        user_id: str,
        key_hash: str,
        key_prefix: str,
        key_suffix: str,
        key_name: str | None,
        expires_at: datetime,
        allowed_models: list[str] | None,
        tags: dict[str, str] | None,
    ) -> APIKey:
        """Stage a key ``parent`` mints for its end user ``user_id``, in ``parent``'s workspace."""
        return await self._keys.add_child(
            parent=parent,
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            key_suffix=key_suffix,
            key_name=key_name,
            expires_at=expires_at,
            allowed_models=allowed_models,
            tags=tags,
        )

    async def list_end_user_keys(
        self,
        parent_id: str,
        *,
        user_id: str | None,
        include_inactive: bool,
        skip: int,
        limit: int,
        jti: str | None = None,
    ) -> list[APIKey]:
        """Return a page of the keys ``parent_id`` minted, oldest first; unless asked, only those that still work.

        ``jti`` narrows the page to the key a signed token with that id was materialized as.
        """
        return await self._keys.list_children(
            parent_id,
            user_id=user_id,
            active_at=None if include_inactive else datetime.now(UTC),
            skip=skip,
            limit=limit,
            jti=jti,
        )

    async def deactivate_end_user_key(self, parent_id: str, key_id: str) -> APIKey | None:
        """Deactivate the key ``parent_id`` minted under ``key_id`` and return it, or None when there is none."""
        return await self._keys.deactivate_child(parent_id, key_id)

    async def is_end_user_key(self, key_id: str) -> bool:
        """Whether the key with this id was minted by a service key for one of its end users."""
        return await self._keys.is_child(key_id)

    async def sweep_expired_end_user_keys(self, retention_sec: int) -> int:
        """Delete the minted keys that expired more than ``retention_sec`` ago, and say how many."""
        async with self._uow:
            return await self._keys.delete_children_expired_before(datetime.now(UTC) - timedelta(seconds=retention_sec))

    async def register_signing_key(
        self, api_key: APIKey, *, public_key: str, name: str | None, enabled: bool
    ) -> EndUserSigningKey:
        """Register a public key ``api_key``'s control plane signs end-user tokens with, under a new kid.

        ``enabled`` is the deployment's ``end_user_tokens_enabled``. Only a key that may mint keys for its end users
        registers one, and at most ``MAX_SIGNING_KEYS_PER_KEY`` at once.
        """
        if not enabled:
            raise EndUserTokensDisabledError()
        if not api_key.is_service_key or api_key.max_child_ttl_sec is None:
            raise SigningKeysNotAllowedError()
        normalized = normalize_public_key(public_key)
        async with self._uow:
            # Serializes concurrent registrations, which would otherwise each count the others out.
            await self._keys.lock(api_key.id)
            if await self._signing_keys.count_for(api_key.id) >= MAX_SIGNING_KEYS_PER_KEY:
                raise SigningKeyLimitError(MAX_SIGNING_KEYS_PER_KEY)
            return await self._signing_keys.add(api_key_id=api_key.id, public_key=normalized, name=name)

    async def list_signing_keys(self, api_key: APIKey) -> list[EndUserSigningKey]:
        """The signing keys ``api_key`` holds, oldest first."""
        async with self._uow:
            return await self._signing_keys.list_for(api_key.id)

    async def delete_signing_key(self, api_key: APIKey, kid: str) -> None:
        """Delete a signing key ``api_key`` holds, refusing every token it signed from now on.

        The keys its tokens were materialized as are deactivated with it, so listings show them revoked.
        """
        async with self._uow:
            if not await self._signing_keys.delete_for(api_key.id, kid):
                raise SigningKeyNotFoundError(kid)
            await self._keys.deactivate_token_children(api_key.id, kid)

    async def find_signing_key(self, kid: str) -> tuple[EndUserSigningKey, APIKey] | None:
        """The signing key with this kid and the key that holds it, with that key's user, or None."""
        signing_key = await self._signing_keys.find(kid)
        if signing_key is None:
            return None
        holder = await self._keys.get_with_owner(signing_key.api_key_id)
        return None if holder is None else (signing_key, holder)

    async def hold_signing_key(self, kid: str) -> bool:
        """Hold the signing key with this kid until the caller's block ends, so it cannot be deleted mid-way.

        Says whether it is still there.
        """
        return await self._signing_keys.hold(kid)

    async def find_token_redemption(
        self, parent_id: str, jti: str
    ) -> tuple[EndUserTokenRedemption, APIKey | None] | None:
        """The first use of the token ``parent_id`` signed under ``jti``, and the key it is, if that key remains."""
        return await self._keys.find_redemption(parent_id, jti)

    async def add_token_end_user_key(
        self,
        *,
        parent: APIKey,
        user_id: str,
        key_hash: str,
        key_prefix: str,
        key_name: str,
        expires_at: datetime,
        allowed_models: list[str] | None,
        tags: dict[str, str] | None,
        jti: str,
        signing_key_id: str,
        claims_digest: str,
    ) -> APIKey | None:
        """Stage the key a signed token is materialized as, or None when a concurrent first use did it first."""
        return await self._keys.add_token_child(
            parent=parent,
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            key_name=key_name,
            expires_at=expires_at,
            allowed_models=allowed_models,
            tags=tags,
            jti=jti,
            signing_key_id=signing_key_id,
            claims_digest=claims_digest,
        )
