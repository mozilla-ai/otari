"""Apply the API keys config.yml declares."""

import uuid
from collections.abc import Callable, Collection
from typing import Any

from gateway.auth.models import hash_key, key_suffix
from gateway.core.settings.api_keys import ApiKeyConfig
from gateway.exceptions.api_keys_exceptions import DeclaredKeySecretInUseError
from gateway.repositories.api_keys import ApiKeyRepository


async def apply_declared_key(
    keys: ApiKeyRepository,
    config_name: str,
    spec: ApiKeyConfig,
    *,
    declared_names: Collection[str],
    end_user_budget_ids: list[str] | None,
    workspace_id: uuid.UUID,
    fingerprint: Callable[[str], str],
) -> str:
    """Create the key ``config_name`` or write the declared values over it, and return its id.

    The key is found by its config name, or else adopted by its secret, which is
    how a key minted through the API comes under config.yml with its id, its end
    users and its ceiling intact. A new key goes in the deployment's default
    workspace; an existing one stays in its own.
    """
    secret = spec.secret.get_secret_value()
    key_hash = hash_key(secret)
    owner = await keys.owner_user_id(spec.user_id)

    key = await keys.get_by_config_name(config_name)
    if key is None:
        holder = await keys.get_by_hash(key_hash)
        if holder is not None and holder.config_name is not None and holder.config_name in declared_names:
            raise DeclaredKeySecretInUseError(config_name)
        if holder is None and await keys.add_declared_if_absent(
            config_name=config_name,
            workspace_id=workspace_id,
            user_id=owner,
            key_hash=key_hash,
            key_prefix=fingerprint(secret),
            key_suffix=key_suffix(secret),
        ):
            key = await keys.get_by_config_name(config_name)
        else:
            # Adopted, or inserted by another replica first: either way the row is there now.
            key = await keys.get_by_config_name(config_name) or await keys.get_by_hash(key_hash)
        if key is None:
            raise RuntimeError("A declared key's insert conflicted with a row that is not there")

    changes: dict[str, Any] = {
        "config_name": config_name,
        "key_name": spec.key_name or config_name,
        "user_id": owner,
        "is_active": True,
        "is_service_key": spec.is_service_key,
        "end_user_budget_id": spec.end_user_budget_id,
        "end_user_budget_ids": end_user_budget_ids,
        "exclude_from_budget": spec.exclude_from_budget,
        "reject_user_mismatch": spec.reject_user_mismatch,
    }
    if key.key_hash != key_hash:
        holder = await keys.get_by_hash(key_hash)
        if holder is not None and holder.id != key.id:
            raise DeclaredKeySecretInUseError(config_name)
        changes |= {
            "key_hash": key_hash,
            "key_prefix": fingerprint(secret),
            "key_suffix": key_suffix(secret),
            "last_used_at": None,
        }
    key = await keys.update(key, changes)
    return key.id
