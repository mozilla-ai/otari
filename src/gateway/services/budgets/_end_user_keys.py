"""Keys a service key mints, each bound to one of its end users.

The key bills its end user whatever a request names, so the holder reaches that
end user's budget and, through the service key's own ceiling, nothing past it.
"""

import secrets
import time
from datetime import UTC, datetime, timedelta

from gateway.auth.models import hash_key, key_suffix
from gateway.core.config import GatewayConfig
from gateway.exceptions import TenancyError
from gateway.exceptions.api_keys_exceptions import EndUserTokenRefusedError
from gateway.exceptions.budget_exceptions import (
    EndUserBudgetNotAllowedError,
    EndUserBudgetNotFoundError,
    EndUserIdInvalidError,
    EndUserKeyLifetimeInvalidError,
    EndUserKeyModelsNotAllowedError,
    EndUserKeyNotFoundError,
    EndUserKeysNotAllowedError,
    EndUserOwnerUnavailableError,
    NotAServiceKeyError,
)
from gateway.log_config import logger
from gateway.models.api_keys import APIKey, EndUserTokenRedemption
from gateway.ports.api_key_format_port import ApiKeyFormatPort
from gateway.repositories.budgets import BudgetRepositories
from gateway.schemas.budgets import EndUserKeyCreate, EndUserKeyCreated, EndUserKeyPublic, EndUserTokenClaims
from gateway.services.api_keys import (
    END_USER_TOKEN_PREFIX,
    ApiKeyService,
    claims_digest,
    read_token_header,
    verify_token_claims,
)
from gateway.services.budgets._end_users import _EndUsers, require_service_key
from gateway.services.model_access import is_allowlist_subset, validate_allowed_models

# The refusal each mint bound gives a token that breaks it.
_TOKEN_REFUSALS: tuple[tuple[type[TenancyError], str], ...] = (
    (EndUserBudgetNotAllowedError, "end_user_token_budget"),
    (EndUserBudgetNotFoundError, "end_user_token_budget"),
    (EndUserKeyModelsNotAllowedError, "end_user_token_models"),
    (EndUserIdInvalidError, "end_user_token_claims"),
    (EndUserOwnerUnavailableError, "end_user_token_owner_unavailable"),
    (NotAServiceKeyError, "end_user_token_not_allowed"),
)


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
        allowed_models = await self._bounded_models(parent, request.allowed_models, config)

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
        self,
        parent: APIKey,
        *,
        external_id: str | None,
        include_inactive: bool,
        skip: int,
        limit: int,
        jti: str | None = None,
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
            parent.id, user_id=user_id, include_inactive=include_inactive, skip=skip, limit=limit, jti=jti
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

    async def _bounded_models(
        self, parent: APIKey, requested: list[str] | None, config: GatewayConfig
    ) -> list[str] | None:
        """``requested`` once checked against the deployment and found within ``parent``'s own model access."""
        try:
            allowed_models = validate_allowed_models(config, requested)
        except ValueError as exc:
            raise EndUserKeyModelsNotAllowedError(str(exc)) from exc
        owner = await self._end_users.available_owner(parent)
        parent_models = parent.allowed_models if parent.allowed_models is not None else owner.allowed_models
        if not is_allowlist_subset(allowed_models, parent_models):
            raise EndUserKeyModelsNotAllowedError()
        return allowed_models

    async def authenticate_token(self, credential: str, config: GatewayConfig) -> APIKey:
        """The key a signed end-user token authenticates as, materialized on its first use.

        The token is verified against the signing key its kid names, and its claims are held to the bounds the mint
        route holds a request to. Its first use creates the end user, when it is new, and the key; every later use
        must carry the same claims, signed by the same key, and returns that key. Every refusal is an
        ``EndUserTokenRefusedError`` naming its reason.
        """
        if not config.end_user_tokens_enabled:
            raise EndUserTokenRefusedError("end_user_token_disabled")
        header = read_token_header(credential)
        found = await self._api_keys.find_signing_key(header.kid)
        if found is None:
            raise EndUserTokenRefusedError("end_user_token_unknown_kid")
        signing_key, parent = found
        claims = verify_token_claims(
            header, public_key_pem=signing_key.public_key, parent_key_id=parent.id, now=time.time()
        )
        # Rechecked on every use, so clearing max_child_ttl_sec stops every token at once.
        if not parent.is_service_key or parent.max_child_ttl_sec is None:
            raise EndUserTokenRefusedError("end_user_token_not_allowed")
        # Also checked on the key the token becomes; here, so that a dead service key materializes nothing.
        if not parent.is_active or _expired(parent.expires_at):
            raise EndUserTokenRefusedError("end_user_token_issuer_unusable")
        if claims.exp - claims.iat > parent.max_child_ttl_sec:
            raise EndUserTokenRefusedError("end_user_token_lifetime")
        digest = claims_digest(claims)

        redeemed = await self._api_keys.find_token_redemption(parent.id, claims.jti)
        if redeemed is None:
            # A delete of the signing key racing this first use would otherwise miss the key it creates.
            if not await self._api_keys.hold_signing_key(signing_key.id):
                raise EndUserTokenRefusedError("end_user_token_unknown_kid")
            child = await self._materialize(parent, claims, signing_key.id, digest, config)
            if child is not None:
                logger.info(
                    "End-user token signed with kid %s for service key %s materialized as key %s",
                    signing_key.id,
                    parent.id,
                    child.id,
                )
                return child
            redeemed = await self._api_keys.find_token_redemption(parent.id, claims.jti)
            if redeemed is None:
                raise EndUserTokenRefusedError("end_user_token_race")
        return _redeemed_key(*redeemed, signing_key_id=signing_key.id, digest=digest)

    async def _materialize(
        self, parent: APIKey, claims: EndUserTokenClaims, signing_key_id: str, digest: str, config: GatewayConfig
    ) -> APIKey | None:
        """Create the key a token's first use stands for, or None when a concurrent first use created it."""
        try:
            allowed_models = await self._bounded_models(parent, claims.allowed_models, config)
            resolved = await self._end_users.resolve(parent, claims.sub, claims.budget_id)
        except TenancyError as exc:
            for error_class, reason in _TOKEN_REFUSALS:
                if isinstance(exc, error_class):
                    raise EndUserTokenRefusedError(reason) from exc
            raise
        return await self._api_keys.add_token_end_user_key(
            parent=parent,
            user_id=resolved.user_id,
            # Nobody is given this secret: the key is reached through its token alone.
            key_hash=hash_key(secrets.token_urlsafe(48)),
            key_prefix=END_USER_TOKEN_PREFIX,
            key_name=claims.sub,
            expires_at=datetime.fromtimestamp(claims.exp, UTC),
            allowed_models=allowed_models,
            tags=claims.tags,
            jti=claims.jti,
            signing_key_id=signing_key_id,
            claims_digest=digest,
        )


def _expired(expires_at: datetime | None) -> bool:
    # SQLite reads the column back naive, and it is always written in UTC.
    if expires_at is None:
        return False
    return (expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=UTC)) <= datetime.now(UTC)


def _redeemed_key(
    redemption: EndUserTokenRedemption, child: APIKey | None, *, signing_key_id: str, digest: str
) -> APIKey:
    """The key an already redeemed token is, refusing a token that is not the one first redeemed under its jti."""
    if redemption.signing_key_id != signing_key_id or redemption.claims_digest != digest:
        raise EndUserTokenRefusedError("end_user_token_claims_mismatch")
    if child is None:
        raise EndUserTokenRefusedError("end_user_token_key_deleted")
    return child
