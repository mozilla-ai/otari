"""The token exchange: a verified identity token, naming one federation rule, for a short-lived access token."""

import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from http import HTTPStatus

from gateway.core.database import DATABASE_ERRORS
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.repositories.api_keys import ApiKeyRepositories
from gateway.services.api_keys._access_tokens import (
    JWT_BEARER_GRANT,
    AccessTokenClaims,
    FederationGrantError,
    access_token_expiry,
    grant_names_key,
    mint_access_token,
)
from gateway.services.api_keys._admission import key_refusal
from gateway.services.api_keys._federation import (
    FederatedTokenVerifier,
    FederationRefusedError,
    FederationUnavailableError,
    rule_matches,
    unverified_issuer,
)

OrganizationOfWorkspace = Callable[[uuid.UUID], Awaitable[uuid.UUID | None]]

_INVALID_GRANT = "The identity token does not match the federation rule."


def _invalid_grant(reason: str) -> FederationGrantError:
    return FederationGrantError("invalid_grant", _INVALID_GRANT, status_code=HTTPStatus.UNAUTHORIZED, reason=reason)


def _unavailable(reason: str) -> FederationGrantError:
    return FederationGrantError(
        "temporarily_unavailable",
        "The token exchange is temporarily unavailable, please retry.",
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        reason=reason,
    )


class FederationTokenExchange:
    """Exchange a verified identity token, naming one federation rule, for a short-lived access token.

    The RFC 7523 ``jwt-bearer`` grant. The identity token must verify and match the named rule, the rule's key must
    be active and unexpired, and any organization, workspace or service account the request names must be that
    key's own. Its one Unit of Work block only reads, and a refusal raised inside it rolls it back, so an exchange,
    refused or not, writes nothing.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: ApiKeyRepositories,
        *,
        verifier: FederatedTokenVerifier,
        organization_of_workspace: OrganizationOfWorkspace,
        ttl_seconds: int,
        secrets: Sequence[str],
    ) -> None:
        self._uow = uow
        self._keys = repositories.keys
        self._rules = repositories.federation_rules
        self._verifier = verifier
        self._organization_of_workspace = organization_of_workspace
        self._ttl_seconds = ttl_seconds
        self._secrets = secrets

    async def exchange(
        self,
        *,
        grant_type: str | None,
        assertion: str | None,
        federation_rule_id: str | None,
        organization_id: str | None,
        workspace_id: str | None,
        service_account_id: str | None,
    ) -> tuple[str, int]:
        """Return the access token and its lifetime in seconds.

        Raises:
            FederationGrantError: the request is refused.
        """
        if grant_type != JWT_BEARER_GRANT:
            raise FederationGrantError(
                "unsupported_grant_type",
                f"Only the {JWT_BEARER_GRANT} grant is supported.",
                status_code=HTTPStatus.BAD_REQUEST,
                reason="federation_grant_unsupported",
            )
        if not assertion or not federation_rule_id:
            raise FederationGrantError(
                "invalid_request",
                "The request needs an assertion and a federation_rule_id.",
                status_code=HTTPStatus.BAD_REQUEST,
                reason="federation_grant_invalid",
            )
        if not self._secrets:
            raise _unavailable("federation_no_secret_key")
        if unverified_issuer(assertion) is None:
            raise _invalid_grant("federation_invalid_token")
        try:
            identity = await self._verifier.verify(assertion)
            async with self._uow:
                rule = await self._rules.get(federation_rule_id)
                if rule is None or not rule_matches(rule, identity):
                    raise FederationRefusedError("federation_no_rule")
                api_key = await self._keys.get(rule.api_key_id)
                if api_key is None:
                    raise FederationRefusedError("federation_no_rule")
                key_organization_id = await self._organization_of_workspace(api_key.workspace_id)
                if key_organization_id is None or not grant_names_key(
                    organization_id=organization_id,
                    workspace_id=workspace_id,
                    service_account_id=service_account_id,
                    key_organization_id=key_organization_id,
                    key_workspace_id=api_key.workspace_id,
                    api_key_id=api_key.id,
                ):
                    raise FederationRefusedError("federation_grant_mismatch")
                refusal = key_refusal(api_key, datetime.now(UTC))
                if refusal is not None:
                    raise FederationRefusedError(refusal[0])
                rule_id, api_key_id = rule.id, api_key.id
        except FederationUnavailableError as exc:
            raise _unavailable("federation_unavailable") from exc
        except FederationRefusedError as exc:
            raise _invalid_grant(exc.reason) from exc
        except DATABASE_ERRORS as exc:
            raise _unavailable("db_error") from exc
        now = time.time()
        expires_at = access_token_expiry(
            now=now, ttl_seconds=self._ttl_seconds, identity_expires_at=identity.expires_at
        )
        if expires_at - now < 1:
            raise _invalid_grant("federation_expired")
        token = mint_access_token(
            AccessTokenClaims(rule_id=rule_id, api_key_id=api_key_id, expires_at=expires_at), self._secrets
        )
        logger.debug("Exchanged an identity token from %s for rule %s", identity.issuer, rule_id)
        return token, int(expires_at - now)
