import uuid
from collections.abc import Callable, Collection, Sequence

from gateway.core.settings.api_keys import ApiKeyConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.api_keys_exceptions import (
    FederationIssuerNotAllowedError,
    FederationRuleExistsError,
    FederationRuleLimitError,
    FederationRuleNotFoundError,
)
from gateway.models.api_keys import MAX_FEDERATION_RULES_PER_KEY
from gateway.repositories.api_keys import ApiKeyRepositories
from gateway.schemas.api_keys import FederationRuleCreate, FederationRulePublic
from gateway.services.api_keys._declared import apply_declared_key


class ApiKeyService:
    """Answer questions about API keys and the workspaces that hold them, and manage their federation rules.

    The lookups run in the caller's Unit of Work block and raise ``OutsideUnitOfWorkError`` outside one; the
    federation-rule methods are business steps of their own.
    """

    def __init__(self, uow: UnitOfWork, repositories: ApiKeyRepositories) -> None:
        self._uow = uow
        self._keys = repositories.keys
        self._federation_rules = repositories.federation_rules

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

    async def add_federation_rule(
        self, api_key_id: str, rule: FederationRuleCreate, *, allowed_issuers: Collection[str]
    ) -> FederationRulePublic:
        """Let tokens matching ``rule`` authenticate as the key.

        The issuer must be one of ``allowed_issuers``. A rule naming the identity of an existing rule is refused, and
        so is one past the per-key limit: the same identity on this key, or on another key a subject or an email in
        common and no contradicting value, since a token matching rules of two keys authenticates as neither.
        """
        if rule.issuer not in allowed_issuers:
            raise FederationIssuerNotAllowedError(rule.issuer)
        async with self._uow:
            if await self._federation_rules.count_for_key(api_key_id) >= MAX_FEDERATION_RULES_PER_KEY:
                raise FederationRuleLimitError(MAX_FEDERATION_RULES_PER_KEY)
            existing = await self._federation_rules.find_conflicting(
                api_key_id=api_key_id,
                issuer=rule.issuer,
                audience=rule.audience,
                subject=rule.subject,
                email=rule.email,
            )
            if existing is not None:
                raise FederationRuleExistsError
            row = await self._federation_rules.add(
                rule_id=str(uuid.uuid4()),
                api_key_id=api_key_id,
                issuer=rule.issuer,
                audience=rule.audience,
                subject=rule.subject,
                email=rule.email,
            )
            return FederationRulePublic.from_model(row)

    async def list_federation_rules(self, api_key_id: str) -> list[FederationRulePublic]:
        """Return the key's federation rules, oldest first."""
        async with self._uow:
            rows = await self._federation_rules.list_for_key(api_key_id)
            return [FederationRulePublic.from_model(row) for row in rows]

    async def delete_federation_rule(self, api_key_id: str, rule_id: str) -> None:
        """Remove one of the key's federation rules; tokens it matched stop authenticating at once."""
        async with self._uow:
            row = await self._federation_rules.get_for_key(api_key_id, rule_id)
            if row is None:
                raise FederationRuleNotFoundError(rule_id)
            await self._federation_rules.delete(row)
