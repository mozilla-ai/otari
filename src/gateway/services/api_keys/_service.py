import uuid
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any

from gateway.core.config import GatewayConfig
from gateway.core.settings.api_keys import ApiKeyConfig, https_url_problem
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.api_keys_exceptions import (
    FederationIssuerInvalidError,
    FederationIssuerNotAllowedError,
    FederationRuleExistsError,
    FederationRuleLimitError,
    FederationRuleNotFoundError,
    OperatorDisabledKeyError,
)
from gateway.log_config import logger
from gateway.models.api_keys import MAX_FEDERATION_RULES_PER_KEY, APIKey
from gateway.repositories.api_keys import ApiKeyRepositories
from gateway.schemas.api_keys import FederationRuleCreate, FederationRulePublic
from gateway.services.api_keys._declared import apply_declared_key
from gateway.services.model_access import (
    KEY_MODELS_EXCEED_USER_DETAIL,
    KEY_TOOLS_EXCEED_USER_DETAIL,
    effective_tool_allowlist,
    is_allowlist_subset,
    is_model_allowed,
    is_tool_allowed,
    tool_not_allowed_detail,
    tools_within_user,
    validate_allowed_models,
    validate_allowed_tools,
)
from gateway.services.organizations import AccessLists, UserAccess


class ApiKeyService:
    """Answer questions about API keys and their workspaces, and manage their federation rules and access lists.

    The lookups run in the caller's Unit of Work block and raise ``OutsideUnitOfWorkError`` outside one; the
    federation-rule and access-list methods are business steps of their own. A key's owner's lists come from
    ``users``.
    """

    def __init__(self, uow: UnitOfWork, repositories: ApiKeyRepositories, *, users: UserAccess) -> None:
        self._uow = uow
        self._keys = repositories.keys
        self._federation_rules = repositories.federation_rules
        self._users = users

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
    ) -> tuple[str, bool]:
        """Create the key config.yml declares under ``config_name``, or write the declared values over it.

        Returns the key's id, and whether anything was written: a key config.yml
        already wrote this same entry over is left as it is. ``owner`` is the id
        of the user it bills to; ``end_user_budget_ids`` is the declared list once
        checked; ``fingerprint`` is the bound key format's, and a new key lands in
        ``workspace_id``.
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
        self,
        api_key_id: str,
        rule: FederationRuleCreate,
        *,
        allowed_issuers: Collection[str],
        member_managed: bool = False,
    ) -> FederationRulePublic:
        """Let tokens matching ``rule`` authenticate as the key.

        An operator's rule must name one of ``allowed_issuers``, and is refused when it names the identity of an
        existing operator's rule: the same identity on this key, or on another key a subject or an email in common
        and no contradicting value, since a token matching rules of two keys authenticates as neither. Members' rules
        take no part in that check: they never match a bare identity token, so they cannot make one ambiguous, and
        leaving them out keeps a member from blocking the operator's rule.

        A member's rule (``member_managed``) may name any https issuer; that its host is public is checked when its
        keys are fetched, as for every issuer. It is refused only when this key already holds it: it matches only in
        the exchange that names it, so another key's rule for the same identity takes nothing from it, and searching
        further would make a 409 reveal another tenant's identity. Either is refused past the per-key limit.
        """
        if member_managed:
            problem = https_url_problem(rule.issuer)
            if problem is not None:
                raise FederationIssuerInvalidError(rule.issuer, problem)
        elif rule.issuer not in allowed_issuers:
            raise FederationIssuerNotAllowedError(rule.issuer)
        async with self._uow:
            if await self._federation_rules.count_for_key(api_key_id) >= MAX_FEDERATION_RULES_PER_KEY:
                raise FederationRuleLimitError(MAX_FEDERATION_RULES_PER_KEY)
            find = (
                self._federation_rules.find_member_rule if member_managed else self._federation_rules.find_conflicting
            )
            existing = await find(
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
                member_managed=member_managed,
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

    async def update_owned(
        self, config: GatewayConfig, key: APIKey, changes: Mapping[str, Any], *, owner_id: str
    ) -> None:
        """Apply the owner's ``changes`` (column to new value; a column left out keeps its value) to their ``key``.

        Re-reads the key under its row lock first, so a switch-off or a narrowing an
        operator committed after the caller loaded it is decided on and kept rather
        than written over. A changed access list must fit the owner's own.

        Raises:
            OperatorDisabledKeyError: the changes switch on a key an operator switched off.
            ValueError: a changed access list names something unknown, or grants more than the owner's.
        """
        async with self._uow:
            await self._keys.reload_locked(key)
            if changes.get("is_active") is True and key.operator_disabled:
                raise OperatorDisabledKeyError
            stored = AccessLists(key.allowed_models, key.allowed_tools)
            access = await self.check_access_lists(
                config,
                AccessLists(changes.get("allowed_models", stored.models), changes.get("allowed_tools", stored.tools)),
                stored=stored,
                owner_id=owner_id,
            )
            for column, value in changes.items():
                setattr(key, column, value)
            key.allowed_models = access.models
            key.allowed_tools = access.tools

    async def check_access_lists(
        self, config: GatewayConfig, requested: AccessLists, *, stored: AccessLists, owner_id: str | None
    ) -> AccessLists:
        """The ``allowed_models`` and ``allowed_tools`` to store on a key or a user, each checked only if it changes.

        A list sent back unchanged grants nothing new, so it is not checked again: a
        tool deleted since, or an owner narrowed since, would otherwise refuse an
        unrelated edit. A key's changed list must fit its owner's (``owner_id``;
        ``None`` for a user, or a key with no owner). The owner is read in a block of
        its own, whose commit would also store what the caller had staged, so a route
        calls this first.

        Raises:
            ValueError: a changed list names something unknown, or grants more than the owner's.
        """
        models_changed = requested.models != stored.models
        tools_changed = requested.tools != stored.tools
        checked = AccessLists(
            validate_allowed_models(config, requested.models) if models_changed else stored.models,
            validate_allowed_tools(config, requested.tools) if tools_changed else stored.tools,
        )
        if owner_id is None or not (models_changed or tools_changed):
            return checked
        async with self._uow:
            owner = await self._users.lists(owner_id)
        if owner is None:
            return checked
        tool_keys = config.legacy_search_tool_keys()
        if models_changed and not is_allowlist_subset(checked.models, owner.models, tool_keys=tool_keys):
            raise ValueError(KEY_MODELS_EXCEED_USER_DETAIL)
        if tools_changed and not tools_within_user(
            config, checked.tools, user_tools=owner.tools, user_models=owner.models
        ):
            raise ValueError(KEY_TOOLS_EXCEED_USER_DETAIL)
        return checked

    async def search_tool_refusal(
        self, config: GatewayConfig, api_key: APIKey | None, *, provider: str, tool: str
    ) -> str | None:
        """Why a request's key may not run a search tool, or ``None`` when it may.

        Master-key callers (``api_key is None``) run every tool. The key's owner's
        lists are read only when the key has no ``allowed_tools`` of its own.
        """
        if api_key is None:
            return None
        owner = AccessLists(None, None)
        if api_key.allowed_tools is None and api_key.user_id is not None:
            async with self._uow:
                owner = await self._users.lists(api_key.user_id) or owner
        allowed = effective_tool_allowlist(
            config,
            key_tools=api_key.allowed_tools,
            key_models=api_key.allowed_models,
            user_tools=owner.tools,
            user_models=owner.models,
        )
        if is_tool_allowed(allowed, tool):
            return None
        # With no allowed_tools on either, the answer was read off the model lists, the deprecated way.
        legacy_models: list[list[str]] = []
        if api_key.allowed_tools is None and owner.tools is None:
            legacy_models = [models for models in (api_key.allowed_models, owner.models) if models is not None]
        return tool_not_allowed_detail(tool, f"{provider}:{tool}", legacy_models)

    async def pin_before_user_grant(
        self, config: GatewayConfig, user_id: str, *, before: AccessLists, user_tools: list[str] | None
    ) -> None:
        """Fix each of a user's narrowed keys at the tools it may run, as the user is first given ``allowed_tools``.

        A key with a model list of its own and no ``allowed_tools`` reads its tools
        off that list, bounded by its user's model list; once the user has
        ``user_tools``, the key would inherit those instead, which can widen it. It
        keeps what both allow. Only a user going from no list (``before``) to a list
        pins. The keys are saved in a block of their own, so a route calls this
        before staging the user's change.
        """
        if before.tools is not None or user_tools is None:
            return
        async with self._uow:
            for key in await self._keys.keys_reading_tools_off_models(user_id):
                current = effective_tool_allowlist(
                    config, key_tools=None, key_models=key.allowed_models, user_tools=None, user_models=before.models
                )
                key.allowed_tools = [name for name in current or [] if is_tool_allowed(user_tools, name)]

    async def warn_about_legacy_tool_grants(self, config: GatewayConfig) -> None:
        """Warn at startup while keys or users reach a search tool through ``allowed_models``.

        An exact ``<provider>:<tool>`` entry still works, read as the tool's name, for
        a key or user with no ``allowed_tools`` to inherit; a wildcard no longer does.
        Counts and tool names only.
        """
        legacy = config.legacy_search_tool_keys()
        if not legacy:
            return
        named: set[str] = set()
        wildcard_only: set[str] = set()

        def grants(lists: Iterable[AccessLists]) -> int:
            count = 0
            for access in lists:
                if access.tools is not None or not access.models:
                    continue
                hit = legacy.intersection(access.models)
                wildcard_only.update(key for key in legacy - hit if is_model_allowed(access.models, key))
                if hit:
                    named.update(hit)
                    count += 1
            return count

        async with self._uow:
            users = await self._users.live_lists()
            key_lists = await self._keys.access_lists()
        users_with_tools = {user_id for user_id, access in users.items() if access.tools is not None}
        keys = grants(
            AccessLists(models, tools) for user_id, models, tools in key_lists if user_id not in users_with_tools
        )
        user_count = grants(users.values())
        message: list[str] = []
        if keys or user_count:
            message.append(
                f"{keys} API key(s) and {user_count} user(s) grant search tools through allowed_models entries "
                f"({', '.join(sorted(named))}). These still work but are deprecated: name the tools in allowed_tools "
                "instead."
            )
        if wildcard_only:
            message.append(
                "Search tools some keys or users reached only through an allowed_models wildcard "
                f"({', '.join(sorted(wildcard_only))}) are no longer granted that way; name them in allowed_tools."
            )
        if message:
            logger.warning(" ".join(message))
