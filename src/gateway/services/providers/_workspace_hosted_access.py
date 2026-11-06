"""What one workspace may do with the hosted providers its organization may use.

Whether an organization may use a hosted provider at all is the hosted port's
decision, and it keys on the organization alone. A workspace row can only take
away from that answer: switch the provider off for the workspace, or narrow it
to a list of models. It never grants. No row is full inheritance.

A workspace may also require zero data retention of every provider serving it
(``WorkspaceProviderPolicy``). A hosted provider its operator did not declare
is then refused here, and an organization key or a configured instance where
the organization key's own gate runs (``services/model_access``).

The rule is applied on the gateway side of the port, in
:func:`~gateway.services.providers.hosted_credential_for` and in the catalog,
so the port's contract is unchanged. Dispatch is synchronous about it and holds
no session, so the rows are held in a process-wide cache refreshed on a TTL and
right after a write, the arrangement ``_provider_endpoint_cache`` uses: a write
reaches its own worker at once and every other worker within
``WORKSPACE_HOSTED_ACCESS_CACHE_TTL_SECONDS``.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.exceptions.providers_exceptions import (
    WorkspaceHostedModelsInvalidError,
    WorkspaceHostedProviderNotFoundError,
)
from gateway.log_config import logger
from gateway.models.providers import WorkspaceHostedProviderAccess
from gateway.models.tenancy import User
from gateway.ports.model_provider_port import HostedAccessDeniedError, HostedModels
from gateway.repositories.providers import WorkspaceHostedAccessRepository, WorkspaceProviderPolicyRepository
from gateway.schemas.providers import (
    WorkspaceHostedProviderAccessRequest,
    WorkspaceHostedProviderPublic,
    WorkspaceHostedProvidersPublic,
    WorkspaceProviderPolicyPublic,
    WorkspaceProviderPolicyRequest,
)
from gateway.services.providers._hosted_name_index import hosted_zero_data_retention
from gateway.services.tenancy.authorization import WorkspaceAccess

WORKSPACE_HOSTED_ACCESS_CACHE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class WorkspaceHostedRule:
    """One stored departure: the provider switched off, or narrowed to ``models``."""

    enabled: bool
    models: frozenset[str] | None


class WorkspaceHostedAccessDeniedError(HostedAccessDeniedError):
    """The workspace switched the hosted provider off, or does not allow the model on it."""


# workspace_id -> hosted provider name -> rule
_rules: dict[uuid.UUID, dict[str, WorkspaceHostedRule]] = {}
# The workspaces that require zero data retention of every provider serving them.
_zero_retention_workspaces: frozenset[uuid.UUID] = frozenset()
# Refreshes started, and the number of the newest one applied.
_refresh_count = 0
_applied_refresh = 0


def workspace_requires_zero_data_retention(workspace_id: uuid.UUID | None) -> bool:
    """Whether the workspace refuses every provider not declared zero data retention."""
    return workspace_id is not None and workspace_id in _zero_retention_workspaces


def cached_workspace_hosted_rule(workspace_id: uuid.UUID, provider: str) -> WorkspaceHostedRule | None:
    """The rule this worker last loaded for the workspace and provider, or ``None`` when it inherits."""
    return _rules.get(workspace_id, {}).get(provider)


def ensure_workspace_may_use_hosted(workspace_id: uuid.UUID | None, provider: str, model: str | None) -> None:
    """Refuse a hosted dispatch the workspace's own rule forbids.

    ``model`` is ``None`` for a call that names no model, which only the
    provider's switch can refuse.

    Raises:
        WorkspaceHostedAccessDeniedError: the workspace switched the provider
            off, narrowed it to models that leave ``model`` out, or requires
            zero data retention the provider is not declared to keep.
    """
    if workspace_id is None:
        return
    if workspace_requires_zero_data_retention(workspace_id) and not hosted_zero_data_retention(provider):
        raise WorkspaceHostedAccessDeniedError(
            f"Hosted provider '{provider}' is not declared zero data retention, which this workspace requires",
            workspace_id,
        )
    rule = cached_workspace_hosted_rule(workspace_id, provider)
    if rule is None:
        return
    if not rule.enabled:
        raise WorkspaceHostedAccessDeniedError(
            f"Hosted provider '{provider}' is switched off for this workspace", workspace_id
        )
    if model is not None and rule.models is not None and model not in rule.models:
        raise WorkspaceHostedAccessDeniedError(
            f"Model '{model}' is not allowed on hosted provider '{provider}' in this workspace", workspace_id
        )


def workspace_may_use_hosted(workspace_id: uuid.UUID, provider: str, model: str | None) -> bool:
    """Whether :func:`ensure_workspace_may_use_hosted` lets the dispatch through, for a reader that lists."""
    try:
        ensure_workspace_may_use_hosted(workspace_id, provider, model)
    except WorkspaceHostedAccessDeniedError:
        return False
    return True


def narrow_hosted_models(hosted: HostedModels, workspace_id: uuid.UUID | None) -> HostedModels:
    """``hosted`` as one workspace may use it, for a catalog that answers for that workspace.

    A provider the workspace switched off is dropped, and so is one not
    declared zero data retention where the workspace requires it. A narrowed
    provider keeps the models both lists name; where the deployment advertises
    no roster, the workspace's list is the roster.
    """
    if workspace_id is None:
        return hosted
    if workspace_requires_zero_data_retention(workspace_id):
        hosted = {provider: roster for provider, roster in hosted.items() if hosted_zero_data_retention(provider)}
    rules = _rules.get(workspace_id)
    if not rules:
        return hosted
    narrowed: dict[str, frozenset[str] | None] = {}
    for provider, roster in hosted.items():
        rule = rules.get(provider)
        if rule is None:
            narrowed[provider] = roster
        elif not rule.enabled:
            continue
        elif rule.models is None:
            narrowed[provider] = roster
        else:
            narrowed[provider] = rule.models if roster is None else roster & rule.models
    return narrowed


def _rule_of(row: WorkspaceHostedProviderAccess) -> WorkspaceHostedRule:
    return WorkspaceHostedRule(enabled=row.enabled, models=frozenset(row.models) if row.models is not None else None)


async def refresh_workspace_hosted_access_cache(uow: UnitOfWork) -> None:
    """Reload every row and policy, and swap the cache in one step.

    Numbered before the read, so a refresh whose read began earlier never
    replaces what a later one already applied: a write refreshes after it
    commits, and the periodic refresher must not undo that with an older read.
    """
    global _rules, _zero_retention_workspaces, _refresh_count, _applied_refresh  # noqa: PLW0603

    _refresh_count += 1
    number = _refresh_count
    async with uow:
        rows = await WorkspaceHostedAccessRepository(uow).list_all()
        zero_retention = await WorkspaceProviderPolicyRepository(uow).zero_data_retention_workspaces()
    if number < _applied_refresh:
        return
    rules: dict[uuid.UUID, dict[str, WorkspaceHostedRule]] = {}
    for row in rows:
        rules.setdefault(row.workspace_id, {})[row.provider] = _rule_of(row)
    _rules, _zero_retention_workspaces, _applied_refresh = rules, zero_retention, number


def reset_workspace_hosted_access_cache() -> None:
    """Drop the cache so the next load starts clean (startup and tests)."""
    global _rules, _zero_retention_workspaces, _applied_refresh  # noqa: PLW0603

    _rules, _zero_retention_workspaces, _applied_refresh = {}, frozenset(), 0


async def _refresh_on_a_session_of_its_own() -> None:
    async with create_unit_of_work() as uow:
        await refresh_workspace_hosted_access_cache(uow)


async def load_workspace_hosted_access_at_startup() -> None:
    """Prime the cache before the first request. A failure is logged, not raised."""
    reset_workspace_hosted_access_cache()
    try:
        await _refresh_on_a_session_of_its_own()
    except Exception:
        logger.exception("Failed to load workspace hosted provider access; continuing with none cached")


async def run_workspace_hosted_access_refresher(
    interval: float = WORKSPACE_HOSTED_ACCESS_CACHE_TTL_SECONDS,
) -> None:
    """Reload the cache forever, so another worker's writes arrive. Every error is retried next tick."""
    while True:
        await asyncio.sleep(interval)
        try:
            await _refresh_on_a_session_of_its_own()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Workspace hosted provider access refresh failed; retrying in %ss", interval, exc_info=True)


def _validated_models(models: list[str] | None, served: frozenset[str] | None) -> list[str] | None:
    """The list to store: trimmed and deduplicated, and only models the provider serves."""
    if models is None:
        return None
    cleaned = list(dict.fromkeys(name.strip() for name in models if name.strip()))
    if not cleaned:
        raise WorkspaceHostedModelsInvalidError(
            "Choose at least one model, or allow every model. To block the provider, switch it off."
        )
    if served is not None:
        unknown = [name for name in cleaned if name not in served]
        if unknown:
            raise WorkspaceHostedModelsInvalidError(f"Not served on this provider: {', '.join(sorted(unknown))}")
    return cleaned


class WorkspaceHostedAccessService:
    """Read and change which hosted providers, and which of their models, one workspace may use.

    Reading takes membership of the workspace; changing takes an owner or admin
    of the workspace or of its organization, the rule the workspace's own
    provider key overrides follow.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        rules: WorkspaceHostedAccessRepository,
        policies: WorkspaceProviderPolicyRepository,
        access: WorkspaceAccess,
        hosted_models: Callable[[uuid.UUID | None], Awaitable[HostedModels]],
        byo_providers: Callable[[uuid.UUID], Awaitable[frozenset[str]]],
        refresh_cache: Callable[[], Awaitable[None]],
    ) -> None:
        """Bind the unit of work and the lookups this service needs from other domains.

        ``hosted_models`` answers what the hosted port lets an organization use.
        ``byo_providers`` names the providers a workspace calls on its
        organization's own key, where the hosted row is never consulted.
        ``refresh_cache`` reloads the dispatch-path copy after a write commits.
        """
        self.uow = uow
        self.rules = rules
        self.policies = policies
        self.access = access
        self.hosted_models = hosted_models
        self.byo_providers = byo_providers
        self.refresh_cache = refresh_cache

    async def list_for_workspace(self, *, user: User, workspace_id: uuid.UUID) -> WorkspaceHostedProvidersPublic:
        """Every hosted provider the workspace's organization may use, with this workspace's rule on it."""
        workspace = await self.access.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        hosted = await self.hosted_models(workspace.organization_id)
        own_key = await self.byo_providers(workspace.id)
        async with self.uow:
            stored = {row.provider: row for row in await self.rules.list_for_workspace(workspace.id)}
        return WorkspaceHostedProvidersPublic(
            data=[
                _public(provider, hosted[provider], stored.get(provider), served_by_own_key=provider in own_key)
                for provider in sorted(hosted)
            ]
        )

    async def set_access(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        provider: str,
        request: WorkspaceHostedProviderAccessRequest,
    ) -> WorkspaceHostedProviderPublic:
        """Store the workspace's rule for one hosted provider, replacing any it had.

        The inherited state (on, every model) is stored as no row, so sending it
        is the same as a reset.
        """
        workspace = await self.access.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self.access.require_workspace_management_access(user=user, workspace=workspace)
        hosted = await self.hosted_models(workspace.organization_id)
        if provider not in hosted:
            raise WorkspaceHostedProviderNotFoundError(provider)
        # Switching off checks no name against the roster: the list it keeps may
        # name a model the deployment has since dropped, and that must not hold
        # the provider on.
        models = _validated_models(request.models, hosted[provider] if request.enabled else None)
        async with self.uow:
            if request.enabled and models is None:
                await self.rules.delete_for(workspace.id, provider)
                row = None
            else:
                row = await self.rules.put(workspace.id, provider, enabled=request.enabled, models=models)
        await self.refresh_cache()
        own_key = await self.byo_providers(workspace.id)
        return _public(provider, hosted[provider], row, served_by_own_key=provider in own_key)

    async def get_policy(self, *, user: User, workspace_id: uuid.UUID) -> WorkspaceProviderPolicyPublic:
        """What the workspace requires of every provider serving it. Any member may read it."""
        workspace = await self.access.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        async with self.uow:
            policy = await self.policies.get_for(workspace.id)
        return WorkspaceProviderPolicyPublic(
            require_zero_data_retention=policy is not None and policy.require_zero_data_retention
        )

    async def set_policy(
        self, *, user: User, workspace_id: uuid.UUID, request: WorkspaceProviderPolicyRequest
    ) -> WorkspaceProviderPolicyPublic:
        """Store what the workspace requires of every provider serving it."""
        workspace = await self.access.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self.access.require_workspace_management_access(user=user, workspace=workspace)
        async with self.uow:
            policy = await self.policies.put(
                workspace.id, require_zero_data_retention=request.require_zero_data_retention
            )
        await self.refresh_cache()
        return WorkspaceProviderPolicyPublic(require_zero_data_retention=policy.require_zero_data_retention)

    async def reset_access(
        self, *, user: User, workspace_id: uuid.UUID, provider: str
    ) -> WorkspaceHostedProviderPublic:
        """Drop the workspace's rule for one hosted provider, so it inherits the provider whole. Idempotent."""
        return await self.set_access(
            user=user,
            workspace_id=workspace_id,
            provider=provider,
            request=WorkspaceHostedProviderAccessRequest(enabled=True, models=None),
        )


def _sorted(models: Iterable[str] | None) -> list[str] | None:
    return None if models is None else sorted(models)


def _public(
    provider: str,
    served: frozenset[str] | None,
    row: WorkspaceHostedProviderAccess | None,
    *,
    served_by_own_key: bool,
) -> WorkspaceHostedProviderPublic:
    return WorkspaceHostedProviderPublic(
        provider=provider,
        enabled=row.enabled if row is not None else True,
        models=_sorted(row.models) if row is not None else None,
        customized=row is not None,
        served_models=_sorted(served),
        served_by_own_key=served_by_own_key,
        zero_data_retention=hosted_zero_data_retention(provider),
    )
