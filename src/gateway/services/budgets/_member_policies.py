"""Member budget policies: a workspace's template for the ceiling each of its members is capped at.

A policy names a budget, and every active member of the workspace gets a ceiling on it:
the existing members when the policy is created, and a later member when they join,
through :class:`BudgetMembershipListener`.
A ceiling a member already holds for the same provider wins over the policy and is left alone.
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from gateway.exceptions.budget_exceptions import (
    MemberBudgetPolicyAlreadyExistsError,
    WorkspaceBudgetDefaultAlreadyExistsError,
    WorkspaceBudgetDefaultBudgetNotFoundError,
    WorkspaceBudgetDefaultNotFoundError,
)
from gateway.models.budgets import SCOPE_WORKSPACE_MEMBER, Budget, ScopedBudget, WorkspaceBudgetDefault
from gateway.models.tenancy import OrganizationMember, User, WorkspaceMember
from gateway.repositories.budgets import BudgetRepositories
from gateway.schemas.budgets import (
    WorkspaceMemberBudgetPoliciesPublic,
    WorkspaceMemberBudgetPolicyCreate,
    WorkspaceMemberBudgetPolicyPublic,
    WorkspaceMemberBudgetPolicyUpdate,
)
from gateway.services.budgets._periods import budget_window
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tenancy.organization_service import OrganizationService

# How many members one fan-out batch caps, and the ceiling a list read pages at.
_MATERIALIZE_PAGE_SIZE = 500
_MAX_LIST_LIMIT = 1000


def _member_ceiling(member_id: uuid.UUID, policy: WorkspaceBudgetDefault, budget: Budget) -> ScopedBudget:
    """Build one member's ceiling on the budget a policy hands out.

    The ceiling names the budget rather than copying its figure, so editing the budget moves every member on it.
    The period window is the budget's, so every member on it rolls on the same boundary.
    """
    window = budget_window(datetime.now(UTC), budget)
    period_start, period_end = window if window is not None else (None, None)
    return ScopedBudget(
        scope_type=SCOPE_WORKSPACE_MEMBER,
        scope_id=str(member_id),
        provider_key_id=policy.provider_key_id,
        budget_id=budget.budget_id,
        period_start=period_start,
        period_end=period_end,
    )


async def _budget_for(repositories: BudgetRepositories, policy: WorkspaceBudgetDefault) -> Budget:
    """Return the budget a stored policy hands out.

    The foreign key is ``RESTRICT``, so the budget exists. The miss is raised anyway,
    because a database restored without foreign keys would otherwise cap members at no limit.
    """
    budget = await repositories.budgets.get(policy.budget_id)
    if budget is None:
        raise WorkspaceBudgetDefaultBudgetNotFoundError(policy.budget_id)
    return budget


class BudgetMembershipListener:
    """Keeps members' ceilings in step with their memberships, inside the caller's Unit of Work block.

    It holds repositories and nothing that commits, so a refused membership change takes its writes back with it.
    """

    def __init__(self, repositories: BudgetRepositories) -> None:
        self._repositories = repositories

    async def member_joined(self, member: WorkspaceMember) -> None:
        """Cap the member under each of the workspace's policies they are not already capped for."""
        for policy in await self._repositories.member_policies.for_workspace(member.workspace_id):
            if await self._repositories.ceilings.member_ceiling(member.id, policy.provider_key_id) is not None:
                continue
            budget = await _budget_for(self._repositories, policy)
            await self._repositories.ceilings.insert_member_ceilings([_member_ceiling(member.id, policy, budget)])

    async def member_removed(self, member: WorkspaceMember) -> None:
        """Delete the ceilings keyed on this membership."""
        await self._repositories.ceilings.delete_for_member(member.id)

    async def organization_member_removed(self, member: OrganizationMember) -> None:
        """Delete the ceilings keyed on this organization membership.

        Suspension keeps the row and a re-invite revives it, so a ceiling left here
        would bind again on the person's return.
        """
        await self._repositories.ceilings.delete_for_organization_member(member.id)

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        """Delete the ceilings that would outlive the workspace."""
        await self._repositories.ceilings.delete_for_workspace(workspace_id, member_ids)


class _MemberPolicies:
    """A workspace's member budget policy use cases, each run inside the caller's Unit of Work block.

    Any member of the workspace may read its policies. Only a workspace or organization manager may change them.
    """

    def __init__(
        self,
        repositories: BudgetRepositories,
        organizations: OrganizationService,
        access: WorkspaceAccess,
    ) -> None:
        self._repositories = repositories
        self._organizations = organizations
        self._access = access

    async def _require_budget(self, budget_id: str, *, organization_id: uuid.UUID) -> Budget:
        """Return the budget a caller named, refused as not found when it is not theirs to name.

        A budget owned by another organization is refused, or one tenant's admin could hand their workspace
        another tenant's budget and then move that tenant's cap by editing it.
        A budget with no organization is the deployment's and stays nameable, as it always has been.
        """
        budget = await self._repositories.budgets.get(budget_id)
        if budget is None or (budget.organization_id is not None and budget.organization_id != organization_id):
            raise WorkspaceBudgetDefaultBudgetNotFoundError(budget_id)
        return budget

    async def _cap_active_members(self, policy: WorkspaceBudgetDefault, budget: Budget) -> None:
        """Cap every active member of the policy's workspace who is not already capped for its provider."""
        skip = 0
        while True:
            member_ids, total = await self._organizations.page_active_workspace_member_ids(
                policy.workspace_id, skip=skip, limit=_MATERIALIZE_PAGE_SIZE
            )
            if not member_ids:
                return
            covered = await self._repositories.ceilings.members_with_ceiling(member_ids, policy.provider_key_id)
            await self._repositories.ceilings.insert_member_ceilings(
                [_member_ceiling(member_id, policy, budget) for member_id in member_ids if member_id not in covered]
            )
            skip += len(member_ids)
            if skip >= total:
                return

    async def list_policies(
        self, *, user: User, workspace_id: uuid.UUID, skip: int, limit: int
    ) -> WorkspaceMemberBudgetPoliciesPublic:
        """Return a page of a workspace's policies, oldest first, plus how many there are."""
        await self._access.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        policies, count = await self._repositories.member_policies.page_for_workspace(
            workspace_id, skip=skip, limit=min(limit, _MAX_LIST_LIMIT)
        )
        budgets = await self._repositories.budgets.get_many([policy.budget_id for policy in policies])
        # Refused rather than returning a short page, which would disagree with ``count``.
        missing = next((policy for policy in policies if policy.budget_id not in budgets), None)
        if missing is not None:
            raise WorkspaceBudgetDefaultBudgetNotFoundError(missing.budget_id)
        return WorkspaceMemberBudgetPoliciesPublic(
            data=[
                WorkspaceMemberBudgetPolicyPublic.from_model(policy, budgets[policy.budget_id]) for policy in policies
            ],
            count=count,
        )

    async def create_policy(
        self, *, user: User, workspace_id: uuid.UUID, request: WorkspaceMemberBudgetPolicyCreate
    ) -> WorkspaceMemberBudgetPolicyPublic:
        """Create a policy and cap every existing active member under it."""
        workspace = await self._access.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self._access.require_workspace_management_access(user=user, workspace=workspace)
        # The lock every membership-creation path takes, so a member joining concurrently
        # is capped either here or by the listener, and never by neither.
        await self._organizations.lock_workspace(workspace.id)
        # Before the insert, so an unknown budget is not found rather than a foreign key refusal.
        budget = await self._require_budget(request.budget_id, organization_id=workspace.organization_id)
        try:
            policy = await self._repositories.member_policies.add(
                WorkspaceBudgetDefault(
                    workspace_id=workspace.id,
                    budget_id=budget.budget_id,
                    provider_key_id=request.provider_key_id,
                )
            )
        except MemberBudgetPolicyAlreadyExistsError:
            raise WorkspaceBudgetDefaultAlreadyExistsError(workspace_id, request.provider_key_id) from None
        await self._cap_active_members(policy, budget)
        return WorkspaceMemberBudgetPolicyPublic.from_model(policy, budget)

    async def update_policy(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        policy_id: str,
        request: WorkspaceMemberBudgetPolicyUpdate,
    ) -> WorkspaceMemberBudgetPolicyPublic:
        """Point a policy at another budget.

        Members already capped keep the budget they were handed; this changes what a member joining later gets.
        The provider is not editable, because changing it is a delete and a create.
        """
        workspace = await self._access.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self._access.require_workspace_management_access(user=user, workspace=workspace)
        policy = await self._repositories.member_policies.get_in_workspace(policy_id, workspace.id)
        if policy is None:
            raise WorkspaceBudgetDefaultNotFoundError(policy_id)
        budget = await self._require_budget(request.budget_id, organization_id=workspace.organization_id)
        policy = await self._repositories.member_policies.set_budget(policy, budget.budget_id)
        return WorkspaceMemberBudgetPolicyPublic.from_model(policy, budget)

    async def delete_policy(self, *, user: User, workspace_id: uuid.UUID, policy_id: str) -> None:
        """Delete a policy. The ceilings it already created stay, with their spend history."""
        workspace = await self._access.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self._access.require_workspace_management_access(user=user, workspace=workspace)
        policy = await self._repositories.member_policies.get_in_workspace(policy_id, workspace.id)
        if policy is None:
            raise WorkspaceBudgetDefaultNotFoundError(policy_id)
        await self._repositories.member_policies.remove(policy)


__all__ = ["BudgetMembershipListener"]
