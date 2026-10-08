import uuid

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.budget_exceptions import DeploymentBudgetIsEndUserDefaultError
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import User
from gateway.rate_limit import BudgetMinuteLimits
from gateway.repositories.budgets import BudgetRepositories
from gateway.schemas.budgets import (
    BudgetResponse,
    CreateBudgetRequest,
    EndUserPublic,
    OrganizationBudgetCreate,
    OrganizationBudgetPublic,
    OrganizationBudgetsPublic,
    OrganizationBudgetUpdate,
    OrganizationScopedBudgetCreate,
    OrganizationScopedBudgetPublic,
    OrganizationScopedBudgetsPublic,
    OrganizationScopedBudgetUpdate,
    WorkspaceMemberBudgetPoliciesPublic,
    WorkspaceMemberBudgetPolicyCreate,
    WorkspaceMemberBudgetPolicyPublic,
    WorkspaceMemberBudgetPolicyUpdate,
)
from gateway.services.api_keys import ApiKeyService
from gateway.services.budgets._deployment_surface import _DeploymentSurface
from gateway.services.budgets._end_users import ResolvedEndUser, _EndUsers, end_user_budget_list
from gateway.services.budgets._member_policies import _MemberPolicies
from gateway.services.budgets._organization_surface import _OrganizationSurface
from gateway.services.budgets._reservations import _normalize_strategy
from gateway.services.budgets._scopes import ScopeOwnership
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tenancy.organization_service import OrganizationService


class BudgetService:
    """The budgets domain's use cases: budgets, the spend ceilings that enforce them, and member budget policies.

    Each public method is one business step, run in a block of the Unit of Work it was built on.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: BudgetRepositories,
        organizations: OrganizationService,
        api_keys: ApiKeyService,
        workspace_access: WorkspaceAccess,
    ) -> None:
        self._uow = uow
        self._budgets = repositories.budgets
        self._ceilings = repositories.ceilings
        self._api_keys = api_keys
        self._organization = _OrganizationSurface(repositories, ScopeOwnership(organizations, api_keys), organizations)
        self._end_users = _EndUsers(repositories)
        self._deployment = _DeploymentSurface(repositories)
        self._member_policies = _MemberPolicies(repositories, organizations, workspace_access)

    async def create_member_policy(
        self, *, user: User, workspace_id: uuid.UUID, request: WorkspaceMemberBudgetPolicyCreate
    ) -> WorkspaceMemberBudgetPolicyPublic:
        """Give every active member of a workspace a ceiling on one budget, now and whenever a member joins."""
        async with self._uow:
            return await self._member_policies.create_policy(user=user, workspace_id=workspace_id, request=request)

    async def create_organization_budget(
        self, *, user: User, request: OrganizationBudgetCreate
    ) -> OrganizationBudgetPublic:
        """Create a budget owned by the caller's organization, applied to the entities the request names."""
        async with self._uow:
            return await self._organization.create_budget(user=user, request=request)

    async def create_organization_ceiling(
        self, *, user: User, request: OrganizationScopedBudgetCreate
    ) -> OrganizationScopedBudgetPublic:
        """Cap one identity inside the caller's organization at one of its budgets."""
        async with self._uow:
            return await self._organization.create_ceiling(user=user, request=request)

    async def delete_api_key_ceilings(self, key_id: str) -> None:
        """Delete the ceilings on an API key the caller has staged for deletion, committing both.

        ``scope_id`` is not a foreign key, so nothing cascades to them.
        """
        async with self._uow:
            await self._ceilings.delete_for_api_key(key_id)

    async def delete_deployment_budget(self, budget_id: str) -> None:
        """Delete a budget the deployment owns, with its reset history, unless something still names it.

        A key's default end-user budget counts as naming it; a key that only lists it loses it from the list.
        """
        async with self._uow:
            if keys := await self._api_keys.keys_defaulting_end_users_to(budget_id):
                raise DeploymentBudgetIsEndUserDefaultError(keys)
            await self._deployment.delete_budget(budget_id)
            await self._api_keys.forget_end_user_budget(budget_id)

    async def delete_member_policy(self, *, user: User, workspace_id: uuid.UUID, policy_id: str) -> None:
        """Stop handing a workspace's new members a ceiling. The ceilings already handed out stay."""
        async with self._uow:
            await self._member_policies.delete_policy(user=user, workspace_id=workspace_id, policy_id=policy_id)

    async def delete_organization_budget(self, *, user: User, budget_id: str) -> None:
        """Delete a budget the caller's organization owns, unless something still names it."""
        async with self._uow:
            await self._organization.delete_budget(user=user, budget_id=budget_id)

    async def delete_organization_ceiling(self, *, user: User, ceiling_id: str) -> None:
        """Remove a ceiling inside the caller's organization."""
        async with self._uow:
            await self._organization.delete_ceiling(user=user, ceiling_id=ceiling_id)

    async def list_member_policies(
        self, *, user: User, workspace_id: uuid.UUID, skip: int = 0, limit: int = 100
    ) -> WorkspaceMemberBudgetPoliciesPublic:
        """Return a page of a workspace's member budget policies."""
        async with self._uow:
            return await self._member_policies.list_policies(
                user=user, workspace_id=workspace_id, skip=skip, limit=limit
            )

    async def list_organization_budgets(
        self, *, user: User, skip: int = 0, limit: int = 100
    ) -> OrganizationBudgetsPublic:
        """Return a page of the caller's organization's budgets, each with the entities it applies to."""
        async with self._uow:
            return await self._organization.list_budgets(user=user, skip=skip, limit=limit)

    async def list_organization_ceilings(
        self, *, user: User, skip: int = 0, limit: int = 100, budget_id: str | None = None
    ) -> OrganizationScopedBudgetsPublic:
        """Return a page of the ceilings capping identities inside the caller's organization.

        A ceiling on a deployment budget is included, because it caps this organization's spend.
        ``budget_id`` narrows the page to the ceilings applying that budget.
        """
        async with self._uow:
            return await self._organization.list_ceilings(user=user, skip=skip, limit=limit, budget_id=budget_id)

    async def put_deployment_budget(self, budget_id: str, request: CreateBudgetRequest) -> tuple[BudgetResponse, bool]:
        """Create a deployment budget under an id the caller chose, or replace it; True when created."""
        async with self._uow:
            return await self._deployment.put_budget(budget_id, request)

    async def check_end_user_budgets(
        self, budget_ids: list[str] | None, default_id: str | None, *, check_default: bool, check_list: bool = True
    ) -> list[str] | None:
        """Return a key's end-user budget list, deduplicated, once it holds together with the default.

        Refuses a default off the list, and a budget an end user may not be capped at: an unknown one, or a tenant's.
        """
        listed, to_check = end_user_budget_list(
            budget_ids, default_id, check_default=check_default, check_list=check_list
        )
        if to_check:
            async with self._uow:
                await self._end_users.require_assignable_budgets(to_check)
        return listed

    async def minute_limits(self, user_id: str, *, strategy: str | None) -> BudgetMinuteLimits | None:
        """The per-minute limits of the user's own budget, or None when it sets neither or budgets are disabled."""
        if _normalize_strategy(strategy) == "disabled":
            return None
        async with self._uow:
            found = await self._budgets.minute_limits_for_user(user_id)
        return BudgetMinuteLimits(*found) if found is not None else None

    async def resolve_end_user(
        self, *, api_key: APIKey, external_id: str, requested_budget_id: str | None = None
    ) -> ResolvedEndUser:
        """Return the end user a service key named, creating it on first use.

        A new end user starts on ``requested_budget_id``, which must be on the key's list, or else on the key's
        default. End users belong to the key's own user, so a key can only bill end users in its owner's scope.
        """
        async with self._uow:
            return await self._end_users.resolve(api_key, external_id, requested_budget_id)

    async def get_end_user(self, *, api_key: APIKey, external_id: str) -> EndUserPublic:
        """Return the end user of the key's owner that the service named ``external_id``."""
        async with self._uow:
            return await self._end_users.get(api_key, external_id)

    async def put_end_user(self, *, api_key: APIKey, external_id: str, budget_id: str) -> tuple[EndUserPublic, bool]:
        """Put an end user on a budget from the key's list, creating it first if needed; True when created."""
        async with self._uow:
            return await self._end_users.put(api_key, external_id, budget_id)

    async def update_end_user(
        self, *, api_key: APIKey, external_id: str, blocked: bool | None, budget_id: str | None
    ) -> EndUserPublic:
        """Block, unblock or move an end user of the key's owner."""
        async with self._uow:
            return await self._end_users.update(api_key, external_id, blocked=blocked, budget_id=budget_id)

    async def update_member_policy(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        policy_id: str,
        request: WorkspaceMemberBudgetPolicyUpdate,
    ) -> WorkspaceMemberBudgetPolicyPublic:
        """Point a member budget policy at another budget, for members who join from now on."""
        async with self._uow:
            return await self._member_policies.update_policy(
                user=user, workspace_id=workspace_id, policy_id=policy_id, request=request
            )

    async def update_organization_budget(
        self, *, user: User, budget_id: str, request: OrganizationBudgetUpdate
    ) -> OrganizationBudgetPublic:
        """Change a budget the caller's organization owns and the entities it applies to, in one transaction."""
        async with self._uow:
            return await self._organization.update_budget(user=user, budget_id=budget_id, request=request)

    async def update_organization_ceiling(
        self, *, user: User, ceiling_id: str, request: OrganizationScopedBudgetUpdate
    ) -> OrganizationScopedBudgetPublic:
        """Relabel a ceiling inside the caller's organization, or point it at another budget the organization owns."""
        async with self._uow:
            return await self._organization.update_ceiling(user=user, ceiling_id=ceiling_id, request=request)


__all__ = ["BudgetService"]
