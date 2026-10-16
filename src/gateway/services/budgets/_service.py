import asyncio
import uuid
from collections.abc import Callable, Mapping

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.settings.api_keys import ApiKeyConfig
from gateway.core.settings.budgets import BudgetConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions import TenancyError
from gateway.exceptions.budget_exceptions import DeclaredAccessError, DeploymentBudgetIsEndUserDefaultError
from gateway.log_config import logger
from gateway.models.api_keys import APIKey
from gateway.models.budgets import Budget, ScopedBudget, ScopeType
from gateway.models.tenancy import User
from gateway.rate_limit import BudgetMinuteLimits
from gateway.repositories.budgets import BudgetRepositories
from gateway.schemas.budgets import (
    BudgetResponse,
    CreateBudgetRequest,
    EndUserBudgetPublic,
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
from gateway.services.budgets._declared import attach_key_ceiling
from gateway.services.budgets._deployment_surface import _DeploymentSurface
from gateway.services.budgets._end_users import ResolvedEndUser, _EndUsers, end_user_budget_list
from gateway.services.budgets._member_policies import _MemberPolicies
from gateway.services.budgets._organization_surface import _OrganizationSurface
from gateway.services.budgets._reservation_moves import (
    MoveOutcome,
    PricingLookup,
    ReservationMove,
    _ReservationMoves,
)
from gateway.services.budgets._reservations import EstimateInputs, ReservationHandle, _normalize_strategy
from gateway.services.budgets._scopes import ScopeOwnership
from gateway.services.tenancy.authorization import WorkspaceAccess
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.types.attempt import Attempt


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
        self._repositories = repositories
        self._budgets = repositories.budgets
        self._api_keys = api_keys
        self._organizations = organizations
        self._scopes = ScopeOwnership(organizations, api_keys)
        self._organization = _OrganizationSurface(repositories, self._scopes, organizations)
        self._end_users = _EndUsers(repositories)
        self._deployment = _DeploymentSurface(repositories)
        self._member_policies = _MemberPolicies(repositories, organizations, workspace_access)
        self._moves = _ReservationMoves(repositories)

    async def list_deployment_budgets(self, organization_id: uuid.UUID, *, skip: int, limit: int) -> list[Budget]:
        """A page of the budgets the deployment owns and the operator's organization owns."""
        async with self._uow:
            return await self._budgets.list_in_reach_of(organization_id, skip=skip, limit=limit)

    async def get_deployment_budget(self, budget_id: str, organization_id: uuid.UUID) -> Budget | None:
        """The budget with this ID when the deployment or the operator's organization owns it."""
        async with self._uow:
            return await self._budgets.get_in_reach_of(budget_id, organization_id)

    async def list_deployment_ceilings(
        self,
        organization_id: uuid.UUID,
        *,
        scope_type: ScopeType | None,
        scope_id: str | None,
        skip: int,
        limit: int,
    ) -> list[tuple[ScopedBudget, Budget]]:
        """A page of the ceilings on scopes inside the operator's organization, each with the budget it names."""
        async with self._uow:
            scopes = await self._scopes.get_scope_ids_in(organization_id)
            return await self._repositories.ceilings.list_in_scopes(
                scopes, skip=skip, limit=limit, scope_type=scope_type, scope_id=scope_id
            )

    async def get_deployment_ceiling(self, ceiling_id: str, organization_id: uuid.UUID) -> ScopedBudget | None:
        """The ceiling with this ID when its scope is inside the operator's organization."""
        async with self._uow:
            scopes = await self._scopes.get_scope_ids_in(organization_id)
            return await self._repositories.ceilings.get_in_scopes(ceiling_id, scopes)

    async def apply_declared_access(
        self,
        budgets: Mapping[str, BudgetConfig],
        api_keys: Mapping[str, ApiKeyConfig],
        *,
        workspace_id: uuid.UUID | None,
        fingerprint: Callable[[str], str],
    ) -> None:
        """Write the budgets and API keys config.yml declares over what is stored, in one transaction.

        Each budget is created or replaced under its id. Each key is created or
        updated under its config name, with its secret, its end-user budgets and,
        when it names one, the ceiling that caps it. Nothing config.yml no longer
        declares is deleted; it only stops being marked as declared. Replicas that
        start at once each run this, and every insert yields to a row another one
        placed first.

        A new key lands in ``workspace_id``, which may be None only when no key is declared.

        Raises:
            DeclaredAccessError: an entry cannot be applied. Nothing is written.
        """
        declared_keys = list(api_keys)
        async with self._uow:
            for budget_id, budget in budgets.items():
                try:
                    await self._deployment.put_budget(budget_id, CreateBudgetRequest(**budget.model_dump()))
                except TenancyError as exc:
                    raise DeclaredAccessError(f"budgets.{budget_id}", exc.message) from exc
            await self._budgets.mark_declared(list(budgets))
            for name, key in api_keys.items():
                if workspace_id is None:
                    raise ValueError("A declared API key needs a workspace to land in")
                try:
                    listed, to_check = end_user_budget_list(
                        key.end_user_budget_ids, key.end_user_budget_id, check_default=True, check_list=True
                    )
                    await self._end_users.require_assignable_budgets(to_check)
                    key_id = await self._api_keys.apply_declared_key(
                        name,
                        key,
                        owner=await self._organizations.billing_user_id(key.user_id),
                        declared_names=declared_keys,
                        end_user_budget_ids=listed,
                        workspace_id=workspace_id,
                        fingerprint=fingerprint,
                    )
                    if key.ceiling is not None:
                        await attach_key_ceiling(self._repositories, key_id, key.ceiling)
                except TenancyError as exc:
                    raise DeclaredAccessError(f"api_keys.{name}", exc.message) from exc
            await self._api_keys.release_undeclared_keys(declared_keys)

    async def create_member_policy(
        self, *, user: User, workspace_id: uuid.UUID, request: WorkspaceMemberBudgetPolicyCreate
    ) -> WorkspaceMemberBudgetPolicyPublic:
        """Give every active member of a workspace a ceiling on one budget, now and whenever a member joins."""
        async with self._uow:
            return await self._member_policies.create_policy(user=user, workspace_id=workspace_id, request=request)

    async def create_organization_budget(
        self, *, user: User, request: OrganizationBudgetCreate
    ) -> OrganizationBudgetPublic:
        """Create a budget owned by the caller's organization."""
        async with self._uow:
            return await self._organization.create_budget(user=user, request=request)

    async def create_organization_ceiling(
        self, *, user: User, request: OrganizationScopedBudgetCreate
    ) -> OrganizationScopedBudgetPublic:
        """Cap one identity inside the caller's organization at one of its budgets."""
        async with self._uow:
            return await self._organization.create_ceiling(user=user, request=request)

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
        """Return a page of the caller's organization's budgets, with how many ceilings hold each."""
        async with self._uow:
            return await self._organization.list_budgets(user=user, skip=skip, limit=limit)

    async def list_organization_ceilings(
        self, *, user: User, skip: int = 0, limit: int = 100
    ) -> OrganizationScopedBudgetsPublic:
        """Return a page of the ceilings capping identities inside the caller's organization.

        A ceiling on a deployment budget is included, because it caps this organization's spend.
        """
        async with self._uow:
            return await self._organization.list_ceilings(user=user, skip=skip, limit=limit)

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

    async def move_reservation(
        self,
        handle: ReservationHandle,
        attempt: Attempt,
        *,
        estimate_inputs: EstimateInputs | None,
        price_of: PricingLookup,
    ) -> None:
        """Hold ``handle`` on the ceilings of ``attempt``'s provider, at ``attempt``'s estimate, before it is sent.

        A ceiling both providers have grows to that estimate when it holds less, because a fallback's hold only grows.
        A ceiling only ``attempt``'s provider has takes what the others then hold.
        A ceiling only the previous provider had gives its hold back, and all of it lands in one commit.
        ``estimate_inputs`` is the request's size, and ``price_of`` the rate it pays per model.

        Raises:
            ProviderSpendCeilingExceededError: a ceiling on ``attempt``'s provider alone has no room.
            SpendCeilingExceededError: a ceiling on every provider has no room.
        Either way ``handle`` holds what it held before.
        """
        move = await self._moves.plan(handle, attempt, estimate_inputs=estimate_inputs, price_of=price_of)
        if move is None:
            return
        try:
            async with self._uow:
                outcome = await self._moves.apply(handle, move)
        except DATABASE_ERRORS:
            if not await self._move_landed(handle, move):
                raise
            outcome = MoveOutcome()
        except asyncio.CancelledError:
            # The commit may have landed before the cancellation, and the handle must then name the new holds.
            if await asyncio.shield(self._move_landed(handle, move)):
                self._moves.retarget(handle, move)
            raise
        if outcome.refusal is not None:
            raise outcome.refusal
        if outcome.reclaimed:
            logger.warning(
                "Reservation %s was reclaimed as leaked before its fallback; its spend settles on the fallback's "
                "budgets. Raise budget_reservation_ttl_sec above the slowest request served.",
                handle.reservation_id,
            )
        self._moves.retarget(handle, move)

    async def resolve_end_user(
        self, *, api_key: APIKey, external_id: str, requested_budget_id: str | None = None
    ) -> ResolvedEndUser:
        """Return the end user a service key named, creating it on first use.

        A new end user starts on ``requested_budget_id``, which must be on the key's list, or else on the key's
        default. End users belong to the key's own user, so a key can only bill end users in its owner's scope.
        """
        async with self._uow:
            return await self._end_users.resolve(api_key, external_id, requested_budget_id)

    async def require_end_user_owner(self, *, api_key: APIKey) -> None:
        """Refuse a service key whose own user is blocked or deleted, so it manages none of its end users."""
        async with self._uow:
            await self._end_users.require_owner(api_key)

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

    async def list_end_users(
        self,
        *,
        api_key: APIKey,
        external_id: str | None,
        budget_id: str | None,
        blocked: bool | None,
        skip: int,
        limit: int,
    ) -> list[EndUserPublic]:
        """Return a page of the key owner's end users, oldest first; ``external_id`` narrows it to that one."""
        async with self._uow:
            return await self._end_users.page(
                api_key, external_id=external_id, budget_id=budget_id, blocked=blocked, skip=skip, limit=limit
            )

    async def count_end_users(
        self, *, api_key: APIKey, external_id: str | None, budget_id: str | None, blocked: bool | None
    ) -> int:
        """Count the key owner's end users that ``list_end_users`` lists across every page."""
        async with self._uow:
            return await self._end_users.count(api_key, external_id=external_id, budget_id=budget_id, blocked=blocked)

    async def list_end_user_budgets(self, *, api_key: APIKey) -> list[EndUserBudgetPublic]:
        """Return the budgets the key may put its end users on, with how many of its owner's end users each holds."""
        async with self._uow:
            return await self._end_users.budgets(api_key)

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
        """Change a budget the caller's organization owns, and hold every ceiling naming it to the new figure."""
        async with self._uow:
            return await self._organization.update_budget(user=user, budget_id=budget_id, request=request)

    async def update_organization_ceiling(
        self, *, user: User, ceiling_id: str, request: OrganizationScopedBudgetUpdate
    ) -> OrganizationScopedBudgetPublic:
        """Relabel a ceiling inside the caller's organization, or point it at another budget the organization owns."""
        async with self._uow:
            return await self._organization.update_ceiling(user=user, ceiling_id=ceiling_id, request=request)

    async def _move_landed(self, handle: ReservationHandle, move: ReservationMove) -> bool:
        """Report whether a move whose commit did not report success landed, or False when that cannot be told."""
        try:
            async with self._uow:
                return await self._moves.landed(handle, move)
        except DATABASE_ERRORS as exc:
            logger.error(
                "Could not tell whether the fallback holds of reservation %s committed; they are left as they are: %s",
                move.reservation_id,
                describe_database_error(exc),
            )
            return False


__all__ = ["BudgetService"]
