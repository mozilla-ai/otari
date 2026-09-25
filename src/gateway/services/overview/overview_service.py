"""The overview's counts and budget health, judged server-side.

The dashboard used to answer these by downloading four collections and
reducing them in the browser (otari#1425). What it renders is three integers
and one worst-case row, so that is what this returns.

The split with the dashboard is deliberate: **this scans, the dashboard
writes the copy.** Deciding which row is worst means reading every row, which
is the expensive half and belongs here. Turning "two rows over their limit"
into the words on the strip is presentation, and putting it on the wire would
make the API own UI copy and freeze the page's wording into four SDKs.
"""

import uuid
from dataclasses import dataclass

from gateway.exceptions.organizations_exceptions import NotAuthorizedError, WorkspaceNotFoundError
from gateway.models.tenancy import Organization
from gateway.models.tenancy import User as TenancyUser
from gateway.repositories.overview.overview_repository import Allocation, OverviewRepository
from gateway.services.tenancy.deployment_user_service import DeploymentUserService
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.tenancy.workspace_service import WorkspaceService

# Where a row stops being comfortable. The dashboard used the same fraction when
# it did this scan itself; it is here now because the counts below are what it
# is for, and a second definition is how the two would drift apart.
BUDGET_WARN = 0.8


@dataclass(frozen=True)
class WorstAllocation:
    """The single row furthest through its allowance."""

    budget_id: str
    name: str | None
    spent: float
    allocated: float
    scope_type: str | None
    scope_id: str | None


@dataclass(frozen=True)
class AllocationHealth:
    """The scan of one set of capped rows."""

    over_count: int
    near_count: int
    capped_count: int
    total_count: int
    worst: WorstAllocation | None


@dataclass(frozen=True)
class OverviewSummary:
    """Everything the overview renders that is not a usage series."""

    active_keys: int
    active_members: int
    budgets: AllocationHealth | None
    ceilings: AllocationHealth | None


def _utilization(row: Allocation) -> float:
    """How far through its allowance a row is.

    A zero allocation admits nothing, so anything spent against one is over it.
    Reported as a full 1.0 rather than the infinite ratio it really is: the share
    has no finite value, and "over budget" tells a reader more than "Infinity%".
    It is a floor, so a row measurably further past its limit still wins ``worst``.
    """

    if row.allocated > 0:
        return row.spent / row.allocated
    return 1.0 if row.spent > 0 else 0.0


def judge(rows: list[Allocation], *, total_count: int) -> AllocationHealth:
    """Reduce capped rows to counts and the worst of them."""

    scored = [(row, _utilization(row)) for row in rows]
    worst_row = max(scored, key=lambda pair: pair[1], default=None)
    return AllocationHealth(
        over_count=sum(1 for _, pct in scored if pct >= 1),
        near_count=sum(1 for _, pct in scored if BUDGET_WARN <= pct < 1),
        capped_count=len(rows),
        total_count=total_count,
        worst=(
            WorstAllocation(
                budget_id=worst_row[0].budget_id,
                name=worst_row[0].name,
                spent=worst_row[0].spent,
                allocated=worst_row[0].allocated,
                scope_type=worst_row[0].scope_type,
                scope_id=worst_row[0].scope_id,
            )
            if worst_row is not None
            else None
        ),
    )


class OverviewService:
    """Assembles the dashboard overview's summary for one caller."""

    def __init__(
        self,
        repository: OverviewRepository,
        organizations: OrganizationService,
        operators: DeploymentUserService,
        workspaces: WorkspaceService,
    ):
        self._repository = repository
        self._organizations = organizations
        self._operators = operators
        self._workspaces = workspaces

    async def summary(
        self,
        *,
        identity: TenancyUser,
        workspace_id: uuid.UUID | None,
    ) -> OverviewSummary:
        """The counts and health strips this caller may see.

        Both health fields are optional and both are withheld rather than
        emptied, because the page draws a different strip for "nothing to judge"
        than for "not yours to see". Deployment budgets are the operator's;
        spend ceilings are an organization owner's or admin's, which is the line
        the ceilings route already draws.
        """

        organization = await self._organizations.get_active_organization_for_user(identity)
        is_deployment_operator = await self._operators.has_administration_access(identity)
        # A workspace the caller named has to be one they may see, which is the
        # rule `services.tenancy.authorization` already states: an owner or admin
        # sees every workspace in their organization, anyone else only the ones
        # they are an active member of. Reusing it is what keeps these counts
        # equal to the pages they link to, which resolve the same way.
        #
        # A workspace they may not see is treated as none given rather than
        # refused, because the id comes from a switcher whose contents can go
        # stale and a stale one should not fail the whole page.
        scope = workspace_id
        if scope is not None:
            try:
                await self._workspaces.workspace_in_active_organization(user=identity, workspace_id=scope)
            except WorkspaceNotFoundError:
                scope = None

        budgets = None
        if is_deployment_operator:
            budgets = judge(
                await self._repository.budget_allocations(),
                total_count=await self._repository.count_budgets(),
            )

        ceilings = None
        if await self._manages_spend(identity, organization):
            ceilings = judge(
                await self._repository.ceiling_allocations(organization.id),
                total_count=await self._repository.count_ceilings(organization.id),
            )

        return OverviewSummary(
            active_keys=await self._repository.count_active_keys(
                workspace_id=scope,
                organization_id=None if is_deployment_operator else organization.id,
            ),
            active_members=(await self._repository.count_active_workspace_members(scope)) if scope else 0,
            budgets=budgets,
            ceilings=ceilings,
        )

    async def _manages_spend(self, identity: TenancyUser, organization: Organization) -> bool:
        """Whether this caller may see their organization's spend ceilings.

        Asked rather than required: the summary answers 200 for everyone and
        leaves the field out, because a member's overview is a valid page that
        simply has no ceiling strip on it. The refusal is the ceilings route's
        own, so the two cannot disagree about who may read a ceiling.
        """

        try:
            await self._organizations.require_active_organization_management_access(
                user=identity,
                organization=organization,
            )
        except NotAuthorizedError:
            return False
        return True
