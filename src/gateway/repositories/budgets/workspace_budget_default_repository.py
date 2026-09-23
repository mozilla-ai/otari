import uuid
from typing import Never

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.elements import ColumnElement

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.budget_exceptions import MemberBudgetPolicyAlreadyExistsError
from gateway.models.budgets import WorkspaceBudgetDefault
from gateway.repositories.base_repository import BaseRepository


def _for_provider(provider_key_id: str | None) -> ColumnElement[bool]:
    """Match a policy's provider, where None matches the policy that covers every provider."""
    if provider_key_id is None:
        return WorkspaceBudgetDefault.provider_key_id.is_(None)
    return WorkspaceBudgetDefault.provider_key_id == provider_key_id


class WorkspaceBudgetDefaultRepository(BaseRepository[WorkspaceBudgetDefault, Never, Never]):
    """Query and stage member budget policies in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, WorkspaceBudgetDefault)

    async def add(self, policy: WorkspaceBudgetDefault) -> WorkspaceBudgetDefault:
        """Stage a new policy and return it with its generated values.

        Precondition: the budget is one the workspace's organization may name, its own or a deployment budget.
        The foreign key proves only that the budget exists.
        Any refusal that is not a duplicate fails the step rather than being reported as one.

        Raises:
            MemberBudgetPolicyAlreadyExistsError: a policy already caps this workspace's members for this provider.
        """
        # A failed flush expires the row, so the IDs are read before it.
        workspace_id, provider_key_id = policy.workspace_id, policy.provider_key_id
        # Isolated in a savepoint, so a refusal does not abort the caller's step.
        try:
            async with self.db.begin_nested():
                self.db.add(policy)
                await self.db.flush()
        except IntegrityError:
            if not await self.has_policy(workspace_id, provider_key_id):
                raise
            raise MemberBudgetPolicyAlreadyExistsError(workspace_id, provider_key_id) from None
        await self.db.refresh(policy)
        return policy

    async def count_for_budget(self, budget_id: str) -> int:
        """Count the policies that hand out this budget."""
        result = await self.db.execute(
            select(func.count())
            .select_from(WorkspaceBudgetDefault)
            .where(WorkspaceBudgetDefault.budget_id == budget_id)
        )
        return result.scalar_one()

    async def for_workspace(self, workspace_id: uuid.UUID) -> list[WorkspaceBudgetDefault]:
        """Return every policy on a workspace."""
        result = await self.db.execute(
            select(WorkspaceBudgetDefault).where(WorkspaceBudgetDefault.workspace_id == workspace_id)
        )
        return list(result.scalars().all())

    async def get_in_workspace(self, policy_id: str, workspace_id: uuid.UUID) -> WorkspaceBudgetDefault | None:
        """Return the policy with this ID when it sits on this workspace, otherwise None."""
        result = await self.db.execute(
            select(WorkspaceBudgetDefault).where(
                WorkspaceBudgetDefault.id == policy_id,
                WorkspaceBudgetDefault.workspace_id == workspace_id,
            )
        )
        return result.scalar_one_or_none()

    async def has_policy(self, workspace_id: uuid.UUID, provider_key_id: str | None) -> bool:
        """Report whether a policy already caps this workspace's members for this provider."""
        result = await self.db.execute(
            select(func.count())
            .select_from(WorkspaceBudgetDefault)
            .where(WorkspaceBudgetDefault.workspace_id == workspace_id, _for_provider(provider_key_id))
        )
        return result.scalar_one() > 0

    async def page_for_workspace(
        self, workspace_id: uuid.UUID, *, skip: int, limit: int
    ) -> tuple[list[WorkspaceBudgetDefault], int]:
        """Return a page of a workspace's policies, oldest first, plus the total."""
        count_result = await self.db.execute(
            select(func.count())
            .select_from(WorkspaceBudgetDefault)
            .where(WorkspaceBudgetDefault.workspace_id == workspace_id)
        )
        result = await self.db.execute(
            select(WorkspaceBudgetDefault)
            .where(WorkspaceBudgetDefault.workspace_id == workspace_id)
            .order_by(WorkspaceBudgetDefault.created_at, WorkspaceBudgetDefault.id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all()), count_result.scalar_one()

    async def remove(self, policy: WorkspaceBudgetDefault) -> None:
        """Stage the deletion of a policy."""
        await self.db.delete(policy)
        await self.db.flush()
