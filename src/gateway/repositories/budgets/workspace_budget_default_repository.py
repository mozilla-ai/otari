import uuid
from typing import Never

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.budget_exceptions import MemberBudgetPolicyAlreadyExistsError
from gateway.models.budgets import WorkspaceBudgetDefault
from gateway.repositories.base_repository import BaseRepository


class WorkspaceBudgetDefaultRepository(BaseRepository[WorkspaceBudgetDefault, Never, Never]):
    """Query and stage member budget policies in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, WorkspaceBudgetDefault)

    async def add(self, policy: WorkspaceBudgetDefault) -> WorkspaceBudgetDefault:
        """Stage a new policy and return it with its generated values.

        The budget the policy names must exist, because every refusal of the insert is reported as a duplicate.

        Raises:
            MemberBudgetPolicyAlreadyExistsError: a policy already caps this workspace's members for this provider.
        """
        self.db.add(policy)
        try:
            await self.db.flush()
        except IntegrityError:
            raise MemberBudgetPolicyAlreadyExistsError(policy.workspace_id, policy.provider_key_id) from None
        await self.db.refresh(policy)
        return policy

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
