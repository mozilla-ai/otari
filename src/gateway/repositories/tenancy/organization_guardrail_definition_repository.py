"""Data access for the guardrails an organization defined for Otari to build.

``organization_guardrail_definitions`` is mapped by the declarative ``Base``
rather than SQLModel, so the ``sqlmodel.col()`` rule in this package's docstring
does not reach it: a column reference here is a plain attribute and type-checks
as one.

Every write here reports its refusal as a return value rather than letting
``IntegrityError`` out: a name the organization already uses, or a definition a
mandate still names. The service above is the layer that owns the answer a
caller gets, and it may not import SQLAlchemy to recognize one.
"""

import uuid
from typing import Never

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.guardrails import OrganizationGuardrail, OrganizationGuardrailDefinition
from gateway.repositories.base_repository import BaseRepository


class OrganizationGuardrailDefinitionRepository(BaseRepository[OrganizationGuardrailDefinition, Never, Never]):
    """Read and write guardrail definitions in the open block of a Unit of Work.

    ``Never`` for the create and update schemas, as `ApiKeyRepository` does: the
    request shapes this table is written from live in the service layer, which a
    repository may not import, and the generic helpers are unused either way
    because a definition's secrets are encrypted before they reach a column.
    """

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, OrganizationGuardrailDefinition)

    async def count_in_organization(self, organization_id: uuid.UUID) -> int:
        """How many definitions the organization holds, for the ceiling above."""
        result = await self.db.execute(
            select(func.count())
            .select_from(OrganizationGuardrailDefinition)
            .where(OrganizationGuardrailDefinition.organization_id == organization_id)
        )
        return result.scalar_one()

    async def list_in_organization(
        self, organization_id: uuid.UUID, *, skip: int, limit: int
    ) -> list[OrganizationGuardrailDefinition]:
        """One page of the organization's definitions, ordered by name.

        Ordered so paging is stable and a test can assert the sequence, the same
        reason the mandates next door order by profile.
        """
        result = await self.db.execute(
            select(OrganizationGuardrailDefinition)
            .where(OrganizationGuardrailDefinition.organization_id == organization_id)
            .order_by(OrganizationGuardrailDefinition.name)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_enabled_in_every_organization(self) -> list[OrganizationGuardrailDefinition]:
        """Every enabled definition on the deployment, for the runner's cache.

        The one read here with no tenant predicate, and deliberately so: the
        caller is a process-wide cache of built guardrails rather than an answer
        to a request, and the scope a request is subject to is settled by the
        mandate query it already runs. One read for the whole deployment, the
        posture `org_provider_key_service` takes for provider keys, and bounded
        the same way: `MAX_DEFINITIONS_PER_ORGANIZATION` caps each organization.

        Disabled rows are left out in the query rather than skipped afterwards,
        because what the caller does with a row is construct a vendor client.

        Ordered so a test can assert the sequence, as the two reads above are.
        """
        result = await self.db.execute(
            select(OrganizationGuardrailDefinition)
            .where(OrganizationGuardrailDefinition.enabled.is_(True))
            .order_by(OrganizationGuardrailDefinition.organization_id, OrganizationGuardrailDefinition.name)
        )
        return list(result.scalars().all())

    async def get_in_organization(
        self, definition_id: uuid.UUID, organization_id: uuid.UUID
    ) -> OrganizationGuardrailDefinition | None:
        """One definition, or None when it belongs to another organization.

        The tenant predicate is in the query rather than checked on the row
        afterwards, so a foreign id cannot be distinguished from an absent one.
        """
        result = await self.db.execute(
            select(OrganizationGuardrailDefinition).where(
                OrganizationGuardrailDefinition.id == definition_id,
                OrganizationGuardrailDefinition.organization_id == organization_id,
            )
        )
        return result.scalar_one_or_none()

    async def add_unless_name_taken(self, definition: OrganizationGuardrailDefinition) -> bool:
        """Stage the definition, answering False when the organization already uses its name.

        Inserted through a SAVEPOINT for the reason
        `users_repository.get_or_create_attribution_user` uses one: a lost race
        against ``uq_org_guardrail_definitions_org_name`` rolls back this row
        alone and leaves the block's session usable, rather than poisoning
        whatever else the step has staged. Rolling the SAVEPOINT back also drops
        the refused row from the session, so no later autoflush retries the
        insert that just failed.
        """
        try:
            async with self.db.begin_nested():
                self.db.add(definition)
        except IntegrityError:
            return False
        return True

    async def flush_unless_name_taken(self) -> bool:
        """Stage the changes made to loaded definitions, answering False on a name collision.

        The rename case, which the create path's index refuses just as firmly and
        which no prior read can rule out.
        """
        try:
            async with self.db.begin_nested():
                await self.db.flush()
        except IntegrityError:
            return False
        return True

    async def delete_unless_mandated(self, definition: OrganizationGuardrailDefinition) -> bool:
        """Drop the definition, answering False while a mandate still names it.

        Through a SAVEPOINT for the same reason the two writes above use one, and
        for one more: ``fk_organization_guardrails_definition`` is ``RESTRICT``,
        so the DELETE itself raises and the service still has to read the
        mandates it is about to name. Rolling back this statement alone is what
        leaves the session able to answer that read.

        A check before the delete would be a different guarantee, not a simpler
        one: a mandate written in between would reach the DELETE anyway and
        arrive as an unhandled ``IntegrityError``.
        """
        try:
            async with self.db.begin_nested():
                await self.db.delete(definition)
                await self.db.flush()
        except IntegrityError:
            return False
        return True

    async def mandating_profiles(self, definition_id: uuid.UUID) -> list[str]:
        """The profiles of the mandates that point at this definition.

        A read of the neighbouring table, which is the pair's other half rather
        than another aggregate: the delete above is refused *by* that table, and
        the answer a caller can act on is which of its rows did it. Ordered so
        the refusal message is stable.
        """
        result = await self.db.execute(
            select(OrganizationGuardrail.profile)
            .where(OrganizationGuardrail.definition_id == definition_id)
            .order_by(OrganizationGuardrail.profile)
        )
        return list(result.scalars().all())
