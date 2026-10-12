from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Never

from sqlalchemy import case, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.budgets import (
    RESERVATION_ACTIVE,
    RESERVATION_EXPIRED,
    RESERVATION_RELEASED,
    RESERVATION_SETTLED,
    BudgetReservation,
    BudgetReservationScope,
    ReservationStatus,
)
from gateway.repositories.base_repository import BaseRepository

# Asked for by equality, because ``status != ACTIVE`` is an inequality on the leading column of
# ``ix_budget_reservations_status_expires_at`` that the planner cannot use.
_TERMINAL_STATUSES = (RESERVATION_SETTLED, RESERVATION_RELEASED, RESERVATION_EXPIRED)


@dataclass(frozen=True)
class HeldAmounts:
    """What a reservation holds on one counter, on each axis."""

    amount: Decimal
    tokens: int
    requests: int


@dataclass(frozen=True)
class ReservationLine:
    """What a reservation holds on one scoped ceiling."""

    scoped_budget_id: str
    held: HeldAmounts


class ReservationRepository(BaseRepository[BudgetReservation, Never, Never]):
    """Query and stage the reservation ledger: one row per in-flight hold, and one line per ceiling it holds on."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, BudgetReservation)

    async def add(
        self,
        reservation_id: str,
        *,
        user_id: str,
        user_reserved: bool,
        user_held: HeldAmounts,
        ceiling_held: HeldAmounts,
        scoped_budget_ids: Sequence[str],
        expires_at: datetime,
    ) -> None:
        """Stage an active row, and a line holding ``ceiling_held`` on each of ``scoped_budget_ids``."""
        self.db.add(
            BudgetReservation(
                id=reservation_id,
                user_id=user_id,
                estimate=user_held.amount,
                token_estimate=user_held.tokens,
                request_estimate=user_held.requests,
                user_reserved=user_reserved,
                status=RESERVATION_ACTIVE,
                expires_at=expires_at,
            )
        )
        for scoped_budget_id in scoped_budget_ids:
            self.db.add(
                BudgetReservationScope(
                    reservation_id=reservation_id,
                    scoped_budget_id=scoped_budget_id,
                    amount=ceiling_held.amount,
                    token_amount=ceiling_held.tokens,
                    request_amount=ceiling_held.requests,
                )
            )
        await self.db.flush()

    async def add_released(self, reservation_id: str, *, user_id: str) -> None:
        """Stage a released row under ``reservation_id``.

        Raises:
            IntegrityError: a row with this ID exists already.
        """
        self.db.add(
            BudgetReservation(
                id=reservation_id,
                user_id=user_id,
                status=RESERVATION_RELEASED,
                expires_at=datetime.now(UTC),
            )
        )
        await self.db.flush()

    async def delete_by_ids(self, reservation_ids: Sequence[str]) -> None:
        """Delete these rows, and their lines with them."""
        await self.db.execute(
            delete(BudgetReservation)
            .where(BudgetReservation.id.in_(list(reservation_ids)))
            .execution_options(synchronize_session=False)
        )

    async def grow(
        self,
        reservation_id: str,
        *,
        user_delta: Decimal,
        scoped_delta: Decimal,
        token_delta: int,
        scoped_token_delta: int,
    ) -> bool:
        """Add the deltas to an active row and its lines, and report whether the row was active.

        The token delta reaches the row only when the row's per-user leg holds.
        """
        result = await self.db.execute(
            update(BudgetReservation)
            .where(
                BudgetReservation.id == reservation_id,
                BudgetReservation.status == RESERVATION_ACTIVE,
            )
            .values(
                estimate=BudgetReservation.estimate + user_delta,
                token_estimate=BudgetReservation.token_estimate
                + case((BudgetReservation.user_reserved, token_delta), else_=0),
            )
            .execution_options(synchronize_session=False)
        )
        if not getattr(result, "rowcount", 0):
            return False
        if scoped_delta > 0 or scoped_token_delta > 0:
            await self.db.execute(
                update(BudgetReservationScope)
                .where(BudgetReservationScope.reservation_id == reservation_id)
                .values(
                    amount=BudgetReservationScope.amount + scoped_delta,
                    token_amount=BudgetReservationScope.token_amount + scoped_token_delta,
                )
                .execution_options(synchronize_session=False)
            )
        return True

    async def list_expired(self, *, limit: int, user_id: str | None = None) -> list[BudgetReservation]:
        """Return the active rows whose time to live has run out, oldest first, of one user or of every user."""
        query = select(BudgetReservation).where(
            BudgetReservation.status == RESERVATION_ACTIVE,
            BudgetReservation.expires_at < datetime.now(UTC),
        )
        if user_id is not None:
            query = query.where(BudgetReservation.user_id == user_id)
        result = await self.db.execute(query.order_by(BudgetReservation.expires_at).limit(limit))
        return list(result.scalars().all())

    async def list_lines(self, reservation_id: str) -> list[ReservationLine]:
        """Return what this reservation holds on each of its ceilings."""
        result = await self.db.execute(
            select(
                BudgetReservationScope.scoped_budget_id,
                BudgetReservationScope.amount,
                BudgetReservationScope.token_amount,
                BudgetReservationScope.request_amount,
            ).where(BudgetReservationScope.reservation_id == reservation_id)
        )
        return [
            ReservationLine(scoped_budget_id, HeldAmounts(amount=amount, tokens=tokens, requests=requests))
            for scoped_budget_id, amount, tokens, requests in result.tuples().all()
        ]

    async def list_terminal_ids(self, *, expired_before: datetime, limit: int) -> list[str]:
        """Return the IDs of terminal rows whose time to live ran out before ``expired_before``."""
        result = await self.db.execute(
            select(BudgetReservation.id)
            .where(
                BudgetReservation.status.in_(_TERMINAL_STATUSES),
                BudgetReservation.expires_at < expired_before,
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    async def try_settle_reclaimed(self, reservation_id: str) -> bool:
        """Move an expired row to settled, and report whether this call moved it."""
        return await self._transition(reservation_id, RESERVATION_EXPIRED, RESERVATION_SETTLED)

    async def try_terminate(self, reservation_id: str, status: ReservationStatus) -> bool:
        """Move an active row to ``status``, and report whether this call moved it."""
        return await self._transition(reservation_id, RESERVATION_ACTIVE, status)

    async def _transition(self, reservation_id: str, current: ReservationStatus, new: ReservationStatus) -> bool:
        result = await self.db.execute(
            update(BudgetReservation)
            .where(BudgetReservation.id == reservation_id, BudgetReservation.status == current)
            .values(status=new)
            .execution_options(synchronize_session=False)
        )
        return bool(getattr(result, "rowcount", 0))
