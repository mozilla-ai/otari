"""Preflight report for the budget half of the reconciled control plane.

The reconciled engine keeps one budget model, whose reset cadence is one of the
platform's three ``BudgetPeriod`` values. This gateway's ``budgets`` table does
not line up with it in three ways, none of which a migration can decide on an
operator's behalf:

* ``budget_duration_sec`` is an arbitrary second count, so a duration with no
  exact counterpart has to round to the nearest period, changing how much a
  capped user may spend per unit of time.
* Gateway periods roll from each user's ``budget_started_at``; the reconciled
  periods are calendar aligned (midnight UTC, Monday, the 1st). Every budget is
  re-anchored, including the ones whose duration maps exactly.
* One ``budgets`` row may be attached to several users. Enforcement is per user
  (``users.spend + users.reserved`` against that row's ``max_budget``), so the
  row is a shared limit rather than a shared pot of money, and the default
  mapping materializes one member budget per attached user.

This module enumerates all three before an operator crosses the migration,
rather than leaving them to be discovered afterwards from a changed bill. It is
strictly read only: nothing here writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import Budget, BudgetResetLog, User

__all__ = [
    "PERIOD_SECONDS",
    "AttachedUser",
    "BudgetMigrationReport",
    "BudgetPlan",
    "DurationMapping",
    "build_migration_report",
    "map_duration",
    "render_text",
]

# Seconds the reconciled engine's three periods nominally span. Its windows are
# calendar aligned, so a real month runs 28 to 31 days; 30 is the nominal length
# rounding compares against, and the drift it reports is nominal for that reason.
PERIOD_SECONDS: Final[dict[str, int]] = {
    "daily": 86_400,
    "weekly": 604_800,
    "monthly": 2_592_000,
}


@dataclass(frozen=True)
class DurationMapping:
    """Which reconciled period a gateway duration lands on, and what that costs."""

    duration_sec: int
    period: str
    is_exact: bool

    @property
    def drift_sec(self) -> int:
        """Signed gap from the gateway duration to the period it maps onto.

        Positive means the new period is longer than the old duration.
        """
        return PERIOD_SECONDS[self.period] - self.duration_sec

    @property
    def rate_factor(self) -> float:
        """Multiplier on how fast a capped user may spend after the migration.

        The same ``max_budget`` applied over a shorter window is more money per
        unit of time, so a factor above 1.0 loosens the cap and one below 1.0
        tightens it. This is the number that decides whether a rounded duration
        needs an operator to intervene.
        """
        return self.duration_sec / PERIOD_SECONDS[self.period]


def map_duration(duration_sec: int) -> DurationMapping:
    """Map a gateway ``budget_duration_sec`` onto the nearest reconciled period.

    Nearest is measured in seconds. An exact tie (four days sits exactly between
    daily and weekly) resolves to the longer period, which is the direction that
    cannot let a user outspend what the gateway allowed them.
    """
    period, seconds = min(
        PERIOD_SECONDS.items(),
        key=lambda item: (abs(item[1] - duration_sec), -item[1]),
    )
    return DurationMapping(duration_sec=duration_sec, period=period, is_exact=seconds == duration_sec)


@dataclass(frozen=True)
class AttachedUser:
    """A gateway user pointed at a budget, and the counters that carry over."""

    user_id: str
    spend: float
    reserved: float
    is_deleted: bool


@dataclass(frozen=True)
class BudgetPlan:
    """One gateway budget row and what the migration makes of it."""

    budget_id: str
    name: str | None
    max_budget: float | None
    duration_sec: int | None
    attached: list[AttachedUser] = field(default_factory=list)

    @property
    def live_attached(self) -> list[AttachedUser]:
        """Attachments that materialize as member budgets.

        Soft deleted users migrate as deactivated identities, so their budget is
        history rather than a live cap and is counted separately.
        """
        return [user for user in self.attached if not user.is_deleted]

    @property
    def is_shared_pool(self) -> bool:
        return len(self.live_attached) > 1

    @property
    def mapping(self) -> DurationMapping | None:
        """``None`` when the budget has no duration, so it never resets today."""
        return None if self.duration_sec is None else map_duration(self.duration_sec)


@dataclass(frozen=True)
class BudgetMigrationReport:
    """Everything an operator has to decide before the budget migration runs."""

    budgets: list[BudgetPlan]
    reset_log_count: int

    @property
    def unattached(self) -> list[BudgetPlan]:
        return [plan for plan in self.budgets if not plan.live_attached]

    @property
    def member_budgets_to_create(self) -> int:
        return sum(len(plan.live_attached) for plan in self.budgets)

    @property
    def deleted_attachments(self) -> int:
        return sum(len(plan.attached) - len(plan.live_attached) for plan in self.budgets)

    @property
    def rounded(self) -> list[BudgetPlan]:
        return [plan for plan in self.budgets if plan.mapping is not None and not plan.mapping.is_exact]

    @property
    def exact(self) -> list[BudgetPlan]:
        return [plan for plan in self.budgets if plan.mapping is not None and plan.mapping.is_exact]

    @property
    def periodless(self) -> list[BudgetPlan]:
        return [plan for plan in self.budgets if plan.duration_sec is None]

    @property
    def shared_pools(self) -> list[BudgetPlan]:
        return [plan for plan in self.budgets if plan.is_shared_pool]

    def to_dict(self) -> dict[str, Any]:
        """Machine readable form, for diffing a report across dry runs."""
        return {
            "summary": {
                "budgets": len(self.budgets),
                "unattached_budgets": len(self.unattached),
                "member_budgets_to_create": self.member_budgets_to_create,
                "deleted_attachments": self.deleted_attachments,
                "rounded_durations": len(self.rounded),
                "exact_durations": len(self.exact),
                "budgets_without_a_duration": len(self.periodless),
                "shared_pools": len(self.shared_pools),
                "reset_logs_archived": self.reset_log_count,
            },
            "rounded_durations": [_plan_json(plan) for plan in self.rounded],
            "budgets_without_a_duration": [_plan_json(plan) for plan in self.periodless],
            "shared_pools": [_plan_json(plan) for plan in self.shared_pools],
            "unattached_budgets": [plan.budget_id for plan in self.unattached],
        }


def _plan_json(plan: BudgetPlan) -> dict[str, Any]:
    mapping = plan.mapping
    return {
        "budget_id": plan.budget_id,
        "name": plan.name,
        "max_budget": plan.max_budget,
        "duration_sec": plan.duration_sec,
        "period": mapping.period if mapping else None,
        "is_exact": mapping.is_exact if mapping else None,
        "drift_sec": mapping.drift_sec if mapping else None,
        "rate_factor": round(mapping.rate_factor, 4) if mapping else None,
        "attached_users": [
            {
                "user_id": user.user_id,
                "spend": user.spend,
                "reserved": user.reserved,
                "deleted": user.is_deleted,
            }
            for user in plan.attached
        ],
    }


async def build_migration_report(db: AsyncSession) -> BudgetMigrationReport:
    """Read every budget and its attachments, in two queries and no writes."""
    rows = (
        await db.execute(
            select(
                Budget.budget_id,
                Budget.name,
                Budget.max_budget,
                Budget.budget_duration_sec,
                User.user_id,
                User.spend,
                User.reserved,
                User.deleted_at,
            )
            # Outer join so a budget nothing points at still appears: it is a row
            # the migration creates nothing for, which is worth seeing.
            .outerjoin(User, User.budget_id == Budget.budget_id)
            .order_by(Budget.budget_id, User.user_id)
        )
    ).all()

    # The join repeats a budget once per attached user, so the rows are folded
    # back into one plan each before any plan is constructed: BudgetPlan is
    # frozen, and a frozen row whose list is filled in afterwards is only
    # nominally immutable.
    columns: dict[str, tuple[str | None, float | None, int | None]] = {}
    attachments: dict[str, list[AttachedUser]] = {}
    for budget_id, name, max_budget, duration_sec, user_id, spend, reserved, deleted_at in rows:
        columns.setdefault(budget_id, (name, max_budget, duration_sec))
        attached = attachments.setdefault(budget_id, [])
        if user_id is not None:
            attached.append(
                AttachedUser(
                    user_id=user_id,
                    spend=spend or 0.0,
                    reserved=reserved or 0.0,
                    is_deleted=deleted_at is not None,
                )
            )

    reset_log_count = (await db.execute(select(func.count()).select_from(BudgetResetLog))).scalar_one()

    return BudgetMigrationReport(
        budgets=[
            BudgetPlan(
                budget_id=budget_id,
                name=name,
                max_budget=max_budget,
                duration_sec=duration_sec,
                attached=attachments[budget_id],
            )
            for budget_id, (name, max_budget, duration_sec) in columns.items()
        ],
        reset_log_count=reset_log_count,
    )


def _format_duration(seconds: int) -> str:
    """Render a second count the way an operator wrote it, e.g. ``3d 12h``."""
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    parts = [f"{value}{unit}" for value, unit in ((days, "d"), (hours, "h"), (minutes, "m"), (secs, "s")) if value]
    return " ".join(parts) or "0s"


def _label(plan: BudgetPlan) -> str:
    name = f'"{plan.name}" ' if plan.name else ""
    return f"{name}({plan.budget_id})"


def render_text(report: BudgetMigrationReport) -> str:
    """Render the report for a terminal, one section per decision to make."""
    lines: list[str] = ["Budget migration preflight", "=" * 26, ""]

    lines += [
        f"{len(report.budgets)} budget(s) in this gateway.",
        f"{report.member_budgets_to_create} member budget(s) would be created in the default workspace.",
        f"{report.reset_log_count} budget_reset_logs row(s) archive with the legacy tables, unmapped.",
    ]
    if report.deleted_attachments:
        lines.append(
            f"{report.deleted_attachments} attachment(s) belong to soft deleted users, which migrate as "
            "deactivated identities and get no live cap."
        )
    lines.append("")

    lines += [
        "Every budget is re-anchored, not only the rounded ones: gateway periods roll from each",
        "user's budget_started_at, the reconciled periods align to the calendar (midnight UTC,",
        "Monday, the 1st). The first period after the migration is therefore short.",
        "",
    ]

    lines.append(f"Rounded durations ({len(report.rounded)})")
    lines.append("-" * 40)
    if not report.rounded:
        lines.append("  None. Every duration maps onto a period exactly.")
    else:
        lines.append("  rate is the change in how fast a capped user may spend: above 1.00 loosens the cap.")
        for plan in sorted(report.rounded, key=lambda item: item.duration_sec or 0):
            mapping = plan.mapping
            if mapping is None:  # pragma: no cover - report.rounded excludes these
                continue
            direction = "LOOSER" if mapping.rate_factor > 1 else "TIGHTER"
            lines.append(
                f"  {_format_duration(mapping.duration_sec)} -> {mapping.period}"
                f"  (drift {mapping.drift_sec:+d}s, rate {mapping.rate_factor:.2f}x {direction})"
                f"  {_label(plan)}, {len(plan.live_attached)} user(s)"
            )
    lines.append("")

    lines.append(f"Budgets with no duration ({len(report.periodless)})")
    lines.append("-" * 40)
    if not report.periodless:
        lines.append("  None.")
    else:
        lines.append("  These never reset today. The reconciled model requires a period, so each needs one chosen.")
        for plan in report.periodless:
            lines.append(f"  {_label(plan)}, {len(plan.live_attached)} user(s), max_budget {plan.max_budget}")
    lines.append("")

    lines.append(f"Shared budgets ({len(report.shared_pools)})")
    lines.append("-" * 40)
    if not report.shared_pools:
        lines.append("  None. Every attached budget has exactly one user.")
    else:
        lines.append("  Enforcement is already per user against this row's max_budget, so materializing one")
        lines.append("  member budget per user preserves today's behavior. Consolidating to a workspace budget")
        lines.append("  instead would pool the money and tighten the cap, so it is an opt in, not the default.")
        for plan in report.shared_pools:
            lines.append(f"  {_label(plan)}  max_budget {plan.max_budget}, {len(plan.live_attached)} user(s)")
            for user in plan.live_attached:
                lines.append(f"      {user.user_id}: spend {user.spend}, reserved {user.reserved}")
    lines.append("")

    if report.unattached:
        lines.append(f"Unattached budgets ({len(report.unattached)})")
        lines.append("-" * 40)
        lines.append("  No live user points at these, so the migration creates nothing for them.")
        for plan in report.unattached:
            lines.append(f"  {_label(plan)}")
        lines.append("")

    return "\n".join(lines).rstrip()
