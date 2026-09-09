"""Finding the ceilings that have crossed a threshold, and alerting once each.

The periodic half of the feature. One pass reads the enabled rules, reads the
ceilings their organizations own, claims the right to alert on each crossing,
and dispatches. ``gateway.main`` starts :func:`run_budget_alert_evaluator`
alongside the other lifespan refreshers, wherever local ceilings exist
(standalone and hosted; a hybrid gateway reserves nothing locally).

**State, not events.** A pass asks "which ceilings are over a threshold now",
not "which crossed since last time". That is what lets the same code answer the
warning and the refusal, and it is why a ceiling that crossed while its
destination was misconfigured still alerts once the destination is fixed. The
cost is a crossing reported within one interval rather than instantly.

**Utilization counts holds.** Every axis is ``(current + reserved) / cap``,
matching the gate in ``services/scoped_budget_service.reserve``. Measuring
committed spend alone would report 90 percent on a ceiling that is already
refusing requests.

**Claim before send.** :func:`_claim` inserts the ``alert_deliveries`` row and
commits it *before* dispatching, so the row is visible to sibling workers for
the whole length of a slow HTTP send. The insert is the lock, and an
``IntegrityError`` on it is the ordinary outcome on N-1 of N workers rather than
an error worth logging.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Final

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.metrics import record_alert_delivery
from gateway.models.entities import AlertDelivery, AlertRule, Budget, ScopedBudget
from gateway.services.alerts.dispatcher import send_alert
from gateway.services.secret_box import SecretDecryptionError, decrypt_secret

# Matches the cadence of the other lifespan refreshers. A budget warning is not
# a page: a minute of latency on a signal about a cap that took hours to
# approach is invisible.
ALERT_EVALUATION_INTERVAL_SECONDS: Final = 60.0

# Kinds of alert, and the dedupe key's ``kind`` column. Strings rather than an
# enum column, per `entities.AlertDelivery`.
KIND_WARNING: Final = "warning"
KIND_EXCEEDED: Final = "exceeded"

# How long a delivery row is kept after it is written. Long enough that an
# operator can still see why an alert did or did not arrive, short enough that
# the ledger does not grow without bound on a deployment with monthly periods.
DELIVERY_RETENTION_DAYS: Final = 90

# The purge filters on an unindexed ``created_at``, so it rides the loop on a
# much slower schedule than the evaluation itself.
PURGE_INTERVAL_SECONDS: Final = 3600.0


@dataclass(frozen=True)
class _Crossing:
    """One ceiling that is over a threshold, and enough of it to write a message."""

    scoped_budget_id: str
    scope_type: str
    budget_name: str | None
    period_start: datetime | None
    axis: str
    used: str
    cap: str
    percent: int


def _percent(used: Decimal, cap: Decimal) -> int:
    """Utilization as a whole percent, floored, saturating at 999.

    Floored so 79.9 percent does not fire an 80 percent warning; clamped because
    a cap lowered under an already-spent ceiling gives an arbitrary ratio.
    """
    if cap <= 0:
        return 0
    return min(int(used * 100 // cap), 999)


def _worst_axis(ceiling: ScopedBudget, budget: Budget) -> tuple[str, Decimal, Decimal, int] | None:
    """The most-consumed capped axis of one ceiling, or None if nothing is capped.

    A request has to pass every cap, so the alert reports whichever axis is
    closest to refusing one. The others are visible in the dashboard.
    """
    axes: list[tuple[str, Decimal, Decimal | None]] = [
        (
            "spend",
            Decimal(ceiling.current_spend) + Decimal(ceiling.reserved_spend),
            Decimal(budget.max_budget) if budget.max_budget is not None else None,
        ),
        (
            "tokens",
            Decimal(ceiling.current_tokens + ceiling.reserved_tokens),
            Decimal(budget.token_limit) if budget.token_limit is not None else None,
        ),
        (
            "requests",
            Decimal(ceiling.current_requests + ceiling.reserved_requests),
            Decimal(budget.request_limit) if budget.request_limit is not None else None,
        ),
    ]
    scored = [(axis, used, cap, _percent(used, cap)) for axis, used, cap in axes if cap is not None and cap > 0]
    if not scored:
        return None
    return max(scored, key=lambda item: item[3])


def _format_amount(axis: str, value: Decimal) -> str:
    """Render one axis' figure the way its unit reads.

    A Decimal would otherwise render a token count of ``1000`` as ``1000.000000``.
    """
    if axis == "spend":
        return f"${value:.2f}"
    return f"{int(value):,}"


def _message(rule_name: str, crossing: _Crossing, kind: str) -> tuple[str, str]:
    """The title and body of one alert.

    Plain text with no markup: the same body goes to Slack, a pager and a JSON
    webhook, and Apprise does not translate one vendor's markup into another's.
    """
    subject = crossing.budget_name or f"{crossing.scope_type} ceiling {crossing.scoped_budget_id}"
    if kind == KIND_EXCEEDED:
        title = f"Budget exceeded: {subject}"
        opening = f"'{subject}' has reached its {crossing.axis} limit and is now refusing requests."
    else:
        title = f"Budget warning: {subject} at {crossing.percent}%"
        opening = f"'{subject}' has used {crossing.percent}% of its {crossing.axis} limit."
    period = (
        f"Current period started {crossing.period_start.isoformat()}."
        if crossing.period_start is not None
        else "This ceiling has no period and does not reset on its own."
    )
    body = (
        f"{opening}\n\n"
        f"Used: {crossing.used} of {crossing.cap} ({crossing.axis})\n"
        f"Scope: {crossing.scope_type}\n"
        f"{period}\n\n"
        f"Alert rule: {rule_name}"
    )
    return title, body


async def _load_rules(db: AsyncSession) -> list[AlertRule]:
    """Every enabled rule, across organizations.

    One query for the whole deployment rather than one per organization: the
    common case is a handful of rows, and a deployment with none returns after
    exactly one cheap indexed read per tick, which is what makes this worker
    free for everybody who has not configured an alert.
    """
    rows = (await db.execute(select(AlertRule).where(AlertRule.enabled.is_(True)))).scalars()
    return list(rows)


async def _load_ceilings(db: AsyncSession, organization_ids: set[uuid.UUID]) -> list[tuple[ScopedBudget, Budget]]:
    """The ceilings owned by these organizations, with the budget each names.

    The join is what makes a rule organization-scoped: a ``scoped_budgets`` row
    carries no tenant of its own, so its owner is the ``organization_id`` on the
    ``budgets`` row it names. A NULL one is the deployment's own and is excluded
    by the ``IN`` (see `entities.AlertRule`).

    Only ceilings that cap something are returned: one that caps no axis can
    never cross a threshold.
    """
    if not organization_ids:
        return []
    stmt = (
        select(ScopedBudget, Budget)
        .join(Budget, Budget.budget_id == ScopedBudget.budget_id)
        .where(
            Budget.organization_id.in_(organization_ids),
            (Budget.max_budget.is_not(None) | Budget.token_limit.is_not(None) | Budget.request_limit.is_not(None)),
        )
    )
    return [(ceiling, budget) for ceiling, budget in (await db.execute(stmt)).all()]


def _as_utc(value: datetime | None) -> datetime | None:
    """Stamp a naive timestamp UTC, so the dedupe key is comparable either way."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


async def _claim(db: AsyncSession, rule: AlertRule, crossing: _Crossing, kind: str) -> AlertDelivery | None:
    """Take the right to send this alert, or return None if somebody else has it.

    Committed before the caller dispatches, which is the point: the row has to
    be visible to sibling workers while the send is in flight. An
    ``IntegrityError`` is the expected result on every worker but one.
    """
    delivery = AlertDelivery(
        alert_rule_id=rule.id,
        scoped_budget_id=crossing.scoped_budget_id,
        kind=kind,
        period_start=crossing.period_start,
    )
    db.add(delivery)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        return None
    return delivery


async def _record_outcome(db: AsyncSession, delivery: AlertDelivery, *, delivered: bool, detail: str | None) -> None:
    """Write back whether the send worked, best-effort.

    A failure here loses the outcome, not the dedupe: the claim is already
    committed, so the alert is still not sent twice.
    """
    delivery.delivered = delivered
    delivery.detail = detail
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.warning("Could not record the outcome of an alert delivery", exc_info=True)


def _kind_for(rule: AlertRule, percent: int) -> str | None:
    """Which alert, if any, this utilization earns from this rule.

    Exceeded wins over the warning when both apply: they are separate dedupe
    keys, so a ceiling that passes 80 and then 100 between two ticks sends only
    the refusal, which is the more urgent and strictly more informative.

    A rule with the exceeded alert off still warns at 100. The two flags are
    separable so a deployment that routes refusals through its own error
    monitoring can ask for the early warning alone, and going silent at exactly
    the moment that rule cares about would defeat it.
    """
    if percent >= 100 and rule.notify_on_exceeded:
        return KIND_EXCEEDED
    if rule.warn_at_percent is not None and percent >= rule.warn_at_percent:
        return KIND_WARNING
    return None


async def evaluate_budget_alerts(db: AsyncSession) -> int:
    """Run one pass, returning how many alerts were dispatched.

    The return value is for the tests and the log line; nothing branches on it.
    """
    rules = await _load_rules(db)
    if not rules:
        return 0

    by_organization: dict[uuid.UUID, list[AlertRule]] = {}
    for rule in rules:
        by_organization.setdefault(rule.organization_id, []).append(rule)

    ceilings = await _load_ceilings(db, set(by_organization))

    sent = 0
    for ceiling, budget in ceilings:
        organization_id = budget.organization_id
        if organization_id is None:
            continue
        scored = _worst_axis(ceiling, budget)
        if scored is None:
            continue
        axis, used, cap, percent = scored
        crossing = _Crossing(
            scoped_budget_id=ceiling.id,
            scope_type=ceiling.scope_type,
            budget_name=budget.name or ceiling.name,
            period_start=_as_utc(ceiling.period_start),
            axis=axis,
            used=_format_amount(axis, used),
            cap=_format_amount(axis, cap),
            percent=percent,
        )
        for rule in by_organization[organization_id]:
            kind = _kind_for(rule, percent)
            if kind is not None and await _dispatch(db, rule, crossing, kind):
                sent += 1
    return sent


async def _dispatch(db: AsyncSession, rule: AlertRule, crossing: _Crossing, kind: str) -> bool:
    """Claim, decrypt, send, and record one alert. True when it went out.

    Decryption happens after the claim, so a rule whose destination will not
    decrypt still consumes its dedupe key and reports the reason on the row
    rather than retrying the same broken decrypt every tick.

    Claiming first also keeps a pooled connection out of the send:
    ``create_session`` commits as it goes, so :func:`_claim`'s commit returns
    the connection before ``send_alert`` runs. Wrapping a pass in
    ``async with db.begin()`` would give that up.
    """
    delivery = await _claim(db, rule, crossing, kind)
    if delivery is None:
        return False

    try:
        destination = decrypt_secret(rule.encrypted_destination)
    except SecretDecryptionError:
        logger.warning("Alert rule %s has a destination that will not decrypt", rule.id)
        await _record_outcome(db, delivery, delivered=False, detail="The stored destination could not be decrypted")
        record_alert_delivery(kind=kind, delivered=False)
        return False

    title, body = _message(rule.name, crossing, kind)
    result = await send_alert(destination, title=title, body=body)
    await _record_outcome(db, delivery, delivered=result.delivered, detail=result.detail)
    record_alert_delivery(kind=kind, delivered=result.delivered)
    if not result.delivered:
        # The redaction, never the destination: see `dispatcher`'s docstring.
        logger.warning(
            "Alert rule %s could not deliver to %s: %s",
            rule.id,
            rule.redacted_destination,
            result.detail,
        )
    return result.delivered


async def purge_delivered_before(db: AsyncSession, *, older_than: datetime) -> int:
    """Drop delivery rows created before ``older_than``, returning the count.

    Rows outlive the periods they describe, and a deleted ceiling leaves its
    rows behind (``scoped_budget_id`` is deliberately not a foreign key), so
    retention rather than cascade is what bounds the table. Safe to run during a
    pass: a row old enough to purge belongs to a period that has long rolled.
    """
    result = await db.execute(delete(AlertDelivery).where(AlertDelivery.created_at < older_than))
    await db.commit()
    return int(getattr(result, "rowcount", 0) or 0)


async def run_budget_alert_evaluator(interval: float = ALERT_EVALUATION_INTERVAL_SECONDS) -> None:
    """Evaluate budget alerts forever. Cancelled at shutdown.

    Every error is swallowed and retried next tick, matching
    ``run_alias_refresher``: a database blip must not kill the worker, since
    nothing would restart it and the deployment would then stop alerting
    silently, which is the exact failure this feature exists to prevent.

    The retention purge rides this loop on its own slower schedule.
    """
    last_purge = 0.0
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_session() as db:
                sent = await evaluate_budget_alerts(db)
                if sent:
                    logger.info("Budget alerts dispatched: %s", sent)
                # Monotonic rather than wall clock: a clock adjustment must not
                # make the next purge either immediate or an hour late.
                now = asyncio.get_running_loop().time()
                if now - last_purge >= PURGE_INTERVAL_SECONDS:
                    last_purge = now
                    purged = await purge_delivered_before(
                        db, older_than=datetime.now(UTC) - timedelta(days=DELIVERY_RETENTION_DAYS)
                    )
                    if purged:
                        logger.info("Purged %s expired alert delivery rows", purged)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Budget alert evaluation failed; retrying in %ss", interval, exc_info=True)
