"""The budget alert evaluator: what crosses, what fires, and what fires only once.

The dedupe cases carry this file. A crossed threshold stays crossed for the rest
of the budget period and every refresher in ``gateway.main`` runs once per
worker, so "alert once" is not a property of the evaluator's control flow but of
the unique constraint on ``alert_deliveries``. That is why
``test_two_concurrent_passes_send_exactly_one`` exists and why it needs a real
PostgreSQL rather than a unit test with a fake session.

``send_alert`` is stubbed throughout. What is under test is which alerts are
produced and how many times, never whether Apprise can reach Slack.
"""

import asyncio
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gateway.models.entities import AlertDelivery, AlertRule, Budget, ScopedBudget
from gateway.models.tenancy import Organization
from gateway.repositories.tenancy import OrganizationRepository
from gateway.services.alerts.dispatcher import AlertDispatchResult
from gateway.services.alerts.evaluator import (
    KIND_EXCEEDED,
    KIND_WARNING,
    evaluate_budget_alerts,
    purge_delivered_before,
)
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.url_safety import redact_alert_destination

pytestmark = pytest.mark.asyncio

DESTINATION = "slack://xoxb-AAA/xoxb-BBB/xoxb-CCC/#alerts"
PERIOD_START = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Capture what the evaluator would have delivered, and report success."""
    captured: list[tuple[str, str]] = []

    async def _capture(destination: str, *, title: str, body: str) -> AlertDispatchResult:
        captured.append((title, body))
        return AlertDispatchResult(delivered=True)

    monkeypatch.setattr("gateway.services.alerts.evaluator.send_alert", _capture)
    return captured


@pytest.fixture
def failing_send(monkeypatch: pytest.MonkeyPatch) -> None:
    """A destination that refuses, so the delivery row's failure path is exercised."""

    async def _fail(destination: str, *, title: str, body: str) -> AlertDispatchResult:
        return AlertDispatchResult(delivered=False, detail="Delivery failed (RuntimeError)")

    monkeypatch.setattr("gateway.services.alerts.evaluator.send_alert", _fail)


async def _organization(db: AsyncSession, *, slug: str = "acme") -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _rule(
    db: AsyncSession,
    organization: Organization | None,
    *,
    name: str = "Platform Slack",
    warn_at_percent: int | None = 80,
    notify_on_exceeded: bool = True,
    enabled: bool = True,
) -> AlertRule:
    assert organization is not None
    rule = AlertRule(
        organization_id=organization.id,
        name=name,
        encrypted_destination=encrypt_secret(DESTINATION),
        redacted_destination=redact_alert_destination(DESTINATION),
        warn_at_percent=warn_at_percent,
        notify_on_exceeded=notify_on_exceeded,
        enabled=enabled,
    )
    db.add(rule)
    await db.commit()
    await db.refresh(rule)
    return rule


async def _ceiling(
    db: AsyncSession,
    organization: Organization | None,
    *,
    max_budget: Decimal | None = Decimal(100),
    token_limit: int | None = None,
    request_limit: int | None = None,
    spend: Decimal = Decimal(0),
    reserved: Decimal = Decimal(0),
    tokens: int = 0,
    requests: int = 0,
    period_start: datetime | None = PERIOD_START,
    budget_name: str = "Team cap",
) -> ScopedBudget:
    """One ceiling on a budget owned by ``organization`` (or by nobody, when None)."""
    budget = Budget(
        budget_id=str(uuid.uuid4()),
        name=budget_name,
        organization_id=organization.id if organization is not None else None,
        max_budget=max_budget,
        token_limit=token_limit,
        request_limit=request_limit,
    )
    db.add(budget)
    await db.flush()
    ceiling = ScopedBudget(
        id=str(uuid.uuid4()),
        scope_type="workspace",
        scope_id=str(uuid.uuid4()),
        budget_id=budget.budget_id,
        current_spend=spend,
        reserved_spend=reserved,
        current_tokens=tokens,
        current_requests=requests,
        period_start=period_start,
        period_end=(period_start + timedelta(days=30)) if period_start else None,
    )
    db.add(ceiling)
    await db.commit()
    await db.refresh(ceiling)
    return ceiling


async def _deliveries(db: AsyncSession) -> list[AlertDelivery]:
    rows = (await db.execute(select(AlertDelivery).order_by(AlertDelivery.created_at))).scalars().all()
    return list(rows)


# --------------------------------------------------------------------------
# What fires
# --------------------------------------------------------------------------


async def test_a_warning_fires_at_the_threshold(async_db: AsyncSession, sent: list[tuple[str, str]]) -> None:
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    await _ceiling(async_db, organization, spend=Decimal(80))

    assert await evaluate_budget_alerts(async_db) == 1
    (title, body) = sent[0]
    assert "warning" in title.lower()
    assert "80%" in title
    assert "Team cap" in title
    assert "$80.00 of $100.00" in body

    rows = await _deliveries(async_db)
    assert [row.kind for row in rows] == [KIND_WARNING]
    assert rows[0].delivered is True


async def test_nothing_fires_below_the_threshold(async_db: AsyncSession, sent: list[tuple[str, str]]) -> None:
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    await _ceiling(async_db, organization, spend=Decimal(79))

    assert await evaluate_budget_alerts(async_db) == 0
    assert sent == []
    assert await _deliveries(async_db) == []


async def test_reaching_the_cap_fires_exceeded_not_a_warning(
    async_db: AsyncSession, sent: list[tuple[str, str]]
) -> None:
    """Exceeded is the more urgent and strictly more informative of the two."""
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    await _ceiling(async_db, organization, spend=Decimal(100))

    assert await evaluate_budget_alerts(async_db) == 1
    assert "exceeded" in sent[0][0].lower()
    assert [row.kind for row in await _deliveries(async_db)] == [KIND_EXCEEDED]


async def test_a_reservation_counts_toward_utilization(
    async_db: AsyncSession, sent: list[tuple[str, str]]
) -> None:
    """Matching the gate in ``scoped_budget_service.reserve``.

    A ceiling holding 85 of a 100 cap is already refusing arrivals, so reporting
    it as 10 percent used would be a warning that never arrives in time.
    """
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    await _ceiling(async_db, organization, spend=Decimal(10), reserved=Decimal(75))

    assert await evaluate_budget_alerts(async_db) == 1
    assert "85%" in sent[0][0]


async def test_a_token_capped_budget_alerts_on_tokens(
    async_db: AsyncSession, sent: list[tuple[str, str]]
) -> None:
    """A budget need not cap dollars at all."""
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    await _ceiling(async_db, organization, max_budget=None, token_limit=1000, tokens=900)

    assert await evaluate_budget_alerts(async_db) == 1
    assert "90%" in sent[0][0]
    assert "900 of 1,000 (tokens)" in sent[0][1]


async def test_the_axis_closest_to_refusing_is_the_one_reported(
    async_db: AsyncSession, sent: list[tuple[str, str]]
) -> None:
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    await _ceiling(
        async_db,
        organization,
        max_budget=Decimal(100),
        spend=Decimal(10),
        token_limit=1000,
        tokens=950,
    )

    assert await evaluate_budget_alerts(async_db) == 1
    assert "(tokens)" in sent[0][1]


async def test_a_ceiling_capping_nothing_never_alerts(
    async_db: AsyncSession, sent: list[tuple[str, str]]
) -> None:
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    await _ceiling(async_db, organization, max_budget=None, spend=Decimal(10_000))

    assert await evaluate_budget_alerts(async_db) == 0
    assert sent == []


async def test_a_disabled_rule_sends_nothing(async_db: AsyncSession, sent: list[tuple[str, str]]) -> None:
    organization = await _organization(async_db)
    await _rule(async_db, organization, enabled=False)
    await _ceiling(async_db, organization, spend=Decimal(100))

    assert await evaluate_budget_alerts(async_db) == 0
    assert await _deliveries(async_db) == []


async def test_a_rule_that_only_wants_the_refusal_stays_quiet_at_80(
    async_db: AsyncSession, sent: list[tuple[str, str]]
) -> None:
    organization = await _organization(async_db)
    await _rule(async_db, organization, warn_at_percent=None)
    await _ceiling(async_db, organization, spend=Decimal(90))

    assert await evaluate_budget_alerts(async_db) == 0


async def test_no_rules_at_all_is_one_cheap_query(
    async_db: AsyncSession, sent: list[tuple[str, str]]
) -> None:
    """The common case for every deployment that has not configured an alert."""
    organization = await _organization(async_db)
    await _ceiling(async_db, organization, spend=Decimal(100))

    assert await evaluate_budget_alerts(async_db) == 0
    assert sent == []


# --------------------------------------------------------------------------
# Tenant scoping
# --------------------------------------------------------------------------


async def test_a_deployment_owned_budget_is_never_alerted_on(
    async_db: AsyncSession, sent: list[tuple[str, str]]
) -> None:
    """The documented gap of the organization-scoped design, asserted deliberately.

    ``Budget.organization_id`` NULL means the deployment's own, and the
    organization-scoped surface never lists, offers or repoints one. A tenant's
    rule therefore cannot reach it. Deployment-wide rules are a follow-up; this
    test is here so the gap is a decision rather than a surprise.
    """
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    await _ceiling(async_db, None, spend=Decimal(100))

    assert await evaluate_budget_alerts(async_db) == 0
    assert sent == []


async def test_a_rule_never_sees_another_organizations_ceiling(
    async_db: AsyncSession, sent: list[tuple[str, str]]
) -> None:
    first = await _organization(async_db, slug="acme")
    second = await _organization(async_db, slug="globex")
    await _rule(async_db, first)
    await _ceiling(async_db, second, spend=Decimal(100), budget_name="Their cap")

    assert await evaluate_budget_alerts(async_db) == 0
    assert sent == []


async def test_each_organizations_rule_gets_its_own_ceiling(
    async_db: AsyncSession, sent: list[tuple[str, str]]
) -> None:
    first = await _organization(async_db, slug="acme")
    second = await _organization(async_db, slug="globex")
    await _rule(async_db, first, name="Acme")
    await _rule(async_db, second, name="Globex")
    await _ceiling(async_db, first, spend=Decimal(100), budget_name="Acme cap")
    await _ceiling(async_db, second, spend=Decimal(100), budget_name="Globex cap")

    assert await evaluate_budget_alerts(async_db) == 2
    subjects = {title for title, _ in sent}
    assert any("Acme cap" in s for s in subjects)
    assert any("Globex cap" in s for s in subjects)


# --------------------------------------------------------------------------
# Dedupe: the load-bearing behavior
# --------------------------------------------------------------------------


async def test_a_second_pass_sends_nothing(async_db: AsyncSession, sent: list[tuple[str, str]]) -> None:
    """A crossed threshold stays crossed, so state alone would re-alert forever."""
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    await _ceiling(async_db, organization, spend=Decimal(85))

    assert await evaluate_budget_alerts(async_db) == 1
    assert await evaluate_budget_alerts(async_db) == 0
    assert await evaluate_budget_alerts(async_db) == 0
    assert len(sent) == 1
    assert len(await _deliveries(async_db)) == 1


async def test_crossing_from_warning_into_exceeded_sends_the_refusal_too(
    async_db: AsyncSession, sent: list[tuple[str, str]]
) -> None:
    """Separate dedupe keys, so the escalation is not suppressed by the warning."""
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    ceiling = await _ceiling(async_db, organization, spend=Decimal(85))

    assert await evaluate_budget_alerts(async_db) == 1

    ceiling.current_spend = Decimal(100)
    await async_db.commit()

    assert await evaluate_budget_alerts(async_db) == 1
    assert {row.kind for row in await _deliveries(async_db)} == {KIND_WARNING, KIND_EXCEEDED}


async def test_a_new_period_re_arms_the_alert(async_db: AsyncSession, sent: list[tuple[str, str]]) -> None:
    """``period_start`` is in the dedupe key, so a reset needs no cleanup."""
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    ceiling = await _ceiling(async_db, organization, spend=Decimal(85))

    assert await evaluate_budget_alerts(async_db) == 1

    # The period rolls: counters zero, window moves, spend climbs again.
    ceiling.period_start = PERIOD_START + timedelta(days=30)
    ceiling.period_end = PERIOD_START + timedelta(days=60)
    ceiling.current_spend = Decimal(85)
    await async_db.commit()

    assert await evaluate_budget_alerts(async_db) == 1
    assert len(sent) == 2
    assert len(await _deliveries(async_db)) == 2


async def test_a_periodless_ceiling_still_alerts_only_once(
    async_db: AsyncSession, sent: list[tuple[str, str]]
) -> None:
    """The partial unique index, not the constraint, is what dedupes a NULL period.

    PostgreSQL treats two NULL ``period_start`` values as distinct, so without
    the partial index a never-rolling ceiling would alert on every tick forever.
    """
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    await _ceiling(async_db, organization, spend=Decimal(85), period_start=None)

    assert await evaluate_budget_alerts(async_db) == 1
    assert await evaluate_budget_alerts(async_db) == 0
    assert len(await _deliveries(async_db)) == 1


async def test_two_concurrent_passes_send_exactly_one(
    async_db: AsyncSession,
    sent: list[tuple[str, str]],
    async_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The multi-worker case: N workers, one alert.

    Every refresher in ``gateway.main`` runs once per worker, so this is the
    real deployment shape rather than a hypothetical. Four independent sessions
    stand in for four workers; the claim insert is the only thing serializing
    them, which is exactly what is being asserted.

    The sessions come from ``async_session_factory`` rather than from
    ``async_db``: four coroutines sharing one session would serialize on the
    session and prove nothing about the constraint.
    """
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    await _ceiling(async_db, organization, spend=Decimal(85))

    async def _pass() -> int:
        async with async_session_factory() as db:
            return await evaluate_budget_alerts(db)

    results = await asyncio.gather(_pass(), _pass(), _pass(), _pass())

    assert sum(results) == 1, f"expected exactly one alert across concurrent passes, got {results}"
    total = (await async_db.execute(select(func.count()).select_from(AlertDelivery))).scalar_one()
    assert total == 1
    assert len(sent) == 1


# --------------------------------------------------------------------------
# Failure recording and retention
# --------------------------------------------------------------------------


async def test_a_failed_send_is_recorded_and_not_retried(
    async_db: AsyncSession, failing_send: None
) -> None:
    """The claim is committed before the send, so a failure still consumes the key.

    Deliberate: retrying a dead destination on every tick for the rest of the
    period would turn one misconfiguration into a permanent load. The row
    carries the reason so an operator can see what happened.
    """
    organization = await _organization(async_db)
    await _rule(async_db, organization)
    await _ceiling(async_db, organization, spend=Decimal(85))

    assert await evaluate_budget_alerts(async_db) == 0

    rows = await _deliveries(async_db)
    assert len(rows) == 1
    assert rows[0].delivered is False
    assert "RuntimeError" in (rows[0].detail or "")

    assert await evaluate_budget_alerts(async_db) == 0
    assert len(await _deliveries(async_db)) == 1


async def test_a_destination_that_will_not_decrypt_is_recorded(
    async_db: AsyncSession, sent: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rotated OTARI_SECRET_KEY must not crash the periodic worker."""
    organization = await _organization(async_db)
    rule = await _rule(async_db, organization)
    await _ceiling(async_db, organization, spend=Decimal(85))

    rule.encrypted_destination = "not-a-valid-fernet-token"
    await async_db.commit()

    assert await evaluate_budget_alerts(async_db) == 0
    rows = await _deliveries(async_db)
    assert len(rows) == 1
    assert rows[0].delivered is False
    assert "decrypt" in (rows[0].detail or "")
    assert sent == []


async def test_purge_drops_only_the_old_rows(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    rule = await _rule(async_db, organization)

    old = AlertDelivery(alert_rule_id=rule.id, scoped_budget_id="a", kind=KIND_WARNING)
    old.created_at = datetime.now(UTC) - timedelta(days=200)
    recent = AlertDelivery(alert_rule_id=rule.id, scoped_budget_id="b", kind=KIND_WARNING)
    async_db.add_all([old, recent])
    await async_db.commit()

    purged = await purge_delivered_before(async_db, older_than=datetime.now(UTC) - timedelta(days=90))
    assert purged == 1
    assert [row.scoped_budget_id for row in await _deliveries(async_db)] == ["b"]
