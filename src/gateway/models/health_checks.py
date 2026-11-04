"""ORM tables for the scheduled health checks an operator runs against the deployment's own providers."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, CheckConstraint, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime

SETTINGS_ROW_ID = 1


class HealthCheckSettings(Base):
    """The one row that says whether, how often and how the checks run.

    ``next_run_at`` and ``run_requested`` are also the claim: every replica's
    worker polls this row, and a run starts only for the replica whose
    conditional update moved ``next_run_at`` forward.
    """

    __tablename__ = "health_check_settings"
    __table_args__ = (CheckConstraint(f"id = {SETTINGS_ROW_ID}", name="ck_health_check_settings_one_row"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=SETTINGS_ROW_ID)
    # The schedule's switch. Whether the feature runs at all is ``health_checks_enabled``, a startup setting.
    schedule_enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    interval_minutes: Mapped[int] = mapped_column(Integer, default=60, server_default="60")
    prompt: Mapped[str] = mapped_column(String, default="Reply with the single word: pong")
    max_tokens: Mapped[int] = mapped_column(Integer, default=16, server_default="16")
    # Reasoning models spend hidden tokens before answering, so a small cap can end the turn with no text.
    reasoning_max_tokens: Mapped[int] = mapped_column(Integer, default=256, server_default="256")
    # Reasoning models are most of a run's cost: they join every Nth run, and 0 leaves them out.
    reasoning_every_runs: Mapped[int] = mapped_column(Integer, default=6, server_default="6")
    runs_started: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    next_run_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    run_requested: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    alert_emails: Mapped[list[str]] = mapped_column(JSON, default=list)
    # The selectors the recipients were last told are failing. A run emails only what differs from it.
    alerted_selectors: Mapped[list[str]] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=func.now(),
    )


class HealthCheckRun(Base):
    """One pass over every offering. ``finished_at`` is null while it runs, or when its replica died mid-run."""

    __tablename__ = "health_check_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    started_at: Mapped[datetime] = mapped_column(UtcDateTime(), index=True, default=lambda: datetime.now(UTC))
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), default=None)
    included_reasoning: Mapped[bool] = mapped_column(Boolean, default=False)
    # The reasoning models this run left for the next run that includes them.
    deferred_selectors: Mapped[list[str]] = mapped_column(JSON, default=list)
    total: Mapped[int] = mapped_column(Integer, default=0)
    passed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)


class HealthCheckResult(Base):
    """What one offering answered in one run."""

    __tablename__ = "health_check_results"
    __table_args__ = (CheckConstraint("status IN ('pass', 'fail', 'skip')", name="ck_health_check_results_status"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("health_check_runs.id", ondelete="CASCADE"), index=True)
    # The ``instance:model`` selector that was sent, and the instance it names.
    selector: Mapped[str] = mapped_column(String)
    provider: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    status_code: Mapped[int | None] = mapped_column(Integer, default=None)
    error: Mapped[str | None] = mapped_column(String, default=None)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    checked_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))


class HealthCheckRecovery(Base):
    """One outage that ended: a model that failed for a streak of runs and then answered.

    Kept past the results it was read from, which go after 48 hours, so how
    often and how long a model was down stays on record.
    """

    __tablename__ = "health_check_recoveries"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    selector: Mapped[str] = mapped_column(String, index=True)
    provider: Mapped[str] = mapped_column(String)
    failing_since: Mapped[datetime] = mapped_column(UtcDateTime())
    recovered_at: Mapped[datetime] = mapped_column(UtcDateTime(), index=True)
    failed_runs: Mapped[int] = mapped_column(Integer)
    # What the last failing check answered.
    status_code: Mapped[int | None] = mapped_column(Integer, default=None)
    error: Mapped[str | None] = mapped_column(String, default=None)
