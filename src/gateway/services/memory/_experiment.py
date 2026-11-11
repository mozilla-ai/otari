"""What memory saves, measured: a share of a workspace's sessions served without it, compared with the rest.

A session is assigned from its trace id and the instant the experiment began, so
every request of one session lands in the same arm, on every replica, with
nothing stored; restarting the experiment deals the sessions afresh. A request
that names no session is in no arm: it gets memory as usual and is left out of
the comparison, since tokens per session is what is compared.

The comparison is of means per session, with a 95% interval from Welch's normal
approximation. It says nothing until each arm has ``MIN_SESSIONS``: sessions vary
too much for a handful to tell memory's effect from chance.
"""

import hashlib
import math
import statistics
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

ExperimentArm = Literal["control", "memory"]

# Sessions each arm needs before the comparison is shown as an estimate.
MIN_SESSIONS = 20
_Z_95 = 1.96
_BUCKETS = 10_000


def assign_arm(
    workspace_id: uuid.UUID, started_at: datetime | None, session_label: str | None, holdout_percent: int
) -> ExperimentArm | None:
    """The arm a session is in, or None while no experiment runs or the request names no session."""
    if holdout_percent <= 0 or started_at is None or not session_label:
        return None
    material = "\x00".join((str(workspace_id), started_at.isoformat(), session_label))
    bucket = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") % _BUCKETS
    return "control" if bucket < holdout_percent * (_BUCKETS // 100) else "memory"


@dataclass(frozen=True)
class SessionUsage:
    """What one session's requests used, its own model calls only: memory's side calls are counted apart."""

    requests: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost: float | None

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ArmSummary:
    """One arm's sessions, by their mean and median."""

    sessions: int
    requests_per_session: float
    tokens_per_session: float
    median_tokens_per_session: float
    input_tokens_per_session: float
    output_tokens_per_session: float
    cache_read_tokens_per_session: float
    # None where some session's cost is unknown, as for a model with no price.
    cost_per_session: float | None


@dataclass(frozen=True)
class Difference:
    """The memory arm's mean less the control arm's, with a 95% interval; negative is a saving."""

    absolute: float
    relative: float | None
    low: float
    high: float

    @property
    def conclusive(self) -> bool:
        """Whether the interval excludes no difference at all."""
        return self.high < 0 or self.low > 0


def summarize(sessions: Sequence[SessionUsage]) -> ArmSummary:
    count = len(sessions)
    if count == 0:
        return ArmSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)
    costs = [session.cost for session in sessions]
    return ArmSummary(
        sessions=count,
        requests_per_session=statistics.fmean(session.requests for session in sessions),
        tokens_per_session=statistics.fmean(session.tokens for session in sessions),
        median_tokens_per_session=float(statistics.median(session.tokens for session in sessions)),
        input_tokens_per_session=statistics.fmean(session.input_tokens for session in sessions),
        output_tokens_per_session=statistics.fmean(session.output_tokens for session in sessions),
        cache_read_tokens_per_session=statistics.fmean(session.cache_read_tokens for session in sessions),
        cost_per_session=None if any(cost is None for cost in costs) else statistics.fmean(c or 0.0 for c in costs),
    )


def compare(control: Sequence[float], memory: Sequence[float]) -> Difference | None:
    """The difference of the arms' means, or None while either has fewer than ``MIN_SESSIONS``."""
    if len(control) < MIN_SESSIONS or len(memory) < MIN_SESSIONS:
        return None
    control_mean, memory_mean = statistics.fmean(control), statistics.fmean(memory)
    error = math.sqrt(statistics.variance(control) / len(control) + statistics.variance(memory) / len(memory))
    absolute = memory_mean - control_mean
    return Difference(
        absolute=absolute,
        relative=absolute / control_mean if control_mean else None,
        low=absolute - _Z_95 * error,
        high=absolute + _Z_95 * error,
    )
