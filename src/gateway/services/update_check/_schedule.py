"""When an install checks: one fixed time of day each, and a randomized backoff after a failure.

Installs upgrade and restart together after a release, so a check at startup
would bring them all at once. Each install checks at the time of day its id
hashes to instead, which spreads them over the whole day and does not move
when the install restarts.
"""

import random
import re
import uuid
from datetime import UTC, datetime, timedelta

DAY = timedelta(days=1)
FIRST_RETRY = timedelta(minutes=5)
LONGEST_RETRY = timedelta(hours=6)

_RELEASE = re.compile(r"v?(\d{1,6})\.(\d{1,6})\.(\d{1,6})")


def release_version(raw: str) -> tuple[int, int, int] | None:
    """The release a version string names (``v0.30.1`` or ``0.30.1``), or None for a development build."""
    match = _RELEASE.fullmatch(raw.strip())
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def daily_slot(install_id: uuid.UUID) -> timedelta:
    """The install's time of day, as an offset from midnight UTC."""
    return timedelta(seconds=install_id.int % int(DAY.total_seconds()))


def next_daily_check(install_id: uuid.UUID, now: datetime) -> datetime:
    """The first occurrence of the install's time of day after ``now``."""
    midnight = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    slot = midnight + daily_slot(install_id)
    return slot if slot > now else slot + DAY


def next_retry(
    install_id: uuid.UUID,
    now: datetime,
    failures: int,
    *,
    retry_after: timedelta | None = None,
    rng: random.Random | None = None,
) -> datetime:
    """When to try again after ``failures`` failed checks in a row.

    The wait doubles from five minutes up to six hours, scaled by a random factor
    between a half and one and a half, so installs that failed together during an
    outage do not all come back at the same moment. A server's ``Retry-After`` is
    a floor. Never later than the install's next daily check.
    """
    # Capped before it doubles, so a long outage cannot overflow the timedelta.
    base = min(FIRST_RETRY * (1 << min(max(failures - 1, 0), 10)), LONGEST_RETRY)
    wait = base * (rng.uniform(0.5, 1.5) if rng else random.uniform(0.5, 1.5))
    if retry_after is not None:
        wait = max(wait, retry_after)
    return min(now + wait, next_daily_check(install_id, now))
