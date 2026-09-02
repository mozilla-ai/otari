"""Let a budget cap tokens and requests, not dollars alone.

A ``budgets`` row has held one ceiling, ``max_budget``, since ceilings started
naming a budget (``f3a5c7e9d1b4``). Two more axes join it here, each independent
of the other two: a budget may cap dollars, tokens, requests, or any combination.

**The limits are nullable and there is no backfill.** NULL on an axis means
unbounded there, which is what every existing budget wants: it capped dollars
(or nothing) and gains no ceiling it did not have. Deliberately no "at least one
limit" constraint either, because a row with all three NULL is a named period
that admits everything, and rows like that already exist.

**Each counter-holding table gains a counter and a hold per new axis.** A limit
is reached two ways, and both enforce against their own counters:
``users.spend``/``reserved`` for a budget handed to a gateway user, and
``scoped_budgets.current_spend``/``reserved_spend`` for a tenancy ceiling that
names one. An axis with a limit but no counter on one of those paths would be a
cap that binds through one reachability and is silently ignored through the
other, which is the failure this migration exists to avoid rather than to add.

**The counters start at zero mid-window, and that is the only honest option.**
A budget partway through its period has spent tokens this window that nothing
recorded, so there is no figure to backfill from: ``usage_logs`` is per request
and not scoped to a ceiling's window. The first period roll after this upgrade
puts every axis on the same footing. Until then a token or request cap set on a
live budget measures from now, which is what an operator setting a new cap would
expect anyway.

**The ledger records the hold per axis** (``budget_reservations``,
``budget_reservation_scopes``), because the TTL sweep gives back exactly what it
finds recorded. A period roll zeroes ``current_*`` and deliberately leaves the
holds, so a token hold that nothing releases would shrink that ceiling for good.

BIGINT throughout: a monthly token allowance for one organization outgrows a
32-bit counter, and a counter has to be at least as wide as the limit it is
compared against.

Refs mozilla-ai/otari-ai#1994, mozilla-ai/otari-ai#929.

Revision ID: c9f2a6b4e8d7
Revises: b7e1c4a9d2f5
Create Date: 2026-09-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c9f2a6b4e8d7"
down_revision: str | Sequence[str] | None = "b7e1c4a9d2f5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The counter and hold pair each of the two new axes needs, per table that holds
# counters. Named per table because the ledger spells the same amounts
# ``*_estimate`` (the per-user leg) and ``*_amount`` (one scoped line), matching
# the money column already beside them.
_COUNTER_COLUMNS: dict[str, tuple[str, ...]] = {
    "users": ("current_tokens", "reserved_tokens", "current_requests", "reserved_requests"),
    "scoped_budgets": ("current_tokens", "reserved_tokens", "current_requests", "reserved_requests"),
    "budget_reservations": ("token_estimate", "request_estimate"),
    "budget_reservation_scopes": ("token_amount", "request_amount"),
}


def upgrade() -> None:
    op.add_column("budgets", sa.Column("token_limit", sa.BigInteger(), nullable=True))
    op.add_column("budgets", sa.Column("request_limit", sa.BigInteger(), nullable=True))
    for table, columns in _COUNTER_COLUMNS.items():
        for column in columns:
            # ``server_default`` as well as NOT NULL: existing rows need a value,
            # and so does every INSERT written before these columns existed.
            op.add_column(table, sa.Column(column, sa.BigInteger(), nullable=False, server_default="0"))


def downgrade() -> None:
    for table, columns in reversed(list(_COUNTER_COLUMNS.items())):
        for column in reversed(columns):
            op.drop_column(table, column)
    op.drop_column("budgets", "request_limit")
    op.drop_column("budgets", "token_limit")
