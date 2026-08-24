"""Add the budget reservation ledger.

``users.reserved`` and ``scoped_budgets.reserved_spend`` already hold in-flight
budget, but only as counters, so a hold has no identity. Two guarantees the
gateway is supposed to make were unreachable because of that
(mozilla-ai/otari#742):

* a release could not tell whether it had already run, and a second one
  subtracted the hold again. The release expression clamps at zero, so that
  surfaced as an under-count of live holds rather than as an error, weakening
  the overspend guarantee the budget gate exists for;
* a hold leaked between reserve and settle could be seen only in aggregate, and
  cleared only by the budget's next reset (never, for a budget with no reset
  period).

``budget_reservations`` gives each hold a row, a status and a TTL, and
``budget_reservation_scopes`` records what it placed on each tenancy-scoped
ceiling. The counters stay the fast path the gate reads; these rows are what
make it auditable and reclaimable.

Additive only: no backfill and no data change. A hold taken before this lands has
no row, which reads correctly as "not ledgered" and is left to the pre-existing
budget reset, exactly as it was.

Revision ID: c7a1e4d8f3b6
Revises: d8b3f1c6a4e9
Create Date: 2026-08-24 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7a1e4d8f3b6"
down_revision: str | Sequence[str] | None = "d8b3f1c6a4e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The exact money type every budget counter already uses (mozilla-ai/otari#691).
# A hold released at the amount it was taken needs both sides written the same
# way, so a ledger line is stored as exactly as the counter it unwinds.
_COST_TYPE = sa.Numeric(18, 6)


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "budget_reservations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("estimate", _COST_TYPE, nullable=False, server_default="0"),
        sa.Column("user_reserved", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("counts_toward_budget", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    # The per-user reclaim runs on every request that reserves, so this one is on
    # the hot path rather than a reporting convenience.
    op.create_index("ix_budget_reservations_user_id", "budget_reservations", ["user_id"])
    # The sweep's access path. Equality on status leads so the range scan on
    # expires_at rides the same index.
    op.create_index(
        "ix_budget_reservations_status_expires_at",
        "budget_reservations",
        ["status", "expires_at"],
    )

    op.create_table(
        "budget_reservation_scopes",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("reservation_id", sa.String(), nullable=False),
        # Deliberately not a foreign key, following scoped_budgets' own convention:
        # a ceiling deleted mid-flight leaves an orphan line the release skips,
        # rather than forcing the delete to cascade into live holds.
        sa.Column("scoped_budget_id", sa.String(), nullable=False),
        sa.Column("amount", _COST_TYPE, nullable=False, server_default="0"),
        sa.ForeignKeyConstraint(["reservation_id"], ["budget_reservations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_budget_reservation_scopes_reservation_id",
        "budget_reservation_scopes",
        ["reservation_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_budget_reservation_scopes_reservation_id", table_name="budget_reservation_scopes")
    op.drop_table("budget_reservation_scopes")
    op.drop_index("ix_budget_reservations_status_expires_at", table_name="budget_reservations")
    op.drop_index("ix_budget_reservations_user_id", table_name="budget_reservations")
    op.drop_table("budget_reservations")
