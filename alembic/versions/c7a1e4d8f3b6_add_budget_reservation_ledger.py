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
  was released by nothing at all. The budget reset is not the backstop it reads
  as: it zeroes spend and leaves the hold in place, so a leak shrank the
  headroom permanently.

``budget_reservations`` gives each hold a row, a status and a TTL, and
``budget_reservation_scopes`` records what it placed on each tenancy-scoped
ceiling. The counters stay the fast path the gate reads; these rows are what
make it auditable and reclaimable.

Additive only: no backfill and no data change. A hold taken before this lands has
no row, which reads correctly as "not ledgered" and is left to the pre-existing
budget reset, exactly as it was.

Revision ID: c7a1e4d8f3b6
Revises: a7f3c9e2b481
Create Date: 2026-08-24 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7a1e4d8f3b6"
down_revision: str | Sequence[str] | None = "a7f3c9e2b481"
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
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    # The global sweep's access path. Equality on status leads so the range scan
    # on expires_at rides the same index.
    op.create_index(
        "ix_budget_reservations_status_expires_at",
        "budget_reservations",
        ["status", "expires_at"],
    )
    # The per-user reclaim's, which runs on every request that takes a hold. It
    # has to lead on user_id: given only the index above, the planner takes that
    # one and filters user_id, so one user's reclaim pays for the whole
    # deployment's backlog of expired rows (measured at 200k rows: 1415 buffer
    # hits per reserving request against a 100k-row backlog belonging to someone
    # else). Leading on user_id also serves the FK cascade, so this replaces a
    # plain index on that column rather than joining one.
    op.create_index(
        "ix_budget_reservations_user_status_expires",
        "budget_reservations",
        ["user_id", "status", "expires_at"],
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
    op.drop_index("ix_budget_reservations_user_status_expires", table_name="budget_reservations")
    op.drop_table("budget_reservations")
