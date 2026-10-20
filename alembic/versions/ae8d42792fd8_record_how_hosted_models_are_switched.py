"""Record why a hosted model is switched off, and which are served without a rate on purpose.

``disabled_reason`` is null while a row is on, ``unpriced`` while it is held off
for want of a rate, and ``operator`` while an operator has switched it off. A
refresh serves an ``unpriced`` row once a rate appears and never touches an
``operator`` one. ``serves_unpriced`` marks a row switched on without a rate on
purpose, which a refresh does not hold for one.

A row on with no rate under its ``provider:model`` key was switched on that
way, by the move of a provider that serves everything it lists or by an
operator, so it is backfilled as ``serves_unpriced``.

Rows already off carry no record of why, so the backfill reads it from the
timestamps. A model offered at a rate has that rate stored before its row, so a
row whose first rate was stored after the row itself was offered unpriced and
priced later, by a refresh or by hand, which until now left it off. Such a row
is ``unpriced`` unless the row changed after its first rate arrived, and so is a
row with no rate at all that has not changed since it was offered. Every other
row off is ``operator``, the reading that never serves a model on its own: a
row changed after the offer may have been switched off by an operator or by the
health checks, which nothing recorded. The migration logs those rows so an
operator can switch on any that were only waiting for a price.

Revision ID: ae8d42792fd8
Revises: c0e83c6fe801
Create Date: 2026-10-19
"""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision: str = "ae8d42792fd8"
down_revision: str | Sequence[str] | None = "c0e83c6fe801"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "hosted_provider_models"
_COLUMN = "disabled_reason"
_SERVES_UNPRICED = "serves_unpriced"
_UNPRICED = "unpriced"
_OPERATOR = "operator"
# Enough to act on from a deploy log without flooding it.
_LOGGED_ROWS = 50

_log = logging.getLogger("alembic.runtime.migration")

_models = sa.table(
    _TABLE,
    sa.column("id", sa.Uuid()),
    sa.column("provider", sa.String()),
    sa.column("model", sa.String()),
    sa.column("enabled", sa.Boolean()),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
    sa.column(_COLUMN, sa.String()),
    sa.column(_SERVES_UNPRICED, sa.Boolean()),
)
_pricing = sa.table(
    "model_pricing",
    sa.column("model_key", sa.String()),
    sa.column("created_at", sa.DateTime(timezone=True)),
)


def _utc(value: datetime) -> datetime:
    """SQLite hands timestamps back naive; every one these tables hold was written in UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _utc_or_none(value: datetime | None) -> datetime | None:
    return None if value is None else _utc(value)


def disabled_reason(created_at: datetime, updated_at: datetime | None, first_priced_at: datetime | None) -> str:
    """Why a row already off is off, as far as its timestamps can tell."""
    if first_priced_at is None:
        return _UNPRICED if updated_at is None else _OPERATOR
    priced_later = first_priced_at > created_at
    untouched_since = updated_at is None or updated_at <= first_priced_at
    return _UNPRICED if priced_later and untouched_since else _OPERATOR


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(_TABLE, sa.Column(_COLUMN, sa.String(length=16), nullable=True))
    op.add_column(_TABLE, sa.Column(_SERVES_UNPRICED, sa.Boolean(), nullable=False, server_default=sa.false()))
    connection = op.get_bind()
    priced = sa.exists().where(_pricing.c.model_key == _models.c.provider + ":" + _models.c.model)
    connection.execute(sa.update(_models).where(_models.c.enabled.is_(True), ~priced).values({_SERVES_UNPRICED: True}))
    rows = connection.execute(
        sa.select(_models.c.id, _models.c.provider, _models.c.model, _models.c.created_at, _models.c.updated_at).where(
            _models.c.enabled.is_(False)
        )
    ).all()
    if not rows:
        return
    keys = {f"{row.provider}:{row.model}" for row in rows}
    first_priced: dict[str, datetime] = {
        key: priced_at
        for key, priced_at in connection.execute(
            sa.select(_pricing.c.model_key, sa.func.min(_pricing.c.created_at))
            .where(_pricing.c.model_key.in_(keys))
            .group_by(_pricing.c.model_key)
        ).all()
    }
    by_reason: dict[str, list[object]] = {_UNPRICED: [], _OPERATOR: []}
    kept_off: list[str] = []
    for row in rows:
        key = f"{row.provider}:{row.model}"
        reason = disabled_reason(
            _utc(row.created_at), _utc_or_none(row.updated_at), _utc_or_none(first_priced.get(key))
        )
        by_reason[reason].append(row.id)
        if reason == _OPERATOR:
            kept_off.append(key)
    for reason, ids in by_reason.items():
        if ids:
            connection.execute(sa.update(_models).where(_models.c.id.in_(ids)).values({_COLUMN: reason}))
    _log.info(
        "hosted models off: %d held for a price, %d kept off as an operator's choice",
        len(by_reason[_UNPRICED]),
        len(kept_off),
    )
    if kept_off:
        shown = sorted(kept_off)[:_LOGGED_ROWS]
        _log.info(
            "kept off, switch on any that only waited for a price: %s%s",
            ", ".join(shown),
            f" and {len(kept_off) - len(shown)} more" if len(kept_off) > len(shown) else "",
        )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column(_TABLE, _SERVES_UNPRICED)
    op.drop_column(_TABLE, _COLUMN)
