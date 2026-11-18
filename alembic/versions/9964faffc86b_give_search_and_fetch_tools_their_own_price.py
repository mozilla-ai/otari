"""Give each search and fetch tool a price of its own.

A call on a search or fetch tool is billed at that tool's own price. A tool
with no price of its own is billed at the general Web search or Web fetch
price (the `otari:web_search` and `otari:web_fetch` rows). This migration
copies the general price onto each such tool, so that every tool keeps its
current price once only its own price counts.

For each search or fetch tool saved in the database, and for `builtin_fetch`
(the built-in fetcher), it writes:
- nothing, if the tool already has a price of its own, set under its name
  or under its older `<provider>:<name>` key;
- otherwise, the general price of its kind, as a new price starting now;
- for `builtin_fetch`, a price of 0 if there is no general Web fetch price
  either.

Tools defined in config.yml are not in the database, so this migration does
not touch them: their `price_per_call` setting prices them.

Downgrading deletes the prices this migration wrote. It has to, because an
older migration refuses to downgrade while any search or fetch price exists.

Revision ID: 9964faffc86b
Revises: 07802ec50d76
Create Date: 2026-11-03
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

import sqlalchemy as sa
from alembic import op

revision: str = "9964faffc86b"
down_revision: str | Sequence[str] | None = "07802ec50d76"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_prices = sa.table(
    "model_pricing",
    sa.column("kind", sa.String()),
    sa.column("model_key", sa.String()),
    sa.column("effective_at", sa.DateTime(timezone=True)),
    sa.column("input_price_per_million", sa.Numeric(18, 8)),
    sa.column("output_price_per_million", sa.Numeric(18, 8)),
    sa.column("pricing_tiers", sa.JSON()),
    sa.column("unit", sa.String()),
    sa.column("origin", sa.String()),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)
_tools = sa.table(
    "search_tool_credentials",
    sa.column("name", sa.String()),
    sa.column("kind", sa.String()),
    sa.column("provider", sa.String()),
)


def _spellings(model_key: str) -> list[str]:
    """A model key and the ``provider/name`` spelling a row written before keys were canonical may use."""
    provider, _, name = model_key.partition(":")
    return [model_key, f"{provider}/{name}"]


def _priced_keys(bind: sa.Connection, keys: set[tuple[str, str]], now: datetime) -> set[tuple[str, str]]:
    """The ``(kind, model_key)`` pairs among ``keys`` with a version in force.

    Asked in chunks: each pair takes two query parameters, and SQLite allows 999 in one query.
    """
    wanted = sorted(keys)
    found: set[tuple[str, str]] = set()
    for start in range(0, len(wanted), 300):
        query = sa.select(_prices.c.kind, _prices.c.model_key).where(
            sa.tuple_(_prices.c.kind, _prices.c.model_key).in_(wanted[start : start + 300]),
            _prices.c.effective_at <= now,
        )
        found |= {(kind, key) for kind, key in bind.execute(query)}
    return found


def _rate_in_force(bind: sa.Connection, model_key: str, now: datetime) -> Decimal | None:
    """The rate of ``model_key``'s newest version in force, its preferred spelling first, as settlement reads it."""
    for spelling in _spellings(model_key):
        query = (
            sa.select(_prices.c.input_price_per_million)
            .where(_prices.c.kind == "model", _prices.c.model_key == spelling, _prices.c.effective_at <= now)
            .order_by(_prices.c.effective_at.desc())
            .limit(1)
        )
        if (rate := bind.execute(query).scalar()) is not None:
            return Decimal(rate)
    return None


def upgrade() -> None:
    bind = op.get_bind()
    now = datetime.now(UTC)
    builtin = ("fetch", "builtin_fetch", "builtin")
    stored = [tuple(row) for row in bind.execute(sa.select(_tools.c.kind, _tools.c.name, _tools.c.provider))]
    tools = list(dict.fromkeys([*stored, builtin]))
    # A tool's own price: a row of its kind named after it, or its old model row in either spelling.
    own = {
        tool: [(tool[0], tool[1]), *(("model", key) for key in _spellings(f"{tool[2]}:{tool[1]}"))] for tool in tools
    }
    priced = _priced_keys(bind, {key for keys in own.values() for key in keys}, now)
    rates = {kind: _rate_in_force(bind, f"otari:web_{kind}", now) for kind in ("search", "fetch")}
    for tool in tools:
        kind, name, _provider = tool
        rate = rates[kind]
        if priced.intersection(own[tool]) or (rate is None and tool != builtin):
            continue
        bind.execute(
            _prices.insert().values(
                kind=kind,
                model_key=name,
                effective_at=now,
                input_price_per_million=rate or Decimal(0),
                output_price_per_million=Decimal(0),
                pricing_tiers=[],
                unit="requests",
                origin="migration",
                created_at=now,
                updated_at=now,
            )
        )


def downgrade() -> None:
    op.execute(_prices.delete().where(_prices.c.origin == "migration", _prices.c.kind.in_(("search", "fetch"))))
