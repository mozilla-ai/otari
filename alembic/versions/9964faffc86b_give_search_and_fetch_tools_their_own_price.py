"""Give each stored search and fetch tool with no price of its own the otari: price in force, and builtin_fetch 0.

A search or fetch tool is priced by its own row: a ``search`` or ``fetch`` row
named after it, or its old ``<provider>:<name>`` model row. Each stored tool,
and ``builtin_fetch``, with none in force gets the ``otari:web_search`` or
``otari:web_fetch`` price in force as its own, dated now. ``builtin_fetch``
with neither gets a price of 0, on a fresh database too. A tool declared in
config.yml is not in the database, so it gets no copy: its ``price_per_call``,
or a price set for it through the API, prices it.

The downgrade deletes the prices this revision wrote, the search and fetch rows
of origin ``migration``, which the revision that gave prices a kind refuses to
downgrade over.

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
    """The ``(kind, model_key)`` pairs among ``keys`` with a version in force."""
    query = sa.select(_prices.c.kind, _prices.c.model_key).where(
        sa.tuple_(_prices.c.kind, _prices.c.model_key).in_(sorted(keys)), _prices.c.effective_at <= now
    )
    return {(kind, key) for kind, key in bind.execute(query)}


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
