"""Set include_raw_content on organization Tavily search keys, so they keep returning page text.

Every Tavily search on such a key asked for each page's full text. Its calls
send the key's own options, and Tavily returns no page text unless asked, so
each key, archived ones included, gets ``include_raw_content: true`` unless it
sets the option already. The organization can turn it off by editing the key.

The downgrade leaves the option in place: it is a valid option either way.

Revision ID: 8171c59fe077
Revises: d4a7b2c9e510
Create Date: 2026-11-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8171c59fe077"
down_revision: str | Sequence[str] | None = "d4a7b2c9e510"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_keys = sa.table(
    "org_web_search_keys",
    sa.column("id", sa.Uuid()),
    sa.column("kind", sa.String()),
    sa.column("provider", sa.String()),
    sa.column("options", sa.JSON()),
)


def upgrade() -> None:
    bind = op.get_bind()
    tavily_keys = bind.execute(
        sa.select(_keys.c.id, _keys.c.options).where(_keys.c.kind == "search", _keys.c.provider == "tavily")
    )
    for key_id, options in tavily_keys.all():
        options = options if isinstance(options, dict) else {}
        if "include_raw_content" not in options:
            bind.execute(
                sa.update(_keys).where(_keys.c.id == key_id).values(options={**options, "include_raw_content": True})
            )


def downgrade() -> None:
    pass
