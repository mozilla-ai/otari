"""record the cached-token convention on usage_logs

Revision ID: b3d5f7a9c1e6
Revises: b3f8d1c6a4e7
Create Date: 2026-08-21 09:00:00.000000

``usage_logs`` recorded the token counts a provider reported but not which
cached-token convention they arrived under, and the two shapes are
indistinguishable from the numbers alone: under the inclusive (OpenAI) shape
the cache buckets sit inside ``prompt_tokens``, under the additive (Anthropic)
one they are extra. Repricing recovered it from ``billing_meters``, which only
answers for a row something already priced. mozilla-ai/otari#690, following
#661.

Nullable with no server default, on purpose. Every existing row is left NULL,
which reads as "not recorded" and is what keeps the meter-based recovery in
``usage_admin_service._row_cache_tokens_included`` reachable for them. A
default would have every historical row assert a convention nothing checked,
and silently mis-price the half that were the other one. No backfill for the
same reason: the meters already answer for every priced row, and a row with
neither meters nor a flag is one nothing can answer for.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b3d5f7a9c1e6"
down_revision: str | Sequence[str] | None = "b3f8d1c6a4e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("usage_logs", sa.Column("cache_tokens_in_prompt", sa.Boolean(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("usage_logs", "cache_tokens_in_prompt")
