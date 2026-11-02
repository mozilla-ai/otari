"""Set organization web search defaults and pins so each workspace keeps the key it uses.

A workspace's key of a kind is the one it pinned, else its organization's default,
else none, and then the deployment's own tool serves it. Data only:

- An organization with live keys of a kind and no default gets its oldest live key
  as the default: the key its workspaces without a pin or a turned-off key use.
- A workspace that turned the default off and pinned no live key gets the oldest
  live key it kept on pinned, the one it searches with. A pin it left on an
  archived key of that kind is cleared first.
- Whether a key decrypts is not visible here. Where the key a workspace relies on
  (its pin, or the default, including an oldest key made the default here) cannot
  be decrypted, a workspace that searched with the next key uses the deployment's
  tool.

The downgrade leaves the defaults and pins in place: they are valid rows either way.

Revision ID: 90b7e153c41f
Revises: a3d5f7b9c1e4
Create Date: 2026-10-29
"""

import uuid
from collections import defaultdict
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "90b7e153c41f"
down_revision: str | Sequence[str] | None = "a3d5f7b9c1e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_keys = sa.table(
    "org_web_search_keys",
    sa.column("id", sa.Uuid()),
    sa.column("organization_id", sa.Uuid()),
    sa.column("kind", sa.String()),
    sa.column("is_org_default", sa.Boolean()),
    sa.column("archived_at", sa.DateTime(timezone=True)),
    sa.column("created_at", sa.DateTime(timezone=True)),
)
_overrides = sa.table(
    "workspace_web_search_key_overrides",
    sa.column("id", sa.Uuid()),
    sa.column("workspace_id", sa.Uuid()),
    sa.column("organization_id", sa.Uuid()),
    sa.column("org_web_search_key_id", sa.Uuid()),
    sa.column("is_default", sa.Boolean()),
    sa.column("disabled", sa.Boolean()),
)


def _keys_by_organization_and_kind(bind: sa.Connection) -> dict[tuple[uuid.UUID, str], list[sa.Row]]:
    """Every key, grouped by organization and kind, oldest first."""
    groups: dict[tuple[uuid.UUID, str], list[sa.Row]] = defaultdict(list)
    for row in bind.execute(
        sa.select(
            _keys.c.id, _keys.c.organization_id, _keys.c.kind, _keys.c.is_org_default, _keys.c.archived_at
        ).order_by(_keys.c.created_at, _keys.c.id)
    ):
        groups[(row.organization_id, row.kind)].append(row)
    return groups


def _pin(
    bind: sa.Connection,
    organization_id: uuid.UUID,
    workspace_id: uuid.UUID,
    key_id: uuid.UUID,
    kind_keys: list[uuid.UUID],
) -> None:
    """Make ``key_id`` the workspace's one pin among the keys of its kind."""
    bind.execute(
        sa.update(_overrides)
        .where(
            _overrides.c.workspace_id == workspace_id,
            _overrides.c.org_web_search_key_id.in_(kind_keys),
            _overrides.c.is_default.is_(True),
        )
        .values(is_default=False)
    )
    updated = bind.execute(
        sa.update(_overrides)
        .where(_overrides.c.workspace_id == workspace_id, _overrides.c.org_web_search_key_id == key_id)
        .values(is_default=True, disabled=False)
    )
    if updated.rowcount == 0:
        bind.execute(
            sa.insert(_overrides).values(
                id=uuid.uuid4(),
                workspace_id=workspace_id,
                organization_id=organization_id,
                org_web_search_key_id=key_id,
                is_default=True,
                disabled=False,
            )
        )


def _keep_the_workspaces_keys(bind: sa.Connection, organization_id: uuid.UUID, keys: list[sa.Row]) -> None:
    """Set the default and the pins that keep one organization's workspaces on the keys of one kind they use."""
    live = [row.id for row in keys if row.archived_at is None]
    if not live:
        return
    default = next((row.id for row in keys if row.is_org_default and row.archived_at is None), None)
    if default is None:
        default = live[0]
        bind.execute(sa.update(_keys).where(_keys.c.id == default).values(is_org_default=True))
    kind_keys = [row.id for row in keys]
    by_workspace: dict[uuid.UUID, dict[uuid.UUID, sa.Row]] = defaultdict(dict)
    for row in bind.execute(sa.select(_overrides).where(_overrides.c.org_web_search_key_id.in_(kind_keys))):
        by_workspace[row.workspace_id][row.org_web_search_key_id] = row
    for workspace_id, overrides in by_workspace.items():
        kept_on = [key_id for key_id in live if key_id not in overrides or not overrides[key_id].disabled]
        if not kept_on or default in kept_on:
            continue
        if any(overrides[key_id].is_default for key_id in kept_on if key_id in overrides):
            continue
        # The default is off for this workspace and nothing live is pinned: it searched with the oldest key it kept on.
        _pin(bind, organization_id, workspace_id, kept_on[0], kind_keys)


def upgrade() -> None:
    bind = op.get_bind()
    for (organization_id, _kind), keys in _keys_by_organization_and_kind(bind).items():
        _keep_the_workspaces_keys(bind, organization_id, keys)


def downgrade() -> None:
    pass
