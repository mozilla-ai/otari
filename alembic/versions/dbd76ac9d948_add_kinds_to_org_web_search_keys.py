"""Give organization web search keys a kind, an endpoint and options, and one default per kind.

A key is now a ``search`` key or a ``fetch`` key, fixed when it is created, and
every existing key is a search key. A key may carry its own ``api_base`` and
provider ``options``, and a provider that takes no key stores none, so
``encrypted_api_key`` becomes nullable. Names are unique per organization,
kind, provider and name.

An organization had one default per provider; it now has one per kind. Where
it had live defaults on several providers, the one a search tried first stays
the default (Tavily before Brave) and the others stop being defaults. A
workspace that turned the kept default off searched with the next default it
had not turned off, so that key is pinned for it, unless it already pinned a
live key; a pin it left on an archived key is cleared, as a new pin clears
one. Whether a key can be decrypted is not something a migration can see, so
an organization whose kept default cannot be decrypted searches with its
oldest usable key, where it searched with the next default.

The downgrade deletes fetch keys and keys stored without a key, which the
previous schema cannot hold, with their workspace overrides.

Revision ID: dbd76ac9d948
Revises: ae8d42792fd8
Create Date: 2026-10-21
"""

import uuid
from collections import defaultdict
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "dbd76ac9d948"
down_revision: str | Sequence[str] | None = "ae8d42792fd8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "org_web_search_keys"
_OVERRIDES = "workspace_web_search_key_overrides"
_LIVE_DEFAULT = "is_org_default AND archived_at IS NULL"
_ORGANIZATION_INDEX = "ix_org_web_search_keys_organization_id"
_OLD_DEFAULT_INDEX = "uq_org_web_search_keys_org_default"
_NEW_DEFAULT_INDEX = "uq_org_web_search_keys_org_kind_default"
_OLD_NAME_RULE = "uq_org_web_search_keys_org_provider_name"
_NEW_NAME_RULE = "uq_org_web_search_keys_org_kind_provider_name"
_KIND_RULE = "ck_org_web_search_keys_kind"
# The order a search tried an organization's defaults in, spelled out so this
# revision keeps its meaning when the application's providers change.
_DEFAULT_RANK = {"tavily": 0, "brave": 1}

_keys = sa.table(
    _TABLE,
    sa.column("id", sa.Uuid()),
    sa.column("organization_id", sa.Uuid()),
    sa.column("kind", sa.String()),
    sa.column("provider", sa.String()),
    sa.column("encrypted_api_key", sa.Text()),
    sa.column("is_org_default", sa.Boolean()),
    sa.column("archived_at", sa.DateTime(timezone=True)),
    sa.column("created_at", sa.DateTime(timezone=True)),
)
_overrides = sa.table(
    _OVERRIDES,
    sa.column("id", sa.Uuid()),
    sa.column("workspace_id", sa.Uuid()),
    sa.column("organization_id", sa.Uuid()),
    sa.column("org_web_search_key_id", sa.Uuid()),
    sa.column("is_default", sa.Boolean()),
    sa.column("disabled", sa.Boolean()),
)


def _keys_table(*, upgraded: bool) -> sa.Table:
    """``org_web_search_keys`` as it stands when its batch rebuild starts, for SQLite.

    ``upgraded`` is the shape this revision gives it; otherwise the new columns
    are added and the rest is as before. Spelled out rather than reflected, as
    ``d8b2f6a4c0e9`` does for the price tables. The indexes are dropped before
    the rebuild and created after it.
    """
    name_rule = ("organization_id", "kind", "provider", "name") if upgraded else ("organization_id", "provider", "name")
    return sa.Table(
        _TABLE,
        sa.MetaData(),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("encrypted_api_key", sa.Text(), nullable=upgraded),
        sa.Column("last4", sa.String(length=8), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_org_default", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("kind", sa.String(length=16), server_default="search", nullable=False),
        sa.Column("api_base", sa.String(length=1024), nullable=True),
        sa.Column("options", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["organization_id"], ["organization.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(*name_rule, name=_NEW_NAME_RULE if upgraded else _OLD_NAME_RULE),
        sa.UniqueConstraint("organization_id", "id", name="uq_org_web_search_keys_org_id"),
        *([sa.CheckConstraint("kind IN ('search', 'fetch')", name=_KIND_RULE)] if upgraded else []),
    )


def _keep_one_default_per_organization(bind: sa.Connection) -> None:
    """Leave each organization one default, pinning the next one where a workspace turned the kept one off."""
    defaults: dict[uuid.UUID, list[sa.Row]] = defaultdict(list)
    for row in bind.execute(
        sa.select(_keys.c.id, _keys.c.organization_id, _keys.c.provider, _keys.c.created_at).where(
            _keys.c.is_org_default.is_(True), _keys.c.archived_at.is_(None)
        )
    ):
        defaults[row.organization_id].append(row)
    for organization_id, rows in defaults.items():
        if len(rows) < 2:
            continue
        ordered = sorted(
            rows, key=lambda row: (_DEFAULT_RANK.get(row.provider, len(_DEFAULT_RANK)), row.created_at, str(row.id))
        )
        _pin_the_next_default(bind, organization_id, [row.id for row in ordered])
        bind.execute(
            sa.update(_keys).where(_keys.c.id.in_([row.id for row in ordered[1:]])).values(is_org_default=False)
        )


def _pin_the_next_default(bind: sa.Connection, organization_id: uuid.UUID, ordered: list[uuid.UUID]) -> None:
    """Pin, for each workspace that turned the first default off, the next default it kept on."""
    live = set(
        bind.execute(
            sa.select(_keys.c.id).where(_keys.c.organization_id == organization_id, _keys.c.archived_at.is_(None))
        ).scalars()
    )
    by_workspace: dict[uuid.UUID, dict[uuid.UUID, sa.Row]] = defaultdict(dict)
    for row in bind.execute(sa.select(_overrides).where(_overrides.c.organization_id == organization_id)):
        by_workspace[row.workspace_id][row.org_web_search_key_id] = row
    for workspace_id, overrides in by_workspace.items():
        first = overrides.get(ordered[0])
        if first is None or not first.disabled:
            continue
        if any(row.is_default and key_id in live for key_id, row in overrides.items()):
            continue
        chosen = next(
            (key_id for key_id in ordered[1:] if key_id not in overrides or not overrides[key_id].disabled), None
        )
        if chosen is None:
            continue
        bind.execute(
            sa.update(_overrides)
            .where(_overrides.c.workspace_id == workspace_id, _overrides.c.is_default.is_(True))
            .values(is_default=False)
        )
        if chosen in overrides:
            bind.execute(sa.update(_overrides).where(_overrides.c.id == overrides[chosen].id).values(is_default=True))
        else:
            bind.execute(
                sa.insert(_overrides).values(
                    id=uuid.uuid4(),
                    workspace_id=workspace_id,
                    organization_id=organization_id,
                    org_web_search_key_id=chosen,
                    is_default=True,
                    disabled=False,
                )
            )


def _create_indexes(*columns: str, default_index: str) -> None:
    op.create_index(_ORGANIZATION_INDEX, _TABLE, ["organization_id"])
    op.create_index(
        default_index,
        _TABLE,
        ["organization_id", *columns],
        unique=True,
        postgresql_where=sa.text(_LIVE_DEFAULT),
        sqlite_where=sa.text(_LIVE_DEFAULT),
    )


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(_TABLE, sa.Column("kind", sa.String(length=16), server_default="search", nullable=False))
    op.add_column(_TABLE, sa.Column("api_base", sa.String(length=1024), nullable=True))
    op.add_column(_TABLE, sa.Column("options", sa.JSON(), nullable=True))
    _keep_one_default_per_organization(op.get_bind())

    op.drop_index(_OLD_DEFAULT_INDEX, table_name=_TABLE)
    op.drop_index(_ORGANIZATION_INDEX, table_name=_TABLE)
    with op.batch_alter_table(_TABLE, copy_from=_keys_table(upgraded=False)) as batch:
        batch.alter_column("encrypted_api_key", existing_type=sa.Text(), nullable=True)
        batch.drop_constraint(_OLD_NAME_RULE, type_="unique")
        batch.create_unique_constraint(_NEW_NAME_RULE, ["organization_id", "kind", "provider", "name"])
        batch.create_check_constraint(_KIND_RULE, "kind IN ('search', 'fetch')")
    _create_indexes("kind", default_index=_NEW_DEFAULT_INDEX)


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    unheld = sa.select(_keys.c.id).where(sa.or_(_keys.c.kind != "search", _keys.c.encrypted_api_key.is_(None)))
    # Explicitly, since a migration's SQLite connection does not enforce the cascade.
    bind.execute(sa.delete(_overrides).where(_overrides.c.org_web_search_key_id.in_(unheld)))
    bind.execute(sa.delete(_keys).where(_keys.c.id.in_(unheld)))

    op.drop_index(_NEW_DEFAULT_INDEX, table_name=_TABLE)
    op.drop_index(_ORGANIZATION_INDEX, table_name=_TABLE)
    with op.batch_alter_table(_TABLE, copy_from=_keys_table(upgraded=True)) as batch:
        batch.drop_constraint(_KIND_RULE, type_="check")
        batch.drop_constraint(_NEW_NAME_RULE, type_="unique")
        batch.create_unique_constraint(_OLD_NAME_RULE, ["organization_id", "provider", "name"])
        batch.alter_column("encrypted_api_key", existing_type=sa.Text(), nullable=False)
        batch.drop_column("options")
        batch.drop_column("api_base")
        batch.drop_column("kind")
    _create_indexes("provider", default_index=_OLD_DEFAULT_INDEX)
