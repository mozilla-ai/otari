"""Let an organization own an MCP server, for every one of its workspaces.

``workspace`` gains a unique index on ``(organization_id, id)``, which composite
keys below reference to hold a row's organization to its workspace's.

``workspace_mcp_servers`` gains ``organization_id``, filled from each server's
workspace, and ``workspace_id`` becomes nullable: a server with no workspace
belongs to its organization. Organization servers' names are unique per
organization, as workspace servers' are per workspace, and ``(organization_id,
id)`` is unique so an override can pin itself to a server of its own
organization. A server whose workspace no longer exists has no organization to
take, and nothing could reach it, so it is deleted with its rows.

``workspace_mcp_server_grants`` gains ``workspace_id``, filled from the
server: a grant on an organization server applies in one workspace. Its two
unique constraints gain the column.

Two tables are new. ``workspace_mcp_server_overrides`` holds a workspace's
settings on an organization server, and ``mcp_server_aliases`` the old id of a
workspace's copy folded into one.

Every existing server stays a workspace server. The downgrade deletes the
organization servers and every row that names one, and the grants that apply
in a workspace other than their server's, none of which the previous schema
can hold.

Revision ID: 124a7328c1b3
Revises: 49e3ccf30619
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "124a7328c1b3"
down_revision: str | Sequence[str] | None = "49e3ccf30619"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SERVERS = "workspace_mcp_servers"
_GRANTS = "workspace_mcp_server_grants"
_OVERRIDES = "workspace_mcp_server_overrides"
_ALIASES = "mcp_server_aliases"

_SERVERS_WORKSPACE_INDEX = "ix_workspace_mcp_servers_workspace_id"
_SERVERS_WORKSPACE_NAME_RULE = "uq_workspace_mcp_servers_workspace_name"
_SERVERS_ORGANIZATION_NAME_RULE = "uq_workspace_mcp_servers_organization_name"
_SERVERS_ORGANIZATION_ID_RULE = "uq_workspace_mcp_servers_organization_id"
# PostgreSQL's own name for the key, which is what it would have been had the table been created with it.
_SERVERS_ORGANIZATION_KEY = "workspace_mcp_servers_organization_id_fkey"
_SERVERS_WORKSPACE_ORGANIZATION_KEY = "fk_workspace_mcp_servers_workspace_organization"
_WORKSPACE_ORGANIZATION_INDEX = "uq_workspace_organization_id"
_ORGANIZATION_SERVER = "workspace_id IS NULL"

_GRANTS_KEY_INDEX = "ix_workspace_mcp_server_grants_api_key_id"
_GRANTS_USER_INDEX = "ix_workspace_mcp_server_grants_user_id"
_GRANTS_WORKSPACE_INDEX = "ix_workspace_mcp_server_grants_workspace_id"
_GRANTS_WORKSPACE_KEY = "workspace_mcp_server_grants_workspace_id_fkey"
_GRANTS_ONE_SUBJECT = "ck_workspace_mcp_server_grants_one_subject"
_GRANTS_OLD_KEY_RULE = "uq_workspace_mcp_server_grants_server_key"
_GRANTS_OLD_USER_RULE = "uq_workspace_mcp_server_grants_server_user"
_GRANTS_NEW_KEY_RULE = "uq_workspace_mcp_server_grants_server_workspace_key"
_GRANTS_NEW_USER_RULE = "uq_workspace_mcp_server_grants_server_workspace_user"

_workspaces = sa.table("workspace", sa.column("id", sa.Uuid()), sa.column("organization_id", sa.Uuid()))
_servers = sa.table(
    _SERVERS,
    sa.column("id", sa.Uuid()),
    sa.column("workspace_id", sa.Uuid()),
    sa.column("organization_id", sa.Uuid()),
)
_grants = sa.table(
    _GRANTS,
    sa.column("server_id", sa.Uuid()),
    sa.column("workspace_id", sa.Uuid()),
)
# Every other table that names a server. A migration's SQLite connection does not enforce
# foreign keys, so a server's rows are deleted with it explicitly rather than by the cascade.
_SERVER_ROWS = (
    "workspace_mcp_server_logos",
    "mcp_user_connections",
    "mcp_shared_connections",
    "mcp_oauth_flows",
)
_requests = sa.table("mcp_server_requests", sa.column("server_id", sa.Uuid()))


def _servers_table(*, upgraded: bool) -> sa.Table:
    """``workspace_mcp_servers`` as it stands when its batch rebuild starts, for SQLite.

    ``upgraded`` is the shape this revision gives it; otherwise ``organization_id`` is
    added, nullable, and the rest is as before. Spelled out rather than reflected, as
    ``dbd76ac9d948`` does. The indexes are dropped before the rebuild and created after it.
    """
    return sa.Table(
        _SERVERS,
        sa.MetaData(),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=upgraded),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("encrypted_token", sa.Text(), nullable=True),
        sa.Column("purpose_hint", sa.Text(), nullable=True),
        sa.Column("allowed_tools", sa.JSON(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("health", sa.JSON(), nullable=True),
        sa.Column("everyone_tools", sa.JSON(), server_default='["*"]', nullable=False),
        sa.Column("read_tools", sa.JSON(), nullable=True),
        sa.Column("credential_mode", sa.String(), server_default="shared", nullable=False),
        sa.Column("oauth_client", sa.JSON(), nullable=True),
        sa.Column("oauth_client_issuer", sa.Text(), nullable=True),
        sa.Column("oauth_client_token_endpoint", sa.Text(), nullable=True),
        sa.Column("oauth_client_revocation_endpoint", sa.Text(), nullable=True),
        sa.Column("attach_by_default", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("read_tools_hinted", sa.JSON(), nullable=True),
        sa.Column("organization_id", sa.Uuid(), nullable=not upgraded),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("workspace_id", "name", name=_SERVERS_WORKSPACE_NAME_RULE),
        *(
            [
                sa.ForeignKeyConstraint(
                    ["organization_id"], ["organization.id"], ondelete="CASCADE", name=_SERVERS_ORGANIZATION_KEY
                ),
                sa.ForeignKeyConstraint(
                    ["organization_id", "workspace_id"],
                    ["workspace.organization_id", "workspace.id"],
                    ondelete="CASCADE",
                    name=_SERVERS_WORKSPACE_ORGANIZATION_KEY,
                ),
                sa.UniqueConstraint("organization_id", "id", name=_SERVERS_ORGANIZATION_ID_RULE),
            ]
            if upgraded
            else []
        ),
    )


def _grants_table(*, upgraded: bool) -> sa.Table:
    """``workspace_mcp_server_grants`` as it stands when its batch rebuild starts, for SQLite.

    As ``_servers_table``: ``workspace_id`` is added, nullable, unless ``upgraded``.
    """
    key_rule = ("server_id", "workspace_id", "api_key_id") if upgraded else ("server_id", "api_key_id")
    user_rule = ("server_id", "workspace_id", "user_id") if upgraded else ("server_id", "user_id")
    return sa.Table(
        _GRANTS,
        sa.MetaData(),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("server_id", sa.Uuid(), nullable=False),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("tools", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=not upgraded),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("(api_key_id IS NULL) <> (user_id IS NULL)", name=_GRANTS_ONE_SUBJECT),
        sa.ForeignKeyConstraint(["server_id"], [f"{_SERVERS}.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.UniqueConstraint(*key_rule, name=_GRANTS_NEW_KEY_RULE if upgraded else _GRANTS_OLD_KEY_RULE),
        sa.UniqueConstraint(*user_rule, name=_GRANTS_NEW_USER_RULE if upgraded else _GRANTS_OLD_USER_RULE),
        *(
            [
                sa.ForeignKeyConstraint(
                    ["workspace_id"], ["workspace.id"], ondelete="CASCADE", name=_GRANTS_WORKSPACE_KEY
                )
            ]
            if upgraded
            else []
        ),
    )


def _delete_servers(bind: sa.Connection, which: sa.ColumnElement[bool]) -> None:
    """Delete the servers ``which`` selects, and every row that names one."""
    doomed = sa.select(_servers.c.id).where(which)
    for name in _SERVER_ROWS:
        rows = sa.table(name, sa.column("server_id", sa.Uuid()))
        bind.execute(sa.delete(rows).where(rows.c.server_id.in_(doomed)))
    bind.execute(sa.delete(_grants).where(_grants.c.server_id.in_(doomed)))
    bind.execute(sa.update(_requests).where(_requests.c.server_id.in_(doomed)).values(server_id=None))
    bind.execute(sa.delete(_servers).where(which))


def _upgrade_servers(bind: sa.Connection) -> None:
    # ``workspace`` is locked before ``workspace_mcp_servers``, the order a workspace's deletion takes them
    # in through its cascade, so the two cannot wait on each other.
    op.create_index(_WORKSPACE_ORGANIZATION_INDEX, "workspace", ["organization_id", "id"], unique=True)
    op.add_column(_SERVERS, sa.Column("organization_id", sa.Uuid(), nullable=True))
    bind.execute(
        sa.update(_servers).values(
            organization_id=sa.select(_workspaces.c.organization_id)
            .where(_workspaces.c.id == _servers.c.workspace_id)
            .scalar_subquery()
        )
    )
    _delete_servers(bind, _servers.c.organization_id.is_(None))

    op.drop_index(_SERVERS_WORKSPACE_INDEX, table_name=_SERVERS)
    with op.batch_alter_table(_SERVERS, copy_from=_servers_table(upgraded=False)) as batch:
        batch.alter_column("organization_id", existing_type=sa.Uuid(), nullable=False)
        batch.alter_column("workspace_id", existing_type=sa.Uuid(), nullable=True)
        batch.create_foreign_key(
            _SERVERS_ORGANIZATION_KEY, "organization", ["organization_id"], ["id"], ondelete="CASCADE"
        )
        batch.create_foreign_key(
            _SERVERS_WORKSPACE_ORGANIZATION_KEY,
            "workspace",
            ["organization_id", "workspace_id"],
            ["organization_id", "id"],
            ondelete="CASCADE",
        )
        batch.create_unique_constraint(_SERVERS_ORGANIZATION_ID_RULE, ["organization_id", "id"])
    op.create_index(_SERVERS_WORKSPACE_INDEX, _SERVERS, ["workspace_id"])
    op.create_index(
        _SERVERS_ORGANIZATION_NAME_RULE,
        _SERVERS,
        ["organization_id", "name"],
        unique=True,
        postgresql_where=sa.text(_ORGANIZATION_SERVER),
        sqlite_where=sa.text(_ORGANIZATION_SERVER),
    )


def _upgrade_grants(bind: sa.Connection) -> None:
    op.add_column(_GRANTS, sa.Column("workspace_id", sa.Uuid(), nullable=True))
    bind.execute(
        sa.update(_grants).values(
            workspace_id=sa.select(_servers.c.workspace_id)
            .where(_servers.c.id == _grants.c.server_id)
            .scalar_subquery()
        )
    )
    # A grant whose server is gone, which only a database that once ran without foreign keys can hold.
    bind.execute(sa.delete(_grants).where(_grants.c.workspace_id.is_(None)))

    op.drop_index(_GRANTS_KEY_INDEX, table_name=_GRANTS)
    op.drop_index(_GRANTS_USER_INDEX, table_name=_GRANTS)
    with op.batch_alter_table(_GRANTS, copy_from=_grants_table(upgraded=False)) as batch:
        batch.alter_column("workspace_id", existing_type=sa.Uuid(), nullable=False)
        batch.create_foreign_key(_GRANTS_WORKSPACE_KEY, "workspace", ["workspace_id"], ["id"], ondelete="CASCADE")
        batch.drop_constraint(_GRANTS_OLD_KEY_RULE, type_="unique")
        batch.drop_constraint(_GRANTS_OLD_USER_RULE, type_="unique")
        batch.create_unique_constraint(_GRANTS_NEW_KEY_RULE, ["server_id", "workspace_id", "api_key_id"])
        batch.create_unique_constraint(_GRANTS_NEW_USER_RULE, ["server_id", "workspace_id", "user_id"])
    op.create_index(_GRANTS_KEY_INDEX, _GRANTS, ["api_key_id"])
    op.create_index(_GRANTS_USER_INDEX, _GRANTS, ["user_id"])
    op.create_index(_GRANTS_WORKSPACE_INDEX, _GRANTS, ["workspace_id"])


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    _upgrade_servers(bind)
    _upgrade_grants(bind)

    op.create_table(
        _OVERRIDES,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("server_id", sa.Uuid(), nullable=False),
        sa.Column("organization_tools", sa.JSON(), nullable=True),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("everyone_tools", sa.JSON(), nullable=True),
        sa.Column("attach_by_default", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["organization_id", "workspace_id"], ["workspace.organization_id", "workspace.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "server_id"], [f"{_SERVERS}.organization_id", f"{_SERVERS}.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("workspace_id", "server_id", name="uq_workspace_mcp_server_overrides_workspace_server"),
    )
    op.create_index("ix_workspace_mcp_server_overrides_server_id", _OVERRIDES, ["server_id"])

    op.create_table(
        _ALIASES,
        sa.Column("old_id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("server_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("old_id"),
        sa.ForeignKeyConstraint(
            ["organization_id", "workspace_id"], ["workspace.organization_id", "workspace.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["organization_id", "server_id"], [f"{_SERVERS}.organization_id", f"{_SERVERS}.id"], ondelete="CASCADE"
        ),
    )
    op.create_index("ix_mcp_server_aliases_workspace_id", _ALIASES, ["workspace_id"])
    op.create_index("ix_mcp_server_aliases_server_id", _ALIASES, ["server_id"])


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    op.drop_index("ix_mcp_server_aliases_server_id", table_name=_ALIASES)
    op.drop_index("ix_mcp_server_aliases_workspace_id", table_name=_ALIASES)
    op.drop_table(_ALIASES)
    op.drop_index("ix_workspace_mcp_server_overrides_server_id", table_name=_OVERRIDES)
    op.drop_table(_OVERRIDES)

    # Every grant on an organization server, and any applying outside its server's workspace.
    bind.execute(
        sa.delete(_grants).where(
            ~sa.exists().where(_servers.c.id == _grants.c.server_id, _servers.c.workspace_id == _grants.c.workspace_id)
        )
    )
    op.drop_index(_GRANTS_WORKSPACE_INDEX, table_name=_GRANTS)
    op.drop_index(_GRANTS_KEY_INDEX, table_name=_GRANTS)
    op.drop_index(_GRANTS_USER_INDEX, table_name=_GRANTS)
    with op.batch_alter_table(_GRANTS, copy_from=_grants_table(upgraded=True)) as batch:
        batch.drop_constraint(_GRANTS_NEW_KEY_RULE, type_="unique")
        batch.drop_constraint(_GRANTS_NEW_USER_RULE, type_="unique")
        batch.create_unique_constraint(_GRANTS_OLD_KEY_RULE, ["server_id", "api_key_id"])
        batch.create_unique_constraint(_GRANTS_OLD_USER_RULE, ["server_id", "user_id"])
        batch.drop_constraint(_GRANTS_WORKSPACE_KEY, type_="foreignkey")
        batch.drop_column("workspace_id")
    op.create_index(_GRANTS_KEY_INDEX, _GRANTS, ["api_key_id"])
    op.create_index(_GRANTS_USER_INDEX, _GRANTS, ["user_id"])

    _delete_servers(bind, _servers.c.workspace_id.is_(None))
    op.drop_index(_SERVERS_ORGANIZATION_NAME_RULE, table_name=_SERVERS)
    op.drop_index(_SERVERS_WORKSPACE_INDEX, table_name=_SERVERS)
    with op.batch_alter_table(_SERVERS, copy_from=_servers_table(upgraded=True)) as batch:
        batch.drop_constraint(_SERVERS_WORKSPACE_ORGANIZATION_KEY, type_="foreignkey")
        batch.drop_constraint(_SERVERS_ORGANIZATION_ID_RULE, type_="unique")
        batch.drop_constraint(_SERVERS_ORGANIZATION_KEY, type_="foreignkey")
        batch.alter_column("workspace_id", existing_type=sa.Uuid(), nullable=False)
        batch.drop_column("organization_id")
    op.create_index(_SERVERS_WORKSPACE_INDEX, _SERVERS, ["workspace_id"])
    op.drop_index(_WORKSPACE_ORGANIZATION_INDEX, table_name="workspace")
