"""Add the passkey tables: a credential per authenticator, and its challenges.

Two tables rather than columns on ``user``, which is why neither could join the
additive credential migration in otari#645: a person registers one passkey per
device, and each carries its own public key, counter, transports and label.

``webauthn_credential.rp_id`` is the column worth knowing about. An
authenticator scopes what it stored to the relying-party ID it was created
under, so recording that ID per row is what lets a deployment tell a live
passkey from one orphaned by a configuration change, instead of offering both
and failing in the browser. It is also where mozilla-ai/otari-ai#1716's standing
constraint lands: migrating otari.ai users import their credentials rather than
claiming new accounts, and an imported row asserts only while that origin stays
``otari.ai``. See ``docs/access-control.md``.

``webauthn_challenge`` holds one single-use nonce per ceremony in flight. It is
a table and not a signed cookie because a challenge has to be *retired*, not
merely verified, and a cookie cannot be revoked server-side; and not a process
dictionary because a deployment runs more than one worker and a ceremony's two
calls need not land on the same one.

Both cascade from ``user``, matching ``dashboard_sessions``: deleting an
identity takes its passkeys and any ceremony it had open with it, rather than
leaving a credential that asserts to a row that is gone.

Purely additive, so there is no data step and the downgrade is a pair of drops.

Revision ID: e9b3d7f1a5c2
Revises: d8b3e5c1f7a2
Create Date: 2026-08-24 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e9b3d7f1a5c2"
down_revision: str | Sequence[str] | None = "d8b3e5c1f7a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# base64url of the 1023 bytes the WebAuthn spec caps a credential ID at. Kept in
# step with ``models.tenancy.MAX_CREDENTIAL_ID_LENGTH``; a literal here rather
# than an import, so the migration keeps describing the schema of its own moment
# even after the model moves on.
_CREDENTIAL_ID_LENGTH = 1364


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "webauthn_credential",
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("credential_id", sa.String(length=_CREDENTIAL_ID_LENGTH), nullable=False),
        # Unbounded, like ``user.hashed_password``: a COSE key carries its own
        # algorithm, and an RSA credential's key is an order of magnitude longer
        # than the EC one a platform passkey emits.
        sa.Column("public_key", sa.String(), nullable=False),
        sa.Column("rp_id", sa.String(length=255), nullable=False),
        sa.Column("sign_count", sa.Integer(), nullable=False),
        sa.Column("transports", sa.JSON(), nullable=False),
        sa.Column("backed_up", sa.Boolean(), nullable=False),
        sa.Column("aaguid", sa.String(length=64), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        # One label per identity, so a list of three passkeys never shows the
        # same name twice and a rename cannot make it.
        sa.UniqueConstraint("user_id", "name", name="uq_webauthn_credential_user_name"),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
    )
    # Unique deployment-wide rather than per identity: an assertion arrives
    # naming only this id, so it is what a usernameless sign-in resolves an
    # identity from, and two rows sharing one would make that ambiguous.
    op.create_index(
        op.f("ix_webauthn_credential_credential_id"), "webauthn_credential", ["credential_id"], unique=True
    )
    op.create_index(op.f("ix_webauthn_credential_rp_id"), "webauthn_credential", ["rp_id"], unique=False)
    op.create_index(op.f("ix_webauthn_credential_user_id"), "webauthn_credential", ["user_id"], unique=False)

    op.create_table(
        "webauthn_challenge",
        # The challenge is its own key: 32 bytes of server-chosen randomness, so
        # it is already unique, and looking a ceremony up by it is the only read
        # this table has. Stored raw rather than hashed, unlike the bearer tokens
        # elsewhere in this schema, because a challenge is a nonce that is sent
        # to the client by design.
        sa.Column("challenge", sa.String(length=255), nullable=False),
        sa.Column("ceremony", sa.String(length=32), nullable=False),
        # Null for an authentication ceremony, which begins before anybody is
        # identified; set for a registration, which is always performed from
        # inside a session.
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("challenge"),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
    )
    # Expiry is indexed because the opportunistic sweep on every ceremony
    # deletes by it, the way the dashboard-session sweep does.
    op.create_index(op.f("ix_webauthn_challenge_expires_at"), "webauthn_challenge", ["expires_at"], unique=False)
    op.create_index(op.f("ix_webauthn_challenge_user_id"), "webauthn_challenge", ["user_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_webauthn_challenge_user_id"), table_name="webauthn_challenge")
    op.drop_index(op.f("ix_webauthn_challenge_expires_at"), table_name="webauthn_challenge")
    op.drop_table("webauthn_challenge")
    op.drop_index(op.f("ix_webauthn_credential_user_id"), table_name="webauthn_credential")
    op.drop_index(op.f("ix_webauthn_credential_rp_id"), table_name="webauthn_credential")
    op.drop_index(op.f("ix_webauthn_credential_credential_id"), table_name="webauthn_credential")
    op.drop_table("webauthn_credential")
