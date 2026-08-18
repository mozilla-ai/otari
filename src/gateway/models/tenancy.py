"""Tenancy: organizations, workspaces, and the identities that belong to them.

The reconciled control plane's tenancy core, rehomed from the platform
(`otari-ai` `backend/app/models/`) as part of the M5 strangle. Five tables, one
graph: an ``organization`` owns ``workspace`` rows, and ``user`` rows join both
through ``organization_member`` and ``workspace_member``.

**Why SQLModel here and plain SQLAlchemy in `entities.py`.** SQLModel is
SQLAlchemy underneath, so these classes bind to the gateway's ``AsyncSession``
unchanged while keeping the ``Create``/``Update``/``Public`` schema layer the
routes and the generated dashboard client are built on. Converting them to
`entities.py`'s declarative style would have rewritten every endpoint contract
in the slice for no behavioral gain. `entities.py` stays as it is: the two
styles coexist deliberately, and new *gateway* tables still belong there.

**One MetaData, two styles.** `entities.py`'s ``Base`` shares
``SQLModel.metadata`` (see the note there), so Alembic, ``create_all`` and
``drop_all`` see one schema no matter which style declared a table.

Three deliberate departures from the platform's models, applied on arrival:

- **Timestamps are timezone-aware, on every engine.** The platform's mixins
  annotate ``created_at``/``updated_at`` as a bare ``datetime`` (so SQLAlchemy
  renders a naive column) while their ``default_factory`` writes an aware
  ``datetime.now(UTC)`` into it: the offset is silently dropped on the way in,
  and the value reads back as local-looking UTC. That is a latent bug, not a
  style difference, so it is fixed here rather than carried. ``timezone=True``
  alone does not fix it, which is why ``UtcDateTime`` below exists: PostgreSQL
  honors the flag and SQLite ignores it, and SQLite is what the OSS edition
  ships by default, so on that engine the departure would have been a comment
  rather than a behavior. ``tests/unit/test_tenancy_timestamps.py`` is what
  keeps it one.
- **``email`` is a plain nullable string, not ``EmailStr``.** A standalone
  operator identity is a label, not a sign-in address (M4: "local identities
  have no email"), and every reader here must already tolerate its absence, so
  the annotation says so rather than being widened at each call site.
- **Hosted-only columns are not carried, with one exception.** The reconciled
  schema is edition-invariant (the overlay contributes adapters and routers,
  never tables), so a column the hosted edition needs has to live here or
  nowhere. ``workspace.activation_classification`` and
  ``user.default_organization_id`` therefore stay, neither of them read by
  anything in this edition. The columns
  that do *not* come are the ones gated on the still-open identity decision
  (otari-ai#1716): password hashes, OAuth provider, verification tokens. They
  arrive with the flow that reads them, rather than being invented ahead of it.
  Purely hosted CRM and onboarding columns are the third case and are simply
  not part of the reconciled schema.

No ORM ``relationship()`` is declared on purpose. Lazy loading on an
``AsyncSession`` raises ``MissingGreenlet`` at the point of attribute access
rather than at the query, so this slice joins explicitly in its repositories,
exactly as the platform's own tenancy models do.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import field_validator
from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, UniqueConstraint, Uuid, func
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.types import TypeDecorator
from sqlmodel import Field, SQLModel

ORGANIZATION_MEMBER_ROLES = {"owner", "admin", "member", "viewer"}
ORGANIZATION_MEMBER_STATUSES = {"active", "invited", "suspended"}
# What a member's status may be *set* to over the API. "invited" is a valid
# stored status (the M4 backfill and the invitation flow both produce it) but
# nothing in this edition can create or accept an invitation, so offering it on
# the update request would advertise a state with no producer and no exit.
# Widening this back when invitations rehome is additive; narrowing later would
# not be.
ORGANIZATION_MEMBER_UPDATABLE_STATUSES = {"active", "suspended"}
WORKSPACE_MEMBER_ROLES = {"owner", "admin", "member", "viewer"}
WORKSPACE_MEMBER_STATUSES = {"active", "invited", "suspended"}

WorkspaceActivationClassification = Literal["eligible", "internal", "automated", "migrated", "enterprise_assisted"]

# Roles that may manage an organization or a workspace. Fixed roles are the
# settled OSS line; anything finer-grained is overlay depth.
MANAGEMENT_ROLES = frozenset({"owner", "admin"})


def _validate_membership(value: str, *, allowed: set[str], kind: str) -> str:
    if value not in allowed:
        msg = f"Invalid {kind}: {value}"
        raise ValueError(msg)
    return value


class UtcDateTime(TypeDecorator[datetime]):
    """A timestamp that reads back UTC-aware on every engine.

    ``DateTime(timezone=True)`` alone is not enough, and the gap is the whole
    reason this exists. PostgreSQL honors it and hands back an aware value;
    SQLite has no timestamp type at all, so SQLAlchemy stores an ISO string and
    the flag is a no-op, and a value written as ``datetime.now(UTC)`` reads back
    with ``tzinfo=None``. A naive datetime then serializes with no offset, and a
    browser parses an offset-less timestamp as **local** time, so every tenancy
    timestamp in the dashboard would be wrong by the deployment's UTC offset on
    the engine the OSS edition ships by default.

    Both directions are handled: an aware value is normalized to UTC before it
    is stored, so a caller in another zone cannot write a wall-clock time that
    means something else, and a naive value read back is stamped UTC, because
    UTC is what everything here writes.

    The rendered DDL is exactly ``impl``'s, so this changes no migration and
    ``compare_metadata`` stays clean.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.utcoffset() is None:
            # Refused rather than assumed. Reading a naive value back as UTC is
            # safe, because UTC is what everything here writes; writing one is
            # not, because the engines disagree about what it means. PostgreSQL
            # interprets it in the *session* time zone, so the same value lands
            # as a different instant depending on who connected, while SQLite
            # stores the wall clock as written. Silently picking one is how a
            # timestamp ends up hours off with nothing to show for it.
            msg = "A tenancy timestamp must be timezone-aware; got a naive datetime"
            raise ValueError(msg)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


def _timestamp_field(*, default: Any = None, default_factory: Any = None, column_kwargs: dict[str, Any]) -> Any:
    """Build a timezone-aware timestamp field.

    Two things are worked around here, once, instead of at five inheriting
    tables. SQLModel's ``Field`` overloads type ``sa_type`` as a *class*, while
    the type we want is an *instance* (the runtime accepts either and hands it
    straight to ``Column``). And the type has to arrive as ``sa_type`` rather
    than a ready-made ``sa_column``, because a ``Column`` instance declared on a
    mixin cannot be attached to more than one table; ``sa_type`` plus kwargs
    lets SQLModel build a fresh column per model.
    """
    if default_factory is not None:
        return Field(  # type: ignore[call-overload]
            default_factory=default_factory,
            sa_type=UtcDateTime(),
            sa_column_kwargs=column_kwargs,
        )
    return Field(  # type: ignore[call-overload]
        default=default,
        sa_type=UtcDateTime(),
        sa_column_kwargs=column_kwargs,
    )


class PrimaryKeyMixin:
    """A UUID primary key, rendered as CHAR(32) on SQLite and native on PostgreSQL."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)


class CreatedAtMixin:
    """Creation timestamp, defaulted in Python and in the database."""

    created_at: datetime = _timestamp_field(
        default_factory=lambda: datetime.now(UTC),
        column_kwargs={"server_default": func.now()},
    )


class UpdatedAtMixin:
    """Last-modification timestamp, stamped by the database on update.

    ``default=None`` and not merely a nullable annotation: without an explicit
    default the field is *required* on the pydantic side, which a table class
    hides (table models skip construction validation) and any schema inheriting
    this mixin would not.
    """

    updated_at: datetime | None = _timestamp_field(default=None, column_kwargs={"onupdate": func.now()})


# =============================================================================
# Identity
# =============================================================================


class UserBase(SQLModel):
    """Fields an identity carries on the wire."""

    email: str | None = Field(default=None, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)


class UserCreate(UserBase):
    """Everything an identity needs to exist: its wire fields plus its scope.

    Separate from ``UserBase`` because ``active_organization_id`` is NOT NULL
    and has no wire representation, so a create schema without it describes a
    row the database will refuse.
    """

    active_organization_id: uuid.UUID


class User(UserBase, PrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, table=True):
    """An identity in the reconciled control plane.

    Not to be confused with `entities.User`, the gateway's own string-keyed
    per-request spend identity. Both exist during the strangle: M4 re-parents
    the gateway's rows onto this table through the identity bridge, and the
    legacy table contracts away one milestone after parity, at which point this
    is the only ``User``.

    ``email`` is nullable and unique. PostgreSQL and SQLite both allow repeated
    NULLs in a unique index, so email-less local identities coexist without
    weakening uniqueness for the addresses that do exist.
    """

    __tablename__ = "user"

    email: str | None = Field(default=None, unique=True, index=True, max_length=255)
    # NOT NULL: every identity is always looking at exactly one organization,
    # which is what lets the tenancy routes resolve a scope from the caller
    # alone. Provisioning therefore creates the organization first.
    active_organization_id: uuid.UUID = Field(foreign_key="organization.id", index=True)
    # The organization provisioned for this identity, which never moves when the
    # active one does. Nothing in this edition reads it: a standalone deployment
    # has one organization and no way to switch, so the two always agree here.
    # It is carried because the schema is edition-invariant, and the hosted
    # edition anchors recurring offered credits to it precisely so they cannot
    # be farmed by creating or switching organizations. A column the hosted
    # edition needs has to live here or nowhere, and the overlay contributes
    # adapters and routers, never tables.
    #
    # ``SET NULL`` rather than cascade, matching the platform: deleting the
    # organization it points at forfeits that anchor rather than re-homing it to
    # another one, which would reopen the vector the column exists to close.
    default_organization_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="organization.id",
        ondelete="SET NULL",
        index=True,
    )


# =============================================================================
# Organizations
# =============================================================================


class OrganizationBase(SQLModel):
    name: str = Field(max_length=255)
    slug: str = Field(max_length=255)


class OrganizationCreate(OrganizationBase):
    pass


class OrganizationUpdate(SQLModel):
    name: str | None = Field(default=None, max_length=255)
    slug: str | None = Field(default=None, max_length=255)


class OrganizationPublic(OrganizationBase):
    id: uuid.UUID
    created_by_user_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime | None = None


class OrganizationsPublic(SQLModel):
    data: list[OrganizationPublic]
    count: int


class Organization(OrganizationBase, PrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, table=True):
    __tablename__ = "organization"
    __table_args__ = (UniqueConstraint("slug", name="uq_organization_slug"),)

    # Declared as an explicit column because this foreign key closes a cycle
    # (``user.active_organization_id`` points back here) and SQLModel's ``Field``
    # cannot name a constraint. The **name** is what matters: SQLAlchemy breaks a
    # cycle by emitting the constraint as a separate ALTER, and it can only do
    # that for a named constraint, so an anonymous one fails ``drop_all`` on
    # PostgreSQL, which the integration fixtures' teardown runs (``CompileError``
    # with ``use_alter`` set, ``CircularDependencyError`` without it). Both
    # unnamed combinations fail; both named ones pass. ``use_alter`` states the
    # same intent explicitly and is how the migration adds the constraint.
    created_by_user_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            "created_by_user_id",
            Uuid(),
            ForeignKey(
                "user.id",
                name="fk_organization_created_by_user_id",
                ondelete="SET NULL",
                use_alter=True,
            ),
            nullable=True,
            index=True,
        ),
    )


class OrganizationMembershipContextPublic(SQLModel):
    """An organization plus the caller's standing in it.

    What every tenancy page reads first: which organization it is looking at,
    and what the caller may do there.
    """

    organization_member_id: uuid.UUID
    role: str
    status: str
    organization: OrganizationPublic
    # Whether the dashboard may offer the BYO provider-keys surface. The
    # platform answers "does this org have a self-hosted gateway attached", which
    # in a standalone deployment is always yes: the deployment reading this *is*
    # that gateway. Kept on the contract, rather than dropped as a constant, so
    # the ported page reads the same field in both editions.
    byo_provider_keys_allowed: bool = False


class OrganizationMembershipContextsPublic(SQLModel):
    data: list[OrganizationMembershipContextPublic]
    count: int


class OrganizationSwitchRequest(SQLModel):
    organization_id: uuid.UUID


class OrganizationCreateRequest(SQLModel):
    name: str = Field(min_length=1, max_length=255)


class ActiveOrganizationUpdateRequest(SQLModel):
    name: str = Field(min_length=1, max_length=255)


class OrganizationMemberBase(SQLModel):
    organization_id: uuid.UUID = Field(foreign_key="organization.id", ondelete="CASCADE", index=True)
    user_id: uuid.UUID = Field(foreign_key="user.id", ondelete="CASCADE", index=True)
    role: str = Field(default="member", max_length=32)
    status: str = Field(default="active", max_length=32)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        return _validate_membership(value, allowed=ORGANIZATION_MEMBER_ROLES, kind="organization role")

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        return _validate_membership(value, allowed=ORGANIZATION_MEMBER_STATUSES, kind="organization member status")


class OrganizationMemberCreate(OrganizationMemberBase):
    pass


class OrganizationMemberUpdate(SQLModel):
    role: str | None = Field(default=None, max_length=32)
    status: str | None = Field(default=None, max_length=32)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_membership(value, allowed=ORGANIZATION_MEMBER_ROLES, kind="organization role")

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_membership(value, allowed=ORGANIZATION_MEMBER_STATUSES, kind="organization member status")


class OrganizationMemberPublic(OrganizationMemberBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None = None


class OrganizationMembersPublic(SQLModel):
    data: list[OrganizationMemberPublic]
    count: int


class ActiveOrganizationMemberPublic(SQLModel):
    """A member row joined to the identity behind it, as the roster shows it.

    Field-for-field the platform's shape, so the ported roster page is not
    rewritten around a new contract, with two consequences of the OSS line:
    ``email`` is nullable here (a local operator identity has no sign-in
    address), and ``invitation_id`` is always null until the invitation flow
    rehomes, which is what fills it.
    """

    organization_member_id: uuid.UUID | None = None
    user_id: uuid.UUID | None = None
    invitation_id: uuid.UUID | None = None
    email: str | None = None
    full_name: str | None = None
    role: str
    status: str
    created_at: datetime
    updated_at: datetime | None = None


class ActiveOrganizationMembersPublic(SQLModel):
    data: list[ActiveOrganizationMemberPublic]
    count: int


class WorkspaceAssignmentRequest(SQLModel):
    """A workspace and the role to grant in it, applied when a member is added."""

    workspace_id: uuid.UUID
    role: str = Field(default="member", max_length=32)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        return _validate_membership(value, allowed=WORKSPACE_MEMBER_ROLES, kind="workspace role")


class ActiveOrganizationMemberCreateRequest(SQLModel):
    """Add someone to the caller's organization, optionally into workspaces at once."""

    # Not ``EmailStr``: that would pull in email-validator for one field, and the
    # address is a claim handle rather than something this edition delivers to.
    # The format hint still reaches the generated client, so a form validates it.
    email: str = Field(max_length=255, schema_extra={"format": "email"})
    role: str = Field(default="member", max_length=32)
    workspace_assignments: list[WorkspaceAssignmentRequest] | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        return _validate_membership(value, allowed=ORGANIZATION_MEMBER_ROLES, kind="organization role")


class ActiveOrganizationMemberCreateResultPublic(SQLModel):
    """The outcome of adding a member.

    The platform answers ``invited`` on both its branches, because being added
    there always needs acceptance: a known address gets an ``invited``
    membership, an unknown one an emailed invitation. This edition has neither
    an invitation to send nor a way to accept one, so it answers on the other
    arm of the same union, ``active``, and the invitation fields stay null until
    that flow rehomes.
    """

    status: Literal["active", "invited"]
    email: str
    role: str
    organization_member_id: uuid.UUID | None = None
    user_id: uuid.UUID | None = None
    invitation_id: uuid.UUID | None = None
    full_name: str | None = None
    expires_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ActiveOrganizationMemberUpdateRequest(SQLModel):
    role: str | None = Field(default=None, max_length=32)
    status: str | None = Field(default=None, max_length=32)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_membership(value, allowed=ORGANIZATION_MEMBER_ROLES, kind="organization role")

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_membership(
            value,
            allowed=ORGANIZATION_MEMBER_UPDATABLE_STATUSES,
            kind="organization member status",
        )


class OrganizationMember(OrganizationMemberBase, PrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, table=True):
    __tablename__ = "organization_member"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "user_id",
            name="uq_organization_member_organization_user",
        ),
    )


# =============================================================================
# Workspaces
# =============================================================================


class WorkspaceBase(SQLModel):
    name: str = Field(max_length=255)
    description: str | None = Field(default=None, max_length=1024)


class WorkspaceCreate(WorkspaceBase):
    pass


class WorkspaceUpdate(SQLModel):
    name: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=1024)


class WorkspacePublic(WorkspaceBase):
    id: uuid.UUID
    organization_id: uuid.UUID
    created_by_user_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime | None = None


class WorkspacesPublic(SQLModel):
    data: list[WorkspacePublic]
    count: int


class Workspace(WorkspaceBase, PrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, table=True):
    __tablename__ = "workspace"
    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_workspace_organization_name"),
        CheckConstraint(
            "activation_classification IN ('eligible', 'internal', 'automated', 'migrated', 'enterprise_assisted')",
            name="check_workspace_activation_classification",
        ),
    )

    organization_id: uuid.UUID = Field(foreign_key="organization.id", ondelete="CASCADE", index=True)
    created_by_user_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="user.id",
        ondelete="SET NULL",
        nullable=True,
        index=True,
    )
    # Edition-invariant schema: nothing in the OSS control plane reads this, and
    # the activation surface that classifies a workspace is hosted depth. It
    # lives here because the overlay adds no tables of its own, so the column
    # has to exist in the one schema both editions boot.
    activation_classification: str = Field(default="eligible", max_length=32)


class WorkspaceMemberBase(SQLModel):
    workspace_id: uuid.UUID = Field(foreign_key="workspace.id", ondelete="CASCADE", index=True)
    user_id: uuid.UUID = Field(foreign_key="user.id", ondelete="CASCADE", index=True)
    role: str = Field(default="member", max_length=32)
    status: str = Field(default="active", max_length=32)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        return _validate_membership(value, allowed=WORKSPACE_MEMBER_ROLES, kind="workspace role")

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        return _validate_membership(value, allowed=WORKSPACE_MEMBER_STATUSES, kind="workspace member status")


class WorkspaceMemberCreate(WorkspaceMemberBase):
    pass


class WorkspaceMemberUpdate(SQLModel):
    role: str | None = Field(default=None, max_length=32)
    status: str | None = Field(default=None, max_length=32)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_membership(value, allowed=WORKSPACE_MEMBER_ROLES, kind="workspace role")

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_membership(value, allowed=WORKSPACE_MEMBER_STATUSES, kind="workspace member status")


class WorkspaceMemberPublic(WorkspaceMemberBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None = None


class WorkspaceMembersPublic(SQLModel):
    data: list[WorkspaceMemberPublic]
    count: int


class WorkspaceMember(WorkspaceMemberBase, PrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, table=True):
    __tablename__ = "workspace_member"
    __table_args__ = (UniqueConstraint("workspace_id", "user_id", name="uq_workspace_member_workspace_user"),)


__all__ = [
    "MANAGEMENT_ROLES",
    "ORGANIZATION_MEMBER_ROLES",
    "ORGANIZATION_MEMBER_STATUSES",
    "ORGANIZATION_MEMBER_UPDATABLE_STATUSES",
    "WORKSPACE_MEMBER_ROLES",
    "WORKSPACE_MEMBER_STATUSES",
    "ActiveOrganizationMemberCreateRequest",
    "ActiveOrganizationMemberCreateResultPublic",
    "ActiveOrganizationMemberPublic",
    "ActiveOrganizationMemberUpdateRequest",
    "ActiveOrganizationMembersPublic",
    "ActiveOrganizationUpdateRequest",
    "Organization",
    "OrganizationCreate",
    "OrganizationCreateRequest",
    "OrganizationMember",
    "OrganizationMemberCreate",
    "OrganizationMemberPublic",
    "OrganizationMemberUpdate",
    "OrganizationMembersPublic",
    "OrganizationMembershipContextPublic",
    "OrganizationMembershipContextsPublic",
    "OrganizationPublic",
    "OrganizationSwitchRequest",
    "OrganizationUpdate",
    "OrganizationsPublic",
    "User",
    "UserCreate",
    "Workspace",
    "WorkspaceActivationClassification",
    "WorkspaceAssignmentRequest",
    "WorkspaceCreate",
    "WorkspaceMember",
    "WorkspaceMemberCreate",
    "WorkspaceMemberPublic",
    "WorkspaceMemberUpdate",
    "WorkspaceMembersPublic",
    "WorkspacePublic",
    "WorkspaceUpdate",
    "WorkspacesPublic",
]
