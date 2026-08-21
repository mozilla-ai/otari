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
  anything in this edition. The identity columns
  (``hashed_password``, ``oauth_provider``, ``email_verification_token``,
  ``email_verified_at``, ``terms_accepted_at``) were held back while
  otari-ai#1716 was open and now join them: that issue settled that the master
  key stays the API credential while sessions become the dashboard login, which
  gives otari-ai#1644 one target schema instead of two. Two of them are read in
  this edition: ``hashed_password`` backs the dashboard password sign-in and
  ``email_verified_at`` is stamped when an operator claims a deployment
  (`gateway.services.tenancy.user_service`). ``oauth_provider`` and
  ``email_verification_token`` are still carried for the hosted edition alone.
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
from sqlalchemy import JSON, CheckConstraint, Column, DateTime, ForeignKey, UniqueConstraint, Uuid, func
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.types import TypeDecorator
from sqlmodel import Field, SQLModel

ORGANIZATION_MEMBER_ROLES = {"owner", "admin", "member", "viewer"}
ORGANIZATION_MEMBER_STATUSES = {"active", "invited", "suspended"}
WORKSPACE_MEMBER_ROLES = {"owner", "admin", "member", "viewer"}
WORKSPACE_MEMBER_STATUSES = {"active", "invited", "suspended"}

WorkspaceActivationClassification = Literal["eligible", "internal", "automated", "migrated", "enterprise_assisted"]

# The request-facing spellings of the vocabularies above. A ``Literal`` is what
# puts the allowed values in the OpenAPI schema and therefore in the generated
# dashboard client; a ``field_validator`` on a plain ``str`` enforces the same
# rule server-side but publishes nothing, so a client cannot tell what it may
# send until it is refused. Table columns stay ``str``: they must also hold the
# statuses this edition does not let anyone set.
OrganizationMemberRole = Literal["owner", "admin", "member", "viewer"]
WorkspaceMemberRole = Literal["owner", "admin", "member", "viewer"]
# What a member's status may be *set* to, which is narrower than what one may
# hold. "invited" stays a valid stored status because the invitation flow
# produces it and will rehome, but nothing in this edition can create or accept
# an invitation, so offering it here would advertise a state with no producer
# and no exit. (The M4 re-parenting backfill does not produce it either: it maps
# a blocked gateway user to "suspended" and every other one to "active".)
# Widening this back when invitations rehome is additive; narrowing later would
# not be.
OrganizationMemberSettableStatus = Literal["active", "suspended"]

# One request may not carry an unbounded assignment list, matching the ceiling
# the read endpoints put on repeatable filters (``MAX_FILTER_VALUES``).
MAX_WORKSPACE_ASSIGNMENTS = 50

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
    per-request spend identity, which is what keys, budgets, and usage attach to.
    Both exist, and how they converge is no longer settled: otari-ai#1719 made
    otari's schema the survivor, which retired the pre-flip plan of re-parenting
    the request plane onto this table through the identity bridge. otari-ai#1727
    holds the open decision and names two candidates, re-pointing the
    request-plane foreign keys here or keeping both with one authoritative. Until
    it lands, ``ActiveOrganizationMemberPublic.attribution_user_id`` is the join
    between the two.

    ``email`` is nullable and unique. PostgreSQL and SQLite both allow repeated
    NULLs in a unique index, so email-less local identities coexist without
    weakening uniqueness for the addresses that do exist.
    """

    __tablename__ = "user"

    email: str | None = Field(default=None, unique=True, index=True, max_length=255)
    # The credential columns, in the platform's own order. All nullable. Two of
    # them are read here: ``hashed_password`` is the dashboard password sign-in
    # and ``email_verified_at`` is stamped when an operator claims a deployment
    # (`gateway.services.tenancy.user_service`). The other three land ahead of
    # the flows that read them (otari-ai#1716) so the re-parenting migration
    # (otari-ai#1644) has one target schema rather than one per edition. A row
    # with every one of them null is the normal standalone state, not an
    # unmigrated one: it is a deployment nobody has claimed yet, and the master
    # key remains the API credential either way.
    #
    # Unbounded, matching the platform's ``AutoString()``: a hash carries its own
    # algorithm and cost parameters, so a length ceiling here would be a bet on
    # which hash the session flow picks.
    hashed_password: str | None = Field(default=None)
    terms_accepted_at: datetime | None = _timestamp_field(default=None, column_kwargs={})
    # ``str`` rather than the platform's native ``oauthprovider`` enum. The
    # vocabulary belongs to the OAuth flow that has not rehomed yet, and a
    # PostgreSQL enum would have to be created and dropped by hand around
    # ``add_column`` while rendering as VARCHAR plus a CHECK on SQLite, which the
    # OSS edition ships by default. This matches how the tenancy tables already
    # store their own vocabularies (``role``, ``status``).
    oauth_provider: str | None = Field(default=None, max_length=50)
    # Unique, like the platform's: two identities holding one verification token
    # would let either confirm the other's address.
    email_verification_token: str | None = Field(default=None, unique=True, index=True)
    email_verified_at: datetime | None = _timestamp_field(default=None, column_kwargs={})
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


class CallerWorkspaceMembershipPublic(SQLModel):
    """One workspace the caller belongs to, and their role in it.

    Carried on the membership context so the shell can populate its workspace
    switcher and choose a default from the first authenticated call, rather than
    listing workspaces and then asking for the caller's role in each. Only the
    caller's own memberships appear, so this is not a directory of the
    organization's workspaces: an admin sees the ones they joined, and the
    workspace list endpoint remains the way to see the rest.
    """

    workspace_id: uuid.UUID
    name: str
    role: str


class OrganizationMembershipContextPublic(SQLModel):
    """An organization plus the caller's standing in it.

    What every tenancy page reads first: which organization it is looking at,
    and what the caller may do there.
    """

    organization_member_id: uuid.UUID
    role: str
    status: str
    organization: OrganizationPublic
    workspace_memberships: list[CallerWorkspaceMembershipPublic] = Field(default_factory=list)
    # Whether the dashboard may offer the BYO provider-keys surface. The
    # platform answers "does this org have a self-hosted gateway attached", which
    # in a standalone deployment is always yes: the deployment reading this *is*
    # that gateway. Kept on the contract, rather than dropped as a constant, so
    # the ported page reads the same field in both editions.
    byo_provider_keys_allowed: bool = False


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

    ``attribution_user_id`` is the addition the platform has no counterpart for.
    Keys, budgets, and usage attach to the gateway's string-keyed ``users`` row,
    not to this UUID identity, so this carries the ``user_id`` a caller passes to
    ``POST /v1/keys`` to give this member a key. It is null when no usable row
    exists (nobody minted one, or it was soft-deleted through
    ``DELETE /v1/users``), which is the signal not to offer this member as a key
    owner: key creation would refuse. How the two ids converge is the open
    question in otari-ai#1727; this field is the join until it is answered, and
    is what lets either answer land without the dashboard changing.
    """

    organization_member_id: uuid.UUID | None = None
    user_id: uuid.UUID | None = None
    attribution_user_id: str | None = None
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
    role: WorkspaceMemberRole = "member"


class ActiveOrganizationMemberCreateRequest(SQLModel):
    """Add someone to the caller's organization, optionally into workspaces at once."""

    # Not ``EmailStr``: that would pull in email-validator for one field, and the
    # address is a claim handle rather than something this edition delivers to.
    # The format hint still reaches the generated client, so a form validates it.
    # SQLModel splats ``schema_extra`` into pydantic's ``FieldInfo``, which drops
    # a key it does not recognize, so the hint has to arrive under
    # ``json_schema_extra`` to reach the schema.
    email: str = Field(max_length=255, schema_extra={"json_schema_extra": {"format": "email"}})
    role: OrganizationMemberRole = "member"
    workspace_assignments: list[WorkspaceAssignmentRequest] | None = Field(
        default=None,
        max_length=MAX_WORKSPACE_ASSIGNMENTS,
    )


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
    attribution_user_id: str | None = None
    invitation_id: uuid.UUID | None = None
    full_name: str | None = None
    expires_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ActiveOrganizationMemberUpdateRequest(SQLModel):
    role: OrganizationMemberRole | None = None
    status: OrganizationMemberSettableStatus | None = None


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


# =============================================================================
# Invitations
# =============================================================================

INVITATION_STATUSES = {"pending", "accepted", "cancelled", "expired"}
InvitationStatus = Literal["pending", "accepted", "cancelled", "expired"]


class InvitationBase(SQLModel):
    email: str = Field(max_length=255)
    status: str = Field(default="pending", max_length=32)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        return _validate_membership(value, allowed=INVITATION_STATUSES, kind="invitation status")


class InvitationCreate(InvitationBase):
    pass


class InvitationUpdate(SQLModel):
    status: str | None = Field(default=None, max_length=32)


class InvitationPreviewPublic(SQLModel):
    """What an unauthenticated visitor sees before committing to accept.

    Deliberately narrow: the address it was sent to, the organization's name,
    and the role on offer. The token is the caller's only credential here, not
    a session, so this carries nothing that identifies who sent it or any
    other member.
    """

    email: str
    organization_name: str
    role: str
    expires_at: datetime


class InviteOrganizationMemberRequest(SQLModel):
    """Invite an address to the caller's organization, optionally into workspaces at once.

    Field-for-field ``ActiveOrganizationMemberCreateRequest``'s twin: the two
    requests ask for the same thing and differ only in what creating one
    produces (this lands ``invited`` and emails a link; that lands ``active``
    immediately).
    """

    email: str = Field(max_length=255, schema_extra={"json_schema_extra": {"format": "email"}})
    role: OrganizationMemberRole = "member"
    workspace_assignments: list[WorkspaceAssignmentRequest] | None = Field(
        default=None,
        max_length=MAX_WORKSPACE_ASSIGNMENTS,
    )


class InviteOrganizationMemberResultPublic(SQLModel):
    """What issuing an invitation produces, and whether the email actually went out."""

    invitation_id: uuid.UUID
    organization_member_id: uuid.UUID
    email: str
    role: str
    status: Literal["invited"] = "invited"
    mail_sent: bool = Field(
        description=(
            "Whether the invitation email was actually dispatched. False when mail is not "
            "configured, or the send itself failed; accept_link is set either way, so the "
            "operator can share it themselves rather than the invitation being a dead end."
        )
    )
    accept_link: str
    expires_at: datetime
    created_at: datetime


class ValidateInvitationRequest(SQLModel):
    """The preview lookup's body.

    A ``POST`` with the token in the body rather than a ``GET`` with it in the
    URL, matching ``AcceptInvitationRequest``: the token is a bearer-style
    credential (see ``Invitation.token_hash``'s docstring), and a URL is one a
    proxy or an access log routinely retains, which a request body is not.
    """

    token: str


class AcceptInvitationRequest(SQLModel):
    token: str


class AcceptInvitationResultPublic(SQLModel):
    """What accepting produces: enough for the accept page to say where the visitor landed.

    No session and no token: accepting resolves the membership to ``active``
    and stops there. Otari has no per-user sign-in yet, so there is nothing to
    sign this visitor into; they see the sign-in screen next, the same as
    anyone else added to an organization before that flow exists.
    """

    organization_name: str
    role: str


class Invitation(InvitationBase, PrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, table=True):
    """One organization-member invitation: an emailed accept link.

    Points at the ``OrganizationMember`` row created at invite time
    (``status="invited"``), rather than the membership being created on
    acceptance: the roster already lists an ``invited`` row for free
    (``LISTABLE_STATUSES``), and accepting flips that same row to ``active``
    rather than creating a fresh one. ``role`` lives on that row, not
    duplicated here.

    A membership can be invited, revoked (which cancels this row and suspends
    the membership, not delete either), and re-invited, and each round mints a
    fresh ``Invitation`` row against the same membership id rather than
    reusing or deleting the cancelled one, so its history stays queryable.
    At most one row is ``pending`` for a given membership at a time; that is a
    service-layer invariant (a membership already ``invited`` refuses a second
    invite), not a database constraint, the same way an organization's
    active-vs-suspended membership history has none either.

    ``workspace_assignments`` are parked here rather than applied immediately,
    since there is no active membership yet to grant them to;
    ``accept_invitation`` (``organization_service.py``) applies them once the
    member is active, the same way immediate assignments are applied on
    ``POST /me/members``.

    The token itself is never stored, only its hash (``token_hash``), matching
    ``dashboard_session_service``'s reasoning: a bearer-style secret sitting in
    a queryable column is the same risk class as a password, so it is hashed at
    rest the same way and compared by hash, never by value.
    """

    __tablename__ = "invitation"

    organization_id: uuid.UUID = Field(foreign_key="organization.id", ondelete="CASCADE", index=True)
    # Not unique: a membership can be invited, revoked (which suspends it, not
    # deletes it, and cancels this row), and re-invited, and each round mints a
    # fresh row against the same membership rather than reusing or deleting the
    # cancelled one, so its history stays queryable. At most one row is
    # ``pending`` for a given membership at a time, which is what the service
    # layer enforces (a membership already ``invited`` refuses a second invite)
    # rather than a constraint here, the same way OrganizationMember's own
    # active-vs-suspended history has no DB-level exclusivity either.
    organization_member_id: uuid.UUID = Field(
        foreign_key="organization_member.id",
        ondelete="CASCADE",
        index=True,
    )
    invited_by_user_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="user.id",
        ondelete="SET NULL",
        index=True,
    )
    token_hash: str = Field(unique=True, index=True, max_length=64)
    workspace_assignments: list[dict[str, str]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    expires_at: datetime = Field(sa_type=UtcDateTime())  # type: ignore[call-overload]


__all__ = [
    "INVITATION_STATUSES",
    "MANAGEMENT_ROLES",
    "MAX_WORKSPACE_ASSIGNMENTS",
    "ORGANIZATION_MEMBER_ROLES",
    "ORGANIZATION_MEMBER_STATUSES",
    "WORKSPACE_MEMBER_ROLES",
    "WORKSPACE_MEMBER_STATUSES",
    "AcceptInvitationRequest",
    "AcceptInvitationResultPublic",
    "ActiveOrganizationMemberCreateRequest",
    "ActiveOrganizationMemberCreateResultPublic",
    "ActiveOrganizationMemberPublic",
    "ActiveOrganizationMemberUpdateRequest",
    "ActiveOrganizationMembersPublic",
    "ActiveOrganizationUpdateRequest",
    "Invitation",
    "InvitationCreate",
    "InvitationPreviewPublic",
    "InvitationStatus",
    "InvitationUpdate",
    "InviteOrganizationMemberRequest",
    "InviteOrganizationMemberResultPublic",
    "Organization",
    "OrganizationCreate",
    "OrganizationMember",
    "OrganizationMemberCreate",
    "OrganizationMemberPublic",
    "OrganizationMemberRole",
    "OrganizationMemberSettableStatus",
    "OrganizationMemberUpdate",
    "OrganizationMembersPublic",
    "OrganizationMembershipContextPublic",
    "OrganizationPublic",
    "OrganizationUpdate",
    "OrganizationsPublic",
    "User",
    "UserCreate",
    "ValidateInvitationRequest",
    "Workspace",
    "WorkspaceActivationClassification",
    "WorkspaceAssignmentRequest",
    "WorkspaceCreate",
    "WorkspaceMember",
    "WorkspaceMemberCreate",
    "WorkspaceMemberPublic",
    "WorkspaceMemberUpdate",
    "WorkspaceMemberRole",
    "WorkspaceMembersPublic",
    "WorkspacePublic",
    "WorkspaceUpdate",
    "WorkspacesPublic",
]
