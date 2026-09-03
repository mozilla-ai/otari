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
  operator identity is a label, not a sign-in address ("local identities have no
  email"), and every reader here must already tolerate its absence, so
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
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import field_validator
from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
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
# and no exit. (A convergence backfill would not produce it either: the mapping
# it describes sends a blocked gateway user to "suspended" and every other one
# to "active".)
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
    # otari#650's own columns, added alongside rather than reusing
    # ``email_verification_token`` above: that one is carried verbatim for
    # hosted-edition parity and stores a raw token per its own comment, and
    # repurposing it to hold a hash would be an undocumented divergence from
    # whatever the hosted platform's production column still expects of it.
    # These four follow the invitation token's own shape instead
    # (`Invitation.token_hash`): only a SHA-256 hash is ever stored
    # (`gateway.services.tenancy.tokens`), and single-use is enforced by
    # clearing the hash and expiry to NULL on success rather than by a status
    # column, so a replayed token simply matches no row.
    email_verification_token_hash: str | None = Field(default=None, unique=True, index=True, max_length=64)
    email_verification_token_expires_at: datetime | None = _timestamp_field(default=None, column_kwargs={})
    password_reset_token_hash: str | None = Field(default=None, unique=True, index=True, max_length=64)
    password_reset_token_expires_at: datetime | None = _timestamp_field(default=None, column_kwargs={})
    # NOT NULL: every identity is always looking at exactly one organization,
    # which is what lets the tenancy routes resolve a scope from the caller
    # alone. Provisioning therefore creates the organization first.
    active_organization_id: uuid.UUID = Field(foreign_key="organization.id", index=True)
    # The organization provisioned for this identity, which never moves when the
    # active one does. Nothing in this edition reads it, and it is written once,
    # by ``create_local_identity``: switching (mozilla-ai/otari#715) moves
    # ``active_organization_id`` and deliberately leaves this where it was,
    # which is the whole point of there being two columns. The hosted edition
    # anchors recurring offered credits to it precisely so they cannot be farmed
    # by creating or switching organizations. A column the hosted edition needs
    # has to live here or nowhere, and the overlay contributes adapters and
    # routers, never tables.
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
    # When this identity last had a dashboard session minted for it. Stamped in
    # one place, ``dashboard_session_service.create_dashboard_session``, so every
    # sign-in flow that mints a session records it and none of them has to
    # remember to; see that function for why a re-mint counts.
    #
    # A column rather than ``max(dashboard_sessions.created_at)``, which would
    # have needed no migration and would have been wrong: session rows are pruned
    # once they expire, so the derived answer decays from "signed in three weeks
    # ago" to "never signed in" with nothing to distinguish the two. NULL here
    # means never, and keeps meaning it.
    last_sign_in_at: datetime | None = _timestamp_field(default=None, column_kwargs={})


class DeploymentUserOrganizationPublic(SQLModel):
    """One organization an identity belongs to, as the operator surface lists it."""

    organization_id: uuid.UUID
    name: str
    slug: str
    role: str
    status: str


class DeploymentUserPublic(SQLModel):
    """An identity on this deployment, whatever organization it belongs to.

    Not ``ActiveOrganizationMemberPublic``: that shape is a *membership* joined
    to an identity, scoped to one organization and hiding the suspended rows.
    This one is the identity itself, and its ``organizations`` list carries every
    membership at whatever status, because an account whose only membership is
    suspended is precisely what an operator comes here to find.

    ``is_bootstrap_operator`` and ``is_self`` are the two rows an operator may
    not deactivate or demote, and they travel on the row so the page can disable
    those controls rather than offering ones the server refuses. Neither is an
    authorization: the server refuses either way. ``is_self`` is answered here
    because nothing else the dashboard fetches names the caller's identity, so
    without it the page could not tell which row is the reader's own.
    """

    # No defaults on any of these, unlike the shapes above that share a base with
    # a table model: ``_to_public`` sets every field on every row, so a default
    # here would only be a default in the *schema*, which is what decides whether
    # the field lands in the spec's ``required`` list and so whether the generated
    # client types it as optional. Nullable and optional are not the same claim,
    # and the four that can be null say so with ``| None`` while staying required.
    id: uuid.UUID
    email: str | None
    full_name: str | None
    is_active: bool
    is_superuser: bool
    is_bootstrap_operator: bool
    is_self: bool
    last_sign_in_at: datetime | None
    created_at: datetime
    organizations: list[DeploymentUserOrganizationPublic]


class DeploymentUsersPublic(SQLModel):
    data: list[DeploymentUserPublic]
    count: int


class DeploymentUserUpdateRequest(SQLModel):
    """The two flags the operator surface may flip, each optional.

    Omitting a field leaves it alone, so deactivating an account and changing
    what it may administer stay separate decisions even though one endpoint
    carries both. A body that sets neither is refused rather than treated as a
    no-op: it is a request that meant something and lost it.
    """

    is_active: bool | None = None
    is_superuser: bool | None = None


class DeploymentAdminAccessPublic(SQLModel):
    """Whether the caller may reach the deployment administration surface.

    The one endpoint in that surface that answers 200 for everybody, and
    deliberately: the rest refuse with 404 so they do not confirm they exist,
    which leaves the dashboard nothing to gate its navigation on but a failed
    request. This says the same thing a caller could learn by trying, without
    the try.
    """

    granted: bool


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


class CallerIdentityPublic(SQLModel):
    """Who the caller is, as against what they may do.

    The account control at the foot of the dashboard's sidebar draws a person,
    and no authenticated route reported the caller's own name or address, so it
    drew a role for everybody who could reach it: "Operator", which is not a
    name and on a multi-tenant deployment was not even true
    (mozilla-ai/otari#832).

    Carried on the membership context for the reason ``deployment_operator`` is:
    the shell reads that context before it paints, so an identity taken from it
    needs no request of its own and cannot arrive a beat after the chrome it
    names. Publishing it costs nothing either, since it is the caller's own
    identity and they are holding the credential that resolved to it.

    Both fields are nullable, and for opposite reasons. A local operator
    identity has no address, because first boot provisions it with a name and
    nothing to sign in with but the master key; a member added to the roster by
    address has no name until they claim the identity and supply one. So a shell
    has to be ready to draw either one alone.
    """

    user_id: uuid.UUID
    email: str | None = None
    full_name: str | None = None


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
    # Who is signed in, which every other field here describes the authority of
    # rather than the person holding it. Optional like the fields below it, and
    # for the same reason: a shell reading a deployment that predates this (or a
    # platform serving this contract without it) falls back to naming nobody,
    # rather than failing to parse the context the rest of the chrome needs.
    caller: CallerIdentityPublic | None = None
    # Whether the dashboard may offer the BYO provider-keys surface. The
    # platform answers "does this org have a self-hosted gateway attached", which
    # in a standalone deployment is always yes: the deployment reading this *is*
    # that gateway. Kept on the contract, rather than dropped as a constant, so
    # the ported page reads the same field in both editions.
    byo_provider_keys_allowed: bool = False
    # Whether this caller also operates the deployment, which is the *other*
    # authority a signed-in identity can hold and the one no organization role
    # confers (mozilla-ai/otari#838). Carried here, on a read the shell already
    # makes before it paints, so the sidebar has no in-flight window in which to
    # show a row it is about to retract (mozilla-ai/otari#836).
    #
    # The same predicate ``GET /v1/admin/access`` publishes, resolved through
    # ``DeploymentUserService.has_administration_access`` rather than re-derived,
    # so the two answers cannot come to disagree about who operates the
    # deployment. This does not retire that endpoint; it is a second publisher of
    # one fact. Two publishers is a transitional state rather than a design: the
    # server cannot give different answers, but they are separate requests
    # resolving on their own schedules, so two surfaces reading different ones can
    # settle out of step. Which is canonical for which layer of the dashboard,
    # and what would retire the endpoint's use there, is in ``web/AGENTS.md``.
    deployment_operator: bool = False
    # Whether the deployment can encrypt a stored credential at all, i.e. whether
    # ``OTARI_SECRET_KEY`` is set. A deployment fact rather than a tenant one, and
    # on the tenant's contract deliberately: it decides whether *this caller's*
    # provider-key write can succeed, and the only endpoint that reported it
    # (``GET /v1/settings``) is operator-gated, so a tenant page inferring it from
    # that read cannot tell a missing key from a refused request
    # (mozilla-ai/otari#839). Says whether a key is configured, never anything
    # about its value.
    provider_key_encryption_available: bool = False


class ActiveOrganizationUpdateRequest(SQLModel):
    name: str = Field(min_length=1, max_length=255)


class OrganizationCreateRequest(SQLModel):
    """Create an organization, with the caller as its owner.

    Name only. The slug is derived server-side and never sent, because it is
    unique where the name is not: two organizations may share a name, and a
    rename deliberately does not move the slug.
    """

    name: str = Field(min_length=1, max_length=255)


class SwitchActiveOrganizationRequest(SQLModel):
    """Point the caller's identity at one of the organizations they belong to.

    The one request in this surface that names an organization by id, and it is
    not a hole in the tenant boundary: an id the caller holds no active
    membership in answers 404, so it says nothing about whether the
    organization exists.
    """

    organization_id: uuid.UUID


class CallerOrganizationMembershipPublic(SQLModel):
    """One organization the caller belongs to, and their standing in it.

    What an organization switcher renders. Deliberately not
    ``OrganizationMembershipContextPublic``: that one answers "the organization
    this request is acting in" and carries the caller's workspaces in it, which
    for an organization they are not currently in would be a second read per
    row.
    """

    organization_member_id: uuid.UUID
    organization: OrganizationPublic
    role: str
    status: str
    # Which row the switcher marks as current. Derived from the caller's
    # ``active_organization_id`` rather than left to the client to work out,
    # because the client would have to read the context to know it and the two
    # answers could then disagree mid-switch.
    is_active_organization: bool = False


class CallerOrganizationMembershipsPublic(SQLModel):
    data: list[CallerOrganizationMembershipPublic]
    count: int


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
# Email-domain auto-join
# =============================================================================


# Roles a domain claim may hand out. Owner and admin are deliberately absent:
# the claim proves control of a *domain*, which is not the same as a decision
# about any one person, so a match must never confer management of the
# organization. Narrower than ORGANIZATION_MEMBER_ROLES for that reason alone,
# and widening it would make publishing one DNS record enough to mint admins.
ORGANIZATION_DOMAIN_ROLES = {"member", "viewer"}
OrganizationDomainRole = Literal["member", "viewer"]

# The TXT record an admin publishes at the claimed domain's apex to prove they
# control it. The stored token is the secret half; this prefix is what makes the
# record recognizable among the other TXT records a domain already publishes.
DOMAIN_VERIFICATION_TXT_PREFIX = "otari-domain-verification="

# How long one DNS proof is good for. A proof is evidence about the moment it
# was taken, and domains change hands: pull the record, transfer the domain, or
# let it lapse, and a stamp kept forever would go on admitting whoever owns the
# domain next, at a role this organization chose. Past this the claim stops
# admitting anyone until an admin re-verifies, which fails closed and needs no
# background sweeper. Re-checking on the sign-in path instead was the other
# option and is worse: it puts a 5s outbound lookup in front of a person
# waiting to sign in.
DOMAIN_PROOF_TTL = timedelta(days=90)

# The most domains one organization may claim. In the shape of
# MAX_WORKSPACE_ASSIGNMENTS, and here for a sharper reason: every unverified
# claim is a name this deployment will run an outbound DNS query against on
# demand, so an uncapped list is an uncapped query relay.
MAX_ORGANIZATION_DOMAINS = 50


class OrganizationDomainBase(SQLModel):
    organization_id: uuid.UUID = Field(foreign_key="organization.id", ondelete="CASCADE", index=True)
    # Indexed, and NOT unique. Uniqueness applies to *proven* claims only (see
    # the partial index on the table below): two organizations may both have an
    # unproven claim on a domain, and whichever proves it first is the one that
    # gets it.
    domain: str = Field(max_length=255, index=True)
    default_role: str = Field(default="member", max_length=32)
    enabled: bool = Field(default=True)

    # Runs on the request and response schemas below, and NOT on the table
    # class: SQLModel skips validation for ``table=True``, so the repository
    # constructing an ``OrganizationDomain`` directly is unchecked. The request
    # ``Literal`` is what actually keeps a management role out; this is a last
    # guard on the way back out, so a row that somehow held one could not be
    # serialized as if it were fine.
    @field_validator("default_role")
    @classmethod
    def validate_default_role(cls, value: str) -> str:
        return _validate_membership(value, allowed=ORGANIZATION_DOMAIN_ROLES, kind="auto-join role")


class OrganizationDomainCreate(OrganizationDomainBase):
    pass


class OrganizationDomainUpdate(SQLModel):
    default_role: str | None = Field(default=None, max_length=32)
    enabled: bool | None = Field(default=None)

    @field_validator("default_role")
    @classmethod
    def validate_default_role(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_membership(value, allowed=ORGANIZATION_DOMAIN_ROLES, kind="auto-join role")


class OrganizationDomainPublic(OrganizationDomainBase):
    """A claim as its organization's admins see it.

    Carries ``verification_record`` (the whole string to publish) rather than
    the raw token: the admin never has a use for the token on its own, and one
    field that can be copied verbatim into a DNS panel is harder to get wrong
    than a prefix they must remember to prepend.
    """

    id: uuid.UUID
    verification_record: str
    verified_at: datetime | None = None
    # When the proof stops being acted on, computed rather than left to the
    # caller: the TTL is a server rule, and a dashboard deriving it would hold a
    # second copy of the constant that could drift from this one.
    proof_expires_at: datetime | None = None
    created_at: datetime
    updated_at: datetime | None = None


class OrganizationDomainsPublic(SQLModel):
    data: list[OrganizationDomainPublic]
    count: int


class OrganizationDomainCreateRequest(SQLModel):
    domain: str = Field(min_length=1, max_length=255)
    default_role: OrganizationDomainRole = "member"
    enabled: bool = True


class OrganizationDomainUpdateRequest(SQLModel):
    default_role: OrganizationDomainRole | None = None
    enabled: bool | None = None


class OrganizationDomain(OrganizationDomainBase, PrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, table=True):
    """One organization's claim on an email domain, and its DNS proof.

    A claim on its own grants nothing. Auto-join reads only rows that are
    ``enabled``, verified, and whose proof is younger than ``DOMAIN_PROOF_TTL``,
    so an organization may name any domain it likes and nothing happens until
    the record is published.

    **Only a proven claim is exclusive.** The unique index is partial, over
    verified rows alone, so any number of organizations may hold an unproven
    claim on one domain and the first to publish the record takes it. A plain
    ``UNIQUE(domain)`` would have made claiming first-come-first-served, which
    hands anyone who can create an organization a way to permanently lock the
    real owner of a domain out of ever claiming it.

    The row outlives verification rather than collapsing into a boolean,
    because the proof is re-checked: ``verified_at`` is the age of the evidence
    and the token stays the value the published record has to keep matching.
    """

    __tablename__ = "organization_domain"
    __table_args__ = (
        Index(
            "uq_organization_domain_verified_domain",
            "domain",
            unique=True,
            postgresql_where=text("verified_at IS NOT NULL"),
            sqlite_where=text("verified_at IS NOT NULL"),
        ),
    )

    verification_token: str = Field(max_length=64)
    verified_at: datetime | None = _timestamp_field(default=None, column_kwargs={})

    @property
    def verification_record(self) -> str:
        """The exact TXT value to publish at the domain's apex."""
        return f"{DOMAIN_VERIFICATION_TXT_PREFIX}{self.verification_token}"

    def proof_expired(self, *, now: datetime) -> bool:
        """Whether the DNS proof is too old to still be acted on."""
        return self.verified_at is None or now - self.verified_at > DOMAIN_PROOF_TTL


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
    and stops there. The identity it resolves to is password-less on the roster
    until it is claimed, so the next step is ``POST /v1/auth/signup`` on the
    invited address, not a sign-in.
    """

    organization_name: str
    role: str


class PendingOrganizationInvitationPublic(SQLModel):
    """One invitation waiting on the caller, as their own inbox lists it.

    The authenticated counterpart to ``InvitationPreviewPublic``, and wider
    than it on purpose: that one answers a visitor whose only credential is
    the token, so it carries nothing it does not strictly need, while this one
    answers the addressee's own session and can name the ids its accept and
    decline calls take.

    ``organization_member_id`` is what those two calls address, not
    ``invitation_id``: the membership is the row that outlives a revoke and a
    re-invite (each round mints a fresh ``Invitation`` against the same
    membership), so a client holding a list from a moment ago names something
    still resolvable rather than a token-shaped id that has since been
    superseded. ``invitation_id`` rides along for the roster's sake, since
    ``ActiveOrganizationMemberPublic`` carries the same field.
    """

    organization_member_id: uuid.UUID
    invitation_id: uuid.UUID
    organization_id: uuid.UUID
    organization_name: str
    # The address the invitation was actually sent to, read off the invitation
    # rather than off the caller's identity. The two are the same address at
    # invite time (an invitation resolves its membership through
    # ``get_by_email``), but this one is a record of where the link was mailed
    # and does not follow a later change to the identity's own address.
    email: str
    role: str
    expires_at: datetime
    created_at: datetime


class PendingOrganizationInvitationsPublic(SQLModel):
    data: list[PendingOrganizationInvitationPublic]
    count: int


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


# =============================================================================
# WebAuthn (passkeys)
# =============================================================================

# The longest label a passkey may be given. A label, not a name anybody else
# sees: it exists so an operator holding three passkeys can tell which one is
# the laptop, so it is bounded generously and validated for emptiness rather
# than for shape.
MAX_WEBAUTHN_CREDENTIAL_NAME = 255
# How long an issued ceremony challenge stays consumable. The spec sets no
# floor; browsers surface a `timeout` hint of 60s, and an authenticator that
# needs a user to find their phone routinely runs past it, so this is longer
# than the hint on purpose. It is still short: the row is a single-use nonce,
# and every one of them that outlives its ceremony is a row a sweep has to
# reach.
WEBAUTHN_CHALLENGE_TTL_SECONDS = 300
# base64url of the 1023 bytes the spec caps a credential ID at, which is the
# widest value an authenticator may hand back. Sized to the spec rather than to
# the ~20 bytes real authenticators emit: a row that cannot be written is a
# passkey that cannot be registered, and the column is text either way.
MAX_CREDENTIAL_ID_LENGTH = 1364

WebAuthnCeremony = Literal["registration", "authentication"]
WEBAUTHN_CEREMONIES: set[str] = {"registration", "authentication"}


class WebAuthnCredentialBase(SQLModel):
    """The fields a registered passkey carries on the wire."""

    name: str = Field(max_length=MAX_WEBAUTHN_CREDENTIAL_NAME)


class WebAuthnCredentialUpdate(SQLModel):
    """Renaming a passkey, which is the only thing about one that is editable.

    Everything else on the row is what the authenticator asserted, so there is
    nothing else a person could correct.
    """

    name: str = Field(max_length=MAX_WEBAUTHN_CREDENTIAL_NAME)

    @field_validator("name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "A passkey name cannot be blank"
            raise ValueError(msg)
        return stripped


class WebAuthnCredentialPublic(WebAuthnCredentialBase):
    """A passkey as the settings page lists it.

    Carries no key material. ``credential_id`` is here because the browser needs
    it to tell the passkey it just used from the others in the list, and it is
    a public identifier the authenticator hands to any site that asks: it is
    what ``allowCredentials`` publishes to an unauthenticated caller during a
    ceremony.
    """

    id: uuid.UUID
    credential_id: str
    rp_id: str
    # Whether this row can still answer a ceremony on this deployment *right
    # now*, which is not something the client could work out for itself: it
    # would need the deployment's current relying-party ID, and publishing that
    # to say "no" would be a worse trade than answering the question here. False
    # means the passkey is orphaned, by the ID having moved or by the deployment
    # no longer being configured for passkeys at all, and the only thing left to
    # do with it is delete it.
    is_usable: bool
    transports: list[str]
    backed_up: bool
    created_at: datetime
    last_used_at: datetime | None


class WebAuthnCredentialsPublic(SQLModel):
    data: list[WebAuthnCredentialPublic]
    count: int


class WebAuthnCredential(WebAuthnCredentialBase, PrimaryKeyMixin, CreatedAtMixin, table=True):
    """A passkey bound to one identity, one relying party, and one authenticator.

    **The relying-party ID is stored, not assumed.** A passkey is scoped by the
    authenticator to the ``rp_id`` it was created under, so a credential
    registered under one ID is unusable under another and asking its
    authenticator for it under a different one gets nothing back. Recording the
    ID the row was made under is what lets this deployment say so: a credential
    whose ``rp_id`` is not the one currently configured is filtered out of the
    ceremonies rather than offered and then failing in the browser with a
    ``SecurityError`` nothing on the server can explain.

    It is also the column that carries mozilla-ai/otari-ai#1716's standing
    constraint. Migrating otari.ai users import their credentials rather than
    claiming new accounts, and an imported row's ``rp_id`` is ``otari.ai``. That
    import therefore holds exactly while the hosted origin stays ``otari.ai``:
    moving it re-scopes every passkey and no amount of data migration recovers
    them, because the key material never left the authenticator. See
    `docs/access-control.md`.

    ``credential_id`` and ``public_key`` are base64url text rather than
    ``LargeBinary``. Both cross the wire in that encoding in every WebAuthn
    payload, the import above arrives in it, and it reads the same on SQLite and
    PostgreSQL, where a bytes column does not (``BLOB`` versus ``BYTEA``, with
    drivers differing on what comes back). Unique on ``credential_id`` alone,
    not per user: a credential ID that resolved to two identities would make
    a usernameless sign-in ambiguous, which is the one thing that flow cannot
    tolerate.

    ``sign_count`` is the authenticator's own monotonic counter, updated on each
    assertion. Not every authenticator keeps one (a platform passkey synced
    across devices reports 0 forever), so it is recorded and compared but a
    non-increase is not by itself proof of a clone; see
    `services.webauthn_service` for what is actually done with it.
    """

    __tablename__ = "webauthn_credential"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_webauthn_credential_user_name"),)

    user_id: uuid.UUID = Field(
        sa_column=Column(Uuid, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    )
    credential_id: str = Field(unique=True, index=True, max_length=MAX_CREDENTIAL_ID_LENGTH)
    # Unbounded, for the reason ``user.hashed_password`` is: a COSE key carries
    # its own algorithm, and an RSA credential's key is an order of magnitude
    # longer than the EC one a platform passkey emits, so a ceiling here would
    # be a bet on which algorithm an authenticator picks.
    public_key: str
    rp_id: str = Field(index=True, max_length=255)
    sign_count: int = Field(default=0)
    transports: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    # What the authenticator said about the credential at registration, kept
    # because it is what a person recognizes their passkey by: a backed-up
    # credential is one their phone or password manager syncs, and a
    # single-device one dies with the device. Nothing enforces on it.
    backed_up: bool = Field(default=False)
    aaguid: str | None = Field(default=None, max_length=64)
    last_used_at: datetime | None = _timestamp_field(default=None, column_kwargs={})


class WebAuthnChallenge(SQLModel, table=True):
    """A single-use nonce issued for one ceremony and consumed by its answer.

    In the database rather than in process memory for the same reason
    ``dashboard_sessions`` is: a deployment runs more than one worker, and a
    challenge issued by one of them is answered against whichever one the next
    request lands on. An in-memory store works exactly until a deployment scales
    past one process, and then fails as an intermittent, unreproducible sign-in
    refusal.

    The challenge is its own primary key. It is 32 random bytes generated by the
    server, it is handed to the browser in the clear (that is what a challenge
    *is*), and nothing is stored under it, so there is nothing here that hashing
    would protect. What matters is that it is used once: the row is deleted as
    it is consumed, so a replayed assertion matches nothing.

    ``user_id`` is null for an authentication challenge, and that is the
    usernameless sign-in this deployment offers: the browser picks the passkey
    and the assertion names which credential answered, so the ceremony starts
    without knowing who is signing in. A registration challenge always names the
    identity that asked for it, because registration is done from inside a
    session.
    """

    __tablename__ = "webauthn_challenge"

    challenge: str = Field(primary_key=True, max_length=255)
    ceremony: str = Field(max_length=32)
    user_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(Uuid, ForeignKey("user.id", ondelete="CASCADE"), nullable=True, index=True),
    )
    created_at: datetime = _timestamp_field(
        default_factory=lambda: datetime.now(UTC),
        column_kwargs={"server_default": func.now()},
    )
    expires_at: datetime = Field(sa_type=UtcDateTime(), index=True)  # type: ignore[call-overload]

    @field_validator("ceremony")
    @classmethod
    def _known_ceremony(cls, value: str) -> str:
        return _validate_membership(value, allowed=WEBAUTHN_CEREMONIES, kind="WebAuthn ceremony")


__all__ = [
    "DeploymentAdminAccessPublic",
    "DeploymentUserOrganizationPublic",
    "DeploymentUserPublic",
    "DeploymentUserUpdateRequest",
    "DeploymentUsersPublic",
    "DOMAIN_PROOF_TTL",
    "DOMAIN_VERIFICATION_TXT_PREFIX",
    "INVITATION_STATUSES",
    "MAX_CREDENTIAL_ID_LENGTH",
    "MAX_WEBAUTHN_CREDENTIAL_NAME",
    "MANAGEMENT_ROLES",
    "MAX_ORGANIZATION_DOMAINS",
    "MAX_WORKSPACE_ASSIGNMENTS",
    "ORGANIZATION_DOMAIN_ROLES",
    "ORGANIZATION_MEMBER_ROLES",
    "ORGANIZATION_MEMBER_STATUSES",
    "WORKSPACE_MEMBER_ROLES",
    "WEBAUTHN_CEREMONIES",
    "WEBAUTHN_CHALLENGE_TTL_SECONDS",
    "WORKSPACE_MEMBER_STATUSES",
    "AcceptInvitationRequest",
    "AcceptInvitationResultPublic",
    "ActiveOrganizationMemberCreateRequest",
    "ActiveOrganizationMemberCreateResultPublic",
    "ActiveOrganizationMemberPublic",
    "ActiveOrganizationMemberUpdateRequest",
    "ActiveOrganizationMembersPublic",
    "ActiveOrganizationUpdateRequest",
    "CallerOrganizationMembershipPublic",
    "CallerOrganizationMembershipsPublic",
    "CallerWorkspaceMembershipPublic",
    "Invitation",
    "InvitationCreate",
    "InvitationPreviewPublic",
    "InvitationStatus",
    "InvitationUpdate",
    "InviteOrganizationMemberRequest",
    "InviteOrganizationMemberResultPublic",
    "Organization",
    "OrganizationCreate",
    "OrganizationCreateRequest",
    "OrganizationDomain",
    "OrganizationDomainCreate",
    "OrganizationDomainCreateRequest",
    "OrganizationDomainPublic",
    "OrganizationDomainRole",
    "OrganizationDomainUpdate",
    "OrganizationDomainUpdateRequest",
    "OrganizationDomainsPublic",
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
    "PendingOrganizationInvitationPublic",
    "PendingOrganizationInvitationsPublic",
    "SwitchActiveOrganizationRequest",
    "User",
    "UserCreate",
    "ValidateInvitationRequest",
    "WebAuthnCeremony",
    "WebAuthnChallenge",
    "WebAuthnCredential",
    "WebAuthnCredentialPublic",
    "WebAuthnCredentialUpdate",
    "WebAuthnCredentialsPublic",
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
