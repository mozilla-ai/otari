"""Organization-scoped guardrail configuration: CRUD, and the request-path resolve.

The fourth and last of the config planes #655 asked about, and the only one whose
rows sit *above* the deployment rather than below it. ``guardrails_url`` stays a
deployment concern in ``runtime_settings`` (``routes/tool_settings.py``), request
handlers still run whatever a caller asks for, and a deployment that configures
no organization guardrails behaves exactly as it did (otari#654).

**What a row is.** One guardrail an organization mandates: a profile on the
guardrails service, an optional endpoint and credential of its own, the two
failure-handling modes a request-body entry already carries, and the switch that
decides whether it runs in every workspace of the organization or only in the
ones its scope names. At admission the rows in scope for the request's workspace
are merged into the effective guardrail list, exactly as a routing policy's
mandate already is, with the stricter setting winning on every axis.

**Why this stays inside the #655/#678 rule** even though it is a layer adding
configuration rather than narrowing one: a mandated guardrail can only make
fewer requests succeed, never more. That is also why an entry may carry a URL
and a credential where `workspace_code_execution_policy_service` may not. A
sandbox is a capability the tenant would be acquiring; a guardrail is a
restriction the tenant is accepting, and a caller can already point a
request-body guardrail at an endpoint of their own
(``models/guardrails.GuardrailConfig.url``), so nothing here is reachable that
was not reachable before.

**Composition with the layer below is org-controlled only.** There is no
workspace opt-in and no workspace opt-out: a veto would widen what succeeds
beyond what the organization permitted, which is the one thing #678 forbids. The
hosted plane's ``organization_guardrail_key_access`` row, with its three states
two of which mean off, therefore reduces here to plain membership.

**Deliberate divergences from otari-ai's ``OrganizationGuardrailKeyService``:**

- ``profile`` is unique per organization, where the hosted plane keys on a
  ``nickname`` so one profile may be configured twice. This request path merges
  guardrails by profile and always has, so a second row of one profile could
  never run; refusing it at the write beats losing it at admission.
- No ``key_source``. Platform-managed and platform-generated guardrail sources
  are commercial depth (otari-ai#1699); bring-your-own is the whole of the
  open-core shape, so the credential column is either set or it is not.
- An ``enabled`` flag on the row, which the hosted model has no equivalent of:
  an organization can stop a guardrail without losing the credential and the
  workspace list it took to set up.
- ``extra_kwargs_for_creation`` is dropped. This repository's guardrails service
  builds its guardrails at boot from the operator's own YAML, so creation kwargs
  are not ours to send; ``validate_kwargs`` is what ``POST /validate`` accepts.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import OrganizationGuardrail, OrganizationGuardrailWorkspace
from gateway.models.guardrails import GuardrailConfig
from gateway.models.tenancy import User
from gateway.repositories.tenancy import WorkspaceRepository
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.tenancy.errors import (
    OrganizationGuardrailAlreadyExistsError,
    OrganizationGuardrailLimitReachedError,
    OrganizationGuardrailNotFoundError,
    OrganizationGuardrailScopeConflictError,
    OrganizationGuardrailUnsafeUrlError,
    SecretBoxUnavailableTenancyError,
    WorkspaceNotFoundError,
)
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.url_safety import UnsafeURLError, validate_mcp_url

# What one organization may mandate. Every guardrail in scope for a workspace is
# one more call the request waits on before the provider is reached, and
# `run_input_guardrails` runs them one after another, so this bounds latency an
# organization spends on every request rather than only rows it stores.
MAX_GUARDRAILS_PER_ORGANIZATION = 25

# The scope list is written whole on every create and update, so without a bound
# it is the one way to put an unbounded value in this surface's request body.
# Well above any plausible organization; a larger one uses
# ``applies_to_all_workspaces``, which is what it is for.
MAX_SCOPED_WORKSPACES = 500

_MAX_LIST_LIMIT = 1000

# Sentinel for "this PATCH did not mention the field", which is not the same as
# "this PATCH cleared it": see `OrganizationGuardrailUpdate`.
_UNSET = object()


class OrganizationGuardrailCreate(BaseModel):
    """Request body for mandating a guardrail across an organization.

    ``credential`` is never stored as sent: it is encrypted with
    ``OTARI_SECRET_KEY`` and only the ciphertext is kept, the same convention
    `entities.WorkspaceMcpServer` and `entities.ProviderCredential` use. It is
    sent to the endpoint as ``Authorization: Bearer`` when the guardrail runs,
    so it authenticates this gateway to the guardrails service the entry names.
    A guardrail *vendor's* own key is not this: the guardrails service builds
    its guardrails from the operator's YAML and holds those itself.

    ``on`` is not offered. This plane mandates input-direction checks, which is
    the only direction the request path enforces
    (`services.guardrails.run_input_guardrails`); an organization that could
    store an output-direction mandate would be storing one nothing runs.
    """

    profile: str = Field(
        min_length=1,
        max_length=128,
        description="Profile name configured on the guardrails service, unique within the organization",
    )
    url: str | None = Field(
        default=None,
        max_length=2048,
        description="Guardrails endpoint for this entry; null uses the deployment's guardrails_url",
    )
    credential: str | None = Field(
        default=None,
        max_length=8192,
        description="Bearer credential for that endpoint; requires an https URL. Encrypted at rest, never returned",
    )
    mode: Literal["block", "monitor"] = Field(
        default="monitor",
        description="block rejects a flagged request with 403; monitor annotates the response and forwards it",
    )
    on_unavailable: Literal["block", "monitor"] = Field(
        default="block",
        description="What a block-mode entry does when the guardrails service cannot be reached at all",
    )
    validate_kwargs: dict[str, Any] | None = Field(
        default=None, description="Extra kwargs forwarded to the guardrails service /validate call"
    )
    enabled: bool = Field(default=True, description="False stops the guardrail everywhere without discarding it")
    applies_to_all_workspaces: bool = Field(
        default=False,
        description=(
            "True runs this in every workspace of the organization, including one created later; "
            "false runs it only in the workspaces named by workspace_ids"
        ),
    )
    workspace_ids: list[uuid.UUID] = Field(
        default_factory=list,
        max_length=MAX_SCOPED_WORKSPACES,
        description="Workspaces this guardrail runs in. Must be empty when applies_to_all_workspaces is true",
    )

    @model_validator(mode="after")
    def _reject_redundant_scope(self) -> OrganizationGuardrailCreate:
        """Refuse a workspace list alongside ``applies_to_all_workspaces``.

        The two say different things about the same guardrail, and the flag
        wins at resolve time, so accepting both would store a list that never
        decides anything while reading as though it does.
        """
        if self.applies_to_all_workspaces and self.workspace_ids:
            raise ValueError("workspace_ids must be empty when applies_to_all_workspaces is true")
        return self


class OrganizationGuardrailUpdate(BaseModel):
    """Partial update. Only the fields the caller sets are applied.

    ``credential`` and ``url`` have three states rather than two, which is what
    a write-only field and its nullable partner need: omit to leave the stored
    value alone, send ``""`` to clear it, send a value to replace it. An
    explicit ``null`` also leaves them alone, matching
    `WorkspaceMcpServerUpdate`: a client that serializes its whole form back,
    with an empty credential box it never filled in, must not destroy a
    credential it was never shown.

    ``workspace_ids`` replaces the scope whole when sent; ``[]`` clears it.
    """

    # ``SkipJsonSchema[None]`` on the fields backing NOT NULL columns: ``None``
    # is this schema's "not sent" marker and the validator below refuses it as a
    # value, so the published schema must not advertise ``null`` as accepted.
    # Without it the OpenAPI spec and the generated TypeScript client tell a
    # client that ``{"profile": null}`` is valid and the server answers 422.
    profile: Annotated[str, Field(min_length=1, max_length=128)] | SkipJsonSchema[None] = None
    url: str | None = Field(default=None, max_length=2048)
    credential: str | None = Field(default=None, max_length=8192)
    mode: Literal["block", "monitor"] | SkipJsonSchema[None] = None
    on_unavailable: Literal["block", "monitor"] | SkipJsonSchema[None] = None
    validate_kwargs: dict[str, Any] | None = None
    enabled: bool | SkipJsonSchema[None] = None
    applies_to_all_workspaces: bool | SkipJsonSchema[None] = None
    workspace_ids: list[uuid.UUID] | None = Field(default=None, max_length=MAX_SCOPED_WORKSPACES)

    @model_validator(mode="after")
    def _reject_explicit_nulls(self) -> OrganizationGuardrailUpdate:
        """Refuse an explicit ``null`` for a column that cannot hold one.

        Caught here rather than at the flush, for the reason
        `WorkspaceMcpServerUpdate` catches it: a ``NOT NULL`` violation arrives
        as the same ``IntegrityError`` the unique index raises, and the service
        would report a profile collision that never happened.
        """
        nulled = [
            field
            for field in ("profile", "mode", "on_unavailable", "enabled", "applies_to_all_workspaces")
            if field in self.model_fields_set and getattr(self, field) is None
        ]
        if nulled:
            raise ValueError(f"{', '.join(nulled)} cannot be null; omit the field to leave it unchanged")
        return self


class OrganizationGuardrailPublic(BaseModel):
    """The API-facing shape. Never carries the credential, only whether one is set."""

    id: uuid.UUID
    organization_id: uuid.UUID
    profile: str
    url: str | None
    has_credential: bool
    mode: str
    on_unavailable: str
    validate_kwargs: dict[str, Any] | None
    enabled: bool
    applies_to_all_workspaces: bool
    # Empty for an entry that applies to every workspace, where the scope rows
    # are not consulted at all: reporting a list there would invite a client to
    # edit one that decides nothing.
    workspace_ids: list[uuid.UUID]
    created_at: str
    updated_at: str

    @classmethod
    def from_model(
        cls, guardrail: OrganizationGuardrail, *, workspace_ids: list[uuid.UUID]
    ) -> OrganizationGuardrailPublic:
        return cls(
            id=guardrail.id,
            organization_id=guardrail.organization_id,
            profile=guardrail.profile,
            url=guardrail.url,
            has_credential=guardrail.encrypted_credential is not None,
            mode=guardrail.mode,
            on_unavailable=guardrail.on_unavailable,
            validate_kwargs=guardrail.validate_kwargs,
            enabled=guardrail.enabled,
            applies_to_all_workspaces=guardrail.applies_to_all_workspaces,
            workspace_ids=[] if guardrail.applies_to_all_workspaces else workspace_ids,
            created_at=guardrail.created_at.isoformat(),
            updated_at=guardrail.updated_at.isoformat(),
        )


class OrganizationGuardrailsPublic(BaseModel):
    data: list[OrganizationGuardrailPublic]
    count: int


@dataclass(frozen=True)
class ResolvedOrganizationGuardrail:
    """One organization guardrail as the request path reads it.

    A value type rather than the ORM row, for the reason
    `ResolvedCodeExecutionPolicy` is one: the admission check must not lazily
    touch the session after the request has moved on, and nothing ORM-identified
    should ride into a streaming response that outlives the handler.

    The credential is carried beside the config rather than on it because
    :class:`GuardrailConfig` is a *request body* model. A credential field there
    would be one a caller could set, which would turn the guardrail list into a
    way to make this gateway send a secret to an endpoint of the caller's
    choosing.
    """

    config: GuardrailConfig
    credential: str | None


async def resolve_organization_guardrails(
    db: AsyncSession,
    *,
    organization_id: uuid.UUID,
    workspace_id: uuid.UUID,
) -> list[ResolvedOrganizationGuardrail]:
    """The organization's guardrails that are in effect for one workspace.

    One indexed read, on every request that reaches a completion endpoint in
    standalone mode, which is inherent rather than incidental: a mandate nobody
    looked for is not a mandate. It is deliberately not cached and not overlaid
    onto the config object, per the seam #678 settled: an organization that turns
    a guardrail on expects the next request to run it, not the next process.

    No authorization check, and none is missing: both ids come off the key that
    authenticated the request (`services/workspace_scope.py`), never off a
    header.

    Ordered by profile so a request's guardrails run in a stable order and a
    test can assert one.

    Raises `secret_box.SecretDecryptionError` when a stored credential will not
    decrypt. Running the check unauthenticated instead would send the request to
    an endpoint the organization configured a credential for and take whatever
    it said, which for a ``block`` guardrail is an enforcement decision made on
    a call that was never authorized.
    """
    stmt = (
        select(OrganizationGuardrail)
        .outerjoin(
            OrganizationGuardrailWorkspace,
            and_(
                OrganizationGuardrailWorkspace.organization_guardrail_id == OrganizationGuardrail.id,
                OrganizationGuardrailWorkspace.workspace_id == workspace_id,
            ),
        )
        .where(
            OrganizationGuardrail.organization_id == organization_id,
            OrganizationGuardrail.enabled.is_(True),
            or_(
                OrganizationGuardrail.applies_to_all_workspaces.is_(True),
                OrganizationGuardrailWorkspace.workspace_id.is_not(None),
            ),
        )
        .order_by(OrganizationGuardrail.profile)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        ResolvedOrganizationGuardrail(
            config=GuardrailConfig(
                profile=row.profile,
                url=row.url,
                mode=_stored_mode(row.mode),
                on_unavailable=_stored_mode(row.on_unavailable),
                validate_kwargs=row.validate_kwargs or {},
            ),
            credential=decrypt_secret(row.encrypted_credential) if row.encrypted_credential else None,
        )
        for row in rows
    ]


def _stored_mode(value: str) -> Literal["block", "monitor"]:
    """Narrow a stored mode string, resolving anything unexpected to ``block``.

    The column is a plain string and the two request schemas, whose fields are
    already ``Literal["block", "monitor"]``, are its only writers, so this is
    unreachable through the API. It resolves to the enforcing side rather than
    the observing one because the alternative is a guardrail that silently stops
    enforcing when something writes around those schemas, which is the failure
    a security control must not have.
    """
    return "monitor" if value == "monitor" else "block"


async def _validate_url(url: str, *, has_credential: bool) -> None:
    """Run the same SSRF/TLS check a request-body guardrail URL faces, at write time.

    Checked here as well as on the request path
    (`services.guardrails.run_input_guardrails` validates every entry's URL)
    rather than instead of it: this catches an operator's mistake at the moment
    they make it, and the request-path check is what still holds when DNS moves
    under a URL that was safe when it was stored.
    """
    try:
        await validate_mcp_url(url, has_authorization_token=has_credential)
    except UnsafeURLError as exc:
        raise OrganizationGuardrailUnsafeUrlError(str(exc)) from exc


def _encrypted(credential: str) -> str:
    try:
        return encrypt_secret(credential)
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError("organization guardrail credentials") from None


class OrganizationGuardrailService:
    """CRUD for the caller's organization's guardrails. Writes are management-gated."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.organizations = OrganizationService(db)

    async def _manageable_organization_id(self, user: User) -> uuid.UUID:
        """The caller's organization, having checked they may manage its guardrails.

        One gate for reads and writes alike, unlike
        `organization_pricing_service` next door, which opens its list to any
        member. A row here names an external endpoint this gateway connects to
        and says which of them carry a credential, which is
        `workspace_mcp_server_service`'s reason for gating its list too. Rates
        are not a secret from the people billed at them; an organization's
        guardrail topology is not something every member needs.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(
            user=user,
            organization=organization,
        )
        return organization.id

    async def _get_or_404(self, organization_id: uuid.UUID, guardrail_id: uuid.UUID) -> OrganizationGuardrail:
        guardrail = await self.db.get(OrganizationGuardrail, guardrail_id)
        if guardrail is None or guardrail.organization_id != organization_id:
            raise OrganizationGuardrailNotFoundError(guardrail_id)
        return guardrail

    async def _scope_ids(self, guardrail_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[uuid.UUID]]:
        """The scoped workspace ids for a page of guardrails, in one query.

        One read for the whole page rather than one per row: the list endpoint
        would otherwise be an N+1 over a table an organization edits by hand.
        """
        if not guardrail_ids:
            return {}
        rows = (
            await self.db.execute(
                select(
                    OrganizationGuardrailWorkspace.organization_guardrail_id,
                    OrganizationGuardrailWorkspace.workspace_id,
                ).where(OrganizationGuardrailWorkspace.organization_guardrail_id.in_(guardrail_ids))
            )
        ).all()
        scoped: dict[uuid.UUID, list[uuid.UUID]] = {guardrail_id: [] for guardrail_id in guardrail_ids}
        for guardrail_id, workspace_id in rows:
            scoped[guardrail_id].append(workspace_id)
        return scoped

    async def _public(self, guardrail: OrganizationGuardrail) -> OrganizationGuardrailPublic:
        scoped = await self._scope_ids([guardrail.id])
        return OrganizationGuardrailPublic.from_model(guardrail, workspace_ids=scoped.get(guardrail.id, []))

    async def _require_workspaces_in_organization(
        self, organization_id: uuid.UUID, workspace_ids: list[uuid.UUID]
    ) -> list[uuid.UUID]:
        """Refuse the whole write if a scope entry names a workspace elsewhere.

        Checked before anything is written, so a foreign or unknown workspace id
        fails the request rather than silently dropping that one from the scope
        and leaving a guardrail narrower than the operator believes.
        A workspace in another organization is reported as not found, like
        everywhere else, so this is not an existence oracle across tenants.
        """
        requested = list(dict.fromkeys(workspace_ids))
        if not requested:
            return []
        found = {
            workspace.id
            for workspace in await WorkspaceRepository(self.db).get_by_ids(set(requested))
            if workspace.organization_id == organization_id
        }
        missing = set(requested) - found
        if missing:
            raise WorkspaceNotFoundError(next(iter(sorted(missing, key=str))))
        return requested

    async def _replace_scope(self, guardrail_id: uuid.UUID, workspace_ids: list[uuid.UUID]) -> None:
        """Set the guardrail's scope to exactly ``workspace_ids``."""
        await self.db.execute(
            delete(OrganizationGuardrailWorkspace).where(
                OrganizationGuardrailWorkspace.organization_guardrail_id == guardrail_id
            )
        )
        for workspace_id in workspace_ids:
            self.db.add(
                OrganizationGuardrailWorkspace(organization_guardrail_id=guardrail_id, workspace_id=workspace_id)
            )

    async def list_guardrails(
        self,
        *,
        user: User,
        skip: int = 0,
        limit: int = 100,
    ) -> OrganizationGuardrailsPublic:
        """One page of the organization's guardrails, and the total."""
        organization_id = await self._manageable_organization_id(user)
        limit = min(limit, _MAX_LIST_LIMIT)
        total = (
            await self.db.execute(
                select(func.count())
                .select_from(OrganizationGuardrail)
                .where(OrganizationGuardrail.organization_id == organization_id)
            )
        ).scalar_one()
        rows = list(
            (
                await self.db.execute(
                    select(OrganizationGuardrail)
                    .where(OrganizationGuardrail.organization_id == organization_id)
                    .order_by(OrganizationGuardrail.profile)
                    .offset(skip)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        scoped = await self._scope_ids([row.id for row in rows])
        return OrganizationGuardrailsPublic(
            data=[OrganizationGuardrailPublic.from_model(row, workspace_ids=scoped.get(row.id, [])) for row in rows],
            count=total,
        )

    async def create_guardrail(
        self, *, user: User, request: OrganizationGuardrailCreate
    ) -> OrganizationGuardrailPublic:
        """Mandate a guardrail, encrypting its credential before it is stored."""
        organization_id = await self._manageable_organization_id(user)

        existing = (
            await self.db.execute(
                select(func.count())
                .select_from(OrganizationGuardrail)
                .where(OrganizationGuardrail.organization_id == organization_id)
            )
        ).scalar_one()
        if existing >= MAX_GUARDRAILS_PER_ORGANIZATION:
            raise OrganizationGuardrailLimitReachedError(MAX_GUARDRAILS_PER_ORGANIZATION)

        url = _blank_to_none(request.url)
        credential = _blank_to_none(request.credential)
        if url is not None:
            await _validate_url(url, has_credential=credential is not None)
        workspace_ids = await self._require_workspaces_in_organization(
            organization_id, [] if request.applies_to_all_workspaces else request.workspace_ids
        )

        guardrail = OrganizationGuardrail(
            organization_id=organization_id,
            profile=request.profile.strip(),
            url=url,
            encrypted_credential=_encrypted(credential) if credential else None,
            mode=request.mode,
            on_unavailable=request.on_unavailable,
            validate_kwargs=request.validate_kwargs or None,
            enabled=request.enabled,
            applies_to_all_workspaces=request.applies_to_all_workspaces,
        )
        self.db.add(guardrail)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            await self.db.rollback()
            raise OrganizationGuardrailAlreadyExistsError(request.profile.strip()) from exc

        await self._replace_scope(guardrail.id, workspace_ids)
        await self._commit()
        await self.db.refresh(guardrail)
        return await self._public(guardrail)

    async def update_guardrail(
        self, *, user: User, guardrail_id: uuid.UUID, request: OrganizationGuardrailUpdate
    ) -> OrganizationGuardrailPublic:
        """Apply the fields this request set, leaving the rest as they were."""
        organization_id = await self._manageable_organization_id(user)
        guardrail = await self._get_or_404(organization_id, guardrail_id)

        fields = request.model_fields_set
        # Resolved before anything is written, because the URL check and the
        # credential together decide whether an http endpoint is admissible: a
        # request that adds a credential to a stored http URL must be refused
        # even though it never mentioned the URL.
        new_credential: Any = _UNSET
        if "credential" in fields and request.credential is not None:
            new_credential = _blank_to_none(request.credential)
        effective_has_credential = (
            (new_credential is not None) if new_credential is not _UNSET else guardrail.encrypted_credential is not None
        )
        new_url: Any = _UNSET
        if "url" in fields and request.url is not None:
            new_url = _blank_to_none(request.url)
        effective_url = new_url if new_url is not _UNSET else guardrail.url
        if effective_url is not None:
            await _validate_url(effective_url, has_credential=effective_has_credential)

        applies_to_all = (
            request.applies_to_all_workspaces
            if request.applies_to_all_workspaces is not None
            else guardrail.applies_to_all_workspaces
        )
        scope_change: list[uuid.UUID] | None = None
        if applies_to_all:
            if request.workspace_ids:
                raise OrganizationGuardrailScopeConflictError()
            # Cleared rather than left inert. The resolver ignores these rows
            # while the flag is set, so keeping them would change nothing today
            # and reinstate a workspace set nobody had looked at the moment
            # somebody switched the flag back off. Empty is the conservative
            # direction: the entry then runs nowhere until a workspace is chosen.
            scope_change = []
        elif request.workspace_ids is not None:
            scope_change = await self._require_workspaces_in_organization(organization_id, request.workspace_ids)

        if new_url is not _UNSET:
            guardrail.url = new_url
        if new_credential is not _UNSET:
            guardrail.encrypted_credential = _encrypted(new_credential) if new_credential else None
        if request.profile is not None:
            guardrail.profile = request.profile.strip()
        if request.mode is not None:
            guardrail.mode = request.mode
        if request.on_unavailable is not None:
            guardrail.on_unavailable = request.on_unavailable
        if "validate_kwargs" in fields:
            guardrail.validate_kwargs = request.validate_kwargs or None
        if request.enabled is not None:
            guardrail.enabled = request.enabled
        if request.applies_to_all_workspaces is not None:
            guardrail.applies_to_all_workspaces = request.applies_to_all_workspaces

        if scope_change is not None:
            await self._replace_scope(guardrail.id, scope_change)

        # Read off the row here, before the commit: `_commit` rolls back on
        # failure, which expires every instance in the session, so reading
        # `guardrail.profile` in the handler below would be a lazy load in a
        # place that cannot await one.
        attempted_profile = guardrail.profile
        try:
            await self._commit()
        except IntegrityError as exc:
            raise OrganizationGuardrailAlreadyExistsError(attempted_profile) from exc
        await self.db.refresh(guardrail)
        return await self._public(guardrail)

    async def delete_guardrail(self, *, user: User, guardrail_id: uuid.UUID) -> None:
        """Drop the guardrail and its scope rows, which cascade with it."""
        organization_id = await self._manageable_organization_id(user)
        guardrail = await self._get_or_404(organization_id, guardrail_id)
        await self.db.delete(guardrail)
        await self._commit()

    async def _commit(self) -> None:
        """Commit, rolling back before any failure escapes.

        Required rather than tidy, the same as
        `workspace_code_execution_policy_service._commit`: SQLAlchemy leaves a
        session with a failed flush unusable, so a caller that skips the
        rollback gets ``PendingRollbackError`` from the next statement instead
        of the error that actually happened. ``IntegrityError`` is re-raised for
        the unique-profile handling above, after the rollback.
        """
        try:
            await self.db.commit()
        except IntegrityError:
            await self.db.rollback()
            raise
        except Exception:
            await self.db.rollback()
            raise


def _blank_to_none(value: str | None) -> str | None:
    """Treat a whitespace-only value as absent.

    A cleared text input arrives as ``""``, which is how this surface's
    three-state fields say "clear it"; storing the empty string instead would
    read as configured while carrying nothing.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
