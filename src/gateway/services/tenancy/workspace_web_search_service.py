"""A workspace's limits on the deployment-wide web search backend.

The backend is an operator concern and stays one, so nothing here can point a workspace somewhere else.
A workspace's policy decides who on this deployment may search, and how far their searches may reach.

A policy may veto and may narrow, and it never grants.
How it narrows one request is the tools domain's rule, in ``services/tools/_web_access.py``.

Reading or writing a stored policy requires an owner or admin of the workspace or of its organization.
Reads are gated as well as writes, because the row is the workspace's posture and not one member's allowance.

:func:`resolve_workspace_web_search_config` reads a stored row.
:func:`read_web_search_policy` reads a control plane's answer.
None of them takes an identity, because the workspace comes from the authenticated key.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import User, Workspace
from gateway.models.tools import ResolvedWebSearchConfig, WorkspaceWebSearchConfig
from gateway.services.tenancy import authorization
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.web_retrieval_backend import MAX_RESULTS_CAP
from gateway.services.web_retrieval_policy import read_domain_list

# The backend's own ceiling on returned hits. A stored value above it would read
# as a configured limit and do nothing, since the backend clamps to this anyway,
# so it is refused at the write rather than silently clamped at resolve time.
# Same call as the code execution policy service makes for its two.
_MAX_RESULTS = MAX_RESULTS_CAP
# Bound the opaque bag so one workspace's row cannot grow without limit; the
# same numbers the hosted `WorkspaceWebSearchConfigUpdate` uses, since this is
# the same configuration.
_MAX_PROVIDER_OPTION_KEYS = 30
_MAX_PROVIDER_OPTIONS_BYTES = 4096
_MAX_PURPOSE_HINT_LENGTH = 2048


class InvalidStoredWebSearchDomainError(ValueError):
    """A legacy workspace row contains a domain rule that cannot be enforced."""


def _normalize_domains(value: list[str] | None) -> list[str] | None:
    """Read a domain list a caller wrote, where a blank entry is an empty form field and is dropped."""
    if value is None:
        return None
    hosts = read_domain_list([raw for raw in value if not (isinstance(raw, str) and not raw.strip())])
    return list(hosts) if hosts else None


def _check_provider_options(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Bound the opaque bag by key count and by serialized size.

    Both, not just the count: a handful of provider knobs need a few hundred
    bytes, and the key count alone would still admit a multi-megabyte value.
    """
    if value is None:
        return None
    if len(value) > _MAX_PROVIDER_OPTION_KEYS:
        raise ValueError(f"at most {_MAX_PROVIDER_OPTION_KEYS} provider_options keys are allowed")
    if len(json.dumps(value, default=str)) > _MAX_PROVIDER_OPTIONS_BYTES:
        raise ValueError(f"provider_options must serialize to at most {_MAX_PROVIDER_OPTIONS_BYTES} bytes")
    return value or None


class WorkspaceWebSearchConfigUpdate(BaseModel):
    """The configuration to store for a workspace, as a whole.

    ``PUT`` semantics, ported from the hosted ``WorkspaceWebSearchConfigUpdate``:
    what is sent is what the workspace has afterwards, so an omitted field is
    cleared rather than left as it was.
    """

    # Required rather than defaulted (the hosted model defaults it to false),
    # for the reason `WorkspaceCodeExecutionPolicyUpdate.enabled` is: there, no
    # row means disabled, so an omitted flag and the stored default agree. Here
    # no row means *unnarrowed*, so either default would surprise somebody.
    enabled: bool = Field(
        description=(
            "False refuses web access for this workspace through otari_web_search, "
            "otari_web_fetch, and POST /api/v1/search."
        )
    )
    max_results: int | None = Field(
        default=None,
        gt=0,
        le=_MAX_RESULTS,
        description=(
            "Search only: ceiling on results one search returns; only ever lowers "
            f"the effective limit, so at most {_MAX_RESULTS}"
        ),
    )
    purpose_hint: str | None = Field(
        default=None,
        max_length=_MAX_PURPOSE_HINT_LENGTH,
        description="Search only: hint used when a request declares otari_web_search without one of its own",
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        description=(
            "Filters Search results and constrains initial and redirected Fetch destinations; "
            "intersected with any list the request sends"
        ),
    )
    blocked_domains: list[str] | None = Field(
        default=None,
        description=(
            "Filters Search results and blocks initial and redirected Fetch destinations; "
            "added to any list the request sends"
        ),
    )
    provider_options: dict[str, Any] | None = Field(
        default=None,
        description="Search only: provider-specific knobs forwarded to the backend; request keys win",
    )

    @field_validator("allowed_domains", "blocked_domains")
    @classmethod
    def _check_domains(cls, value: list[str] | None) -> list[str] | None:
        return _normalize_domains(value)

    @field_validator("provider_options")
    @classmethod
    def _validate_provider_options(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return _check_provider_options(value)


class WorkspaceWebSearchConfigPublic(BaseModel):
    """A workspace's web-search configuration, or the unconfigured one it has without a row."""

    workspace_id: uuid.UUID
    # False when the workspace has no row: everything below is then the
    # deployment's own behavior rather than a stored decision, which is what a
    # dashboard needs to say "not configured" instead of showing a policy
    # nobody set.
    configured: bool
    # Whether this deployment can run web search at all, i.e. whether an
    # operator has pointed it at a backend. The counterpart of
    # `WorkspaceCodeExecutionPolicyPublic.sandbox_configured`, and the ceiling
    # this row sits under: a workspace switched on where the deployment has no
    # backend reads as unavailable rather than as working.
    web_search_configured: bool
    enabled: bool
    max_results: int | None
    purpose_hint: str | None
    allowed_domains: list[str] | None
    blocked_domains: list[str] | None
    provider_options: dict[str, Any] | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def unconfigured(cls, workspace_id: uuid.UUID, *, web_search_configured: bool) -> WorkspaceWebSearchConfigPublic:
        return cls(
            workspace_id=workspace_id,
            configured=False,
            web_search_configured=web_search_configured,
            enabled=True,
            max_results=None,
            purpose_hint=None,
            allowed_domains=None,
            blocked_domains=None,
            provider_options=None,
            created_at=None,
            updated_at=None,
        )

    @classmethod
    def from_model(
        cls, config: WorkspaceWebSearchConfig, *, web_search_configured: bool
    ) -> WorkspaceWebSearchConfigPublic:
        return cls(
            workspace_id=config.workspace_id,
            configured=True,
            web_search_configured=web_search_configured,
            enabled=config.enabled,
            max_results=config.max_results,
            purpose_hint=config.purpose_hint,
            allowed_domains=config.allowed_domains,
            blocked_domains=config.blocked_domains,
            provider_options=config.provider_options,
            created_at=config.created_at.isoformat(),
            updated_at=config.updated_at.isoformat(),
        )


async def resolve_workspace_web_search_config(
    db: AsyncSession,
    workspace_id: uuid.UUID,
) -> ResolvedWebSearchConfig | None:
    """The workspace's stored configuration, or ``None`` when it has none.

    ``None`` and "a row that narrows nothing" are deliberately the same outcome
    for the caller; the distinction only matters to the management surface,
    which reports it as ``configured``.
    """
    config = await db.get(WorkspaceWebSearchConfig, workspace_id)
    if config is None:
        return None
    return ResolvedWebSearchConfig(
        enabled=config.enabled,
        max_results=config.max_results,
        purpose_hint=config.purpose_hint,
        allowed_domains=_stored_domains(config.allowed_domains),
        blocked_domains=_stored_domains(config.blocked_domains),
        provider_options=config.provider_options,
        authorized_tools=None,
    )


def read_web_search_policy(answer: Mapping[str, Any]) -> ResolvedWebSearchConfig:
    """Read the control plane's answer for one workspace's web search policy.

    Raises ``ValueError`` when a recognized field is malformed, so a policy that cannot be read fails closed.
    """
    enabled = answer.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    max_results = answer.get("max_results")
    if max_results is not None and (
        not isinstance(max_results, int) or isinstance(max_results, bool) or not 1 <= max_results <= _MAX_RESULTS
    ):
        raise ValueError(f"max_results must be an integer from 1 to {_MAX_RESULTS}")
    purpose_hint = answer.get("purpose_hint")
    if purpose_hint is not None and (not isinstance(purpose_hint, str) or len(purpose_hint) > _MAX_PURPOSE_HINT_LENGTH):
        raise ValueError(f"purpose_hint must be a string of at most {_MAX_PURPOSE_HINT_LENGTH} characters")
    provider_options = answer.get("provider_options")
    if provider_options is not None and not isinstance(provider_options, dict):
        raise ValueError("provider_options must be an object")
    provider_options = _check_provider_options(provider_options)
    return ResolvedWebSearchConfig(
        enabled=enabled,
        max_results=max_results,
        purpose_hint=_blank_to_none(purpose_hint),
        allowed_domains=read_domain_list(answer.get("allowed_domains")),
        blocked_domains=read_domain_list(answer.get("blocked_domains")),
        provider_options=provider_options,
        authorized_tools=None,
    )


def _stored_domains(value: object) -> tuple[str, ...] | None:
    """Read a stored row's domain list, refusing one that cannot be enforced."""
    try:
        return read_domain_list(value)
    except ValueError as exc:
        raise InvalidStoredWebSearchDomainError("stored web-search domain list is invalid") from exc


class WorkspaceWebSearchService:
    """Read, upsert and clear one workspace's web-search configuration."""

    def __init__(self, db: AsyncSession, *, web_search_configured: bool):
        self.db = db
        self.organizations = OrganizationService(db, membership_listener=None, workspace_listener=None)
        # Passed in rather than read here: whether a backend is configured is a
        # question about the running deployment's config, which the route layer
        # already holds and a service has no business reaching for.
        self.web_search_configured = web_search_configured

    async def get_config(self, *, user: User, workspace_id: uuid.UUID) -> WorkspaceWebSearchConfigPublic:
        """The workspace's configuration. Reading it takes the same role as setting it."""
        workspace = await self._resolve_manageable(user=user, workspace_id=workspace_id)
        config = await self.db.get(WorkspaceWebSearchConfig, workspace.id)
        if config is None:
            return self._unconfigured(workspace.id)
        return WorkspaceWebSearchConfigPublic.from_model(config, web_search_configured=self.web_search_configured)

    async def set_config(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        request: WorkspaceWebSearchConfigUpdate,
    ) -> WorkspaceWebSearchConfigPublic:
        """Store the workspace's configuration, replacing any existing one.

        Read-then-insert with nothing locking the gap, so two writers can both
        find no row and both insert; the primary key refuses the second. That
        refusal is not a conflict anybody needs to hear about, because a ``PUT``
        of the whole configuration is idempotent: the loser re-reads the row the
        winner created and applies its own values over it, which is the same
        outcome it would have reached had it arrived a moment later. The same
        handling as the code execution policy service, for the same
        reason: one row per workspace and nothing to disambiguate.
        """
        workspace = await self._resolve_manageable(user=user, workspace_id=workspace_id)
        # Read off the row once, here: a rollback below expires every instance in
        # the session, so `workspace.id` after one is a lazy load in a place that
        # cannot await it.
        resolved_id = workspace.id

        config = await self.db.get(WorkspaceWebSearchConfig, resolved_id)
        if config is None:
            config = WorkspaceWebSearchConfig(workspace_id=resolved_id)
            self.db.add(config)
        self._apply(config, request)

        try:
            await self._commit()
        except IntegrityError:
            # `_commit` has already rolled back, which is what makes the read
            # below usable: a failed flush leaves the session unusable, so
            # anything attempted before a rollback raises `PendingRollbackError`
            # and masks this.
            config = await self.db.get(WorkspaceWebSearchConfig, resolved_id)
            if config is None:
                raise  # not the race: nothing is there to have collided with
            self._apply(config, request)
            await self._commit()
        await self.db.refresh(config)
        return WorkspaceWebSearchConfigPublic.from_model(config, web_search_configured=self.web_search_configured)

    async def clear_config(self, *, user: User, workspace_id: uuid.UUID) -> WorkspaceWebSearchConfigPublic:
        """Drop the workspace's configuration, returning it to the deployment's behavior.

        Idempotent: a workspace that has no row is already in the state this
        asks for, so it answers with the unconfigured shape rather than a 404.
        """
        workspace = await self._resolve_manageable(user=user, workspace_id=workspace_id)

        config = await self.db.get(WorkspaceWebSearchConfig, workspace.id)
        if config is not None:
            await self.db.delete(config)
            await self._commit()
        return self._unconfigured(workspace.id)

    async def _commit(self) -> None:
        """Commit, rolling back before any failure escapes.

        Every write path here goes through this rather than repeating the
        pattern, because the rollback is required and not tidy: SQLAlchemy
        leaves a session with a failed flush unusable, so a caller that skips it
        gets `PendingRollbackError` from the next statement instead of the error
        that actually happened.
        """
        try:
            await self.db.commit()
        except SQLAlchemyError:
            await self.db.rollback()
            raise

    @staticmethod
    def _apply(config: WorkspaceWebSearchConfig, request: WorkspaceWebSearchConfigUpdate) -> None:
        """Write the whole request onto the row. Every field, since this is a ``PUT``."""
        config.enabled = request.enabled
        config.max_results = request.max_results
        config.purpose_hint = _blank_to_none(request.purpose_hint)
        config.allowed_domains = request.allowed_domains
        config.blocked_domains = request.blocked_domains
        config.provider_options = request.provider_options

    def _unconfigured(self, workspace_id: uuid.UUID) -> WorkspaceWebSearchConfigPublic:
        return WorkspaceWebSearchConfigPublic.unconfigured(
            workspace_id, web_search_configured=self.web_search_configured
        )

    async def _resolve_manageable(self, *, user: User, workspace_id: uuid.UUID) -> Workspace:
        """Resolve a workspace the caller may see *and* manage.

        Visibility first, so a workspace the caller may not see answers 404
        rather than 403 and stays indistinguishable from one that does not
        exist; the role check then answers 403 for a member who may read the
        workspace but not set its posture.
        """
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        await authorization.require_workspace_management_access(
            self.db, user=user, workspace=workspace, organizations=self.organizations
        )
        return workspace


def _blank_to_none(value: str | None) -> str | None:
    """Treat a whitespace-only hint as absent.

    A cleared text input arrives as ``""``, and storing that would set the
    workspace's hint to an empty string, which reads as "configured" while
    injecting nothing.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
