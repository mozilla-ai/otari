"""A workspace's configuration over the deployment-wide web-search backend.

The backend is an operator concern and stays one: ``web_search_url`` names it,
and any credential belongs to the adapter sitting in front of it, so nothing
here can point a workspace somewhere else. What a row decides is *who on this
deployment may search, and how far their searches may reach*.

Composition follows the rule in ``src/gateway/AGENTS.md`` (#655, settled in
#678): a workspace row may veto and may refine, never grant. So

* ``enabled=False`` refuses ``otari_web_search`` for the workspace;
* ``max_results`` is floored against what the request would otherwise get,
  which is the request's own value or, failing that, the deployment's, so a
  workspace ceiling can only shrink a search;
* ``blocked_domains`` is *added* to the request's own block-list, and
  ``allowed_domains`` is *intersected* with the request's, so neither list can
  be shed by a request that sends one of its own;
* ``purpose_hint`` fills in only when the request named none;
* and **no row means no narrowing**, which is what makes a deployment that
  configures nothing behave exactly as it did.

That parts company with the hybrid path in ``prepare_gateway_tools``, which
treats the policy it resolves from otari.ai as a set of *defaults* a request
overrides. The precedence there is the platform's own contract and is left
alone; the rule above is this repository's, and it is the stricter of the two:
under default-only precedence a request could shed a workspace's block-list
simply by sending a block-list of its own, which is a guardrail that fails open.

``provider_options`` is the one field that keeps the hybrid precedence, merged
per key with the request winning. It is an opaque bag forwarded to the backend
adapter rather than something this gateway enforces, so there is no narrowing
relation between two values of it to apply.

The CRUD half is master-key routed (``routes/workspace_web_search.py``) and
role-gated per workspace: an organization owner/admin, or an owner/admin of the
workspace itself, may read *and* write it. Reads are gated for the same reason
``workspace_code_execution_policy_service`` gates them: the row is the
workspace's posture, not one member's allowance. It is looser than the hosted
service, which admits organization owners/admins only, because
``authorization.require_workspace_management_access`` is the gate every other
per-workspace surface in this repository uses.

The request-path half is :func:`resolve_workspace_web_search_config` plus the
pure :func:`narrow_web_search_tool_entry`, neither of which takes an identity:
the caller has already authenticated, and the workspace comes off the key,
never off a header (``services/workspace_scope``).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import WorkspaceWebSearchConfig
from gateway.models.tenancy import User, Workspace
from gateway.services.tenancy import authorization
from gateway.services.tenancy.errors import WorkspaceWebSearchDomainsExcludedError
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.web_search_backend import MAX_RESULTS_CAP

# The backend's own ceiling on returned hits. A stored value above it would read
# as a configured limit and do nothing, since the backend clamps to this anyway,
# so it is refused at the write rather than silently clamped at resolve time.
# Same call as `workspace_code_execution_policy_service` makes for its two.
_MAX_RESULTS = MAX_RESULTS_CAP
# Bound the two lists and the opaque bag so one workspace's row cannot grow
# without limit; the same numbers the hosted `WorkspaceWebSearchConfigUpdate`
# uses, since this is the same configuration.
_MAX_DOMAINS = 100
_MAX_PROVIDER_OPTION_KEYS = 30
_MAX_PROVIDER_OPTIONS_BYTES = 4096
# The longest a DNS name can be. Not a policy, just the point past which a
# string cannot be a host and is therefore a mistake worth naming at the write.
_MAX_DOMAIN_LENGTH = 253


def _normalize_domains(value: list[str] | None) -> list[str] | None:
    """Lower-case, strip, drop empties, and de-duplicate a domain list.

    Ported from the hosted model's validator. An all-blank list normalizes to
    ``None`` rather than ``[]``, because an empty list here would read as "an
    allow-list permitting nothing" to a reader and as "no allow-list" to
    :func:`narrow_web_search_tool_entry`, and only one of those is what a
    cleared form means.
    """
    if value is None:
        return None
    seen: dict[str, None] = {}
    for raw in value:
        host = raw.strip().lower()
        if not host:
            continue
        if len(host) > _MAX_DOMAIN_LENGTH:
            raise ValueError(f"a domain may be at most {_MAX_DOMAIN_LENGTH} characters")
        seen.setdefault(host, None)
    cleaned = list(seen)
    if len(cleaned) > _MAX_DOMAINS:
        raise ValueError(f"at most {_MAX_DOMAINS} domains are allowed")
    return cleaned or None


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
    enabled: bool = Field(description="False refuses web search for this workspace")
    max_results: int | None = Field(
        default=None,
        gt=0,
        le=_MAX_RESULTS,
        description=(
            f"Ceiling on results one search returns; only ever lowers the effective limit, so at most {_MAX_RESULTS}"
        ),
    )
    purpose_hint: str | None = Field(
        default=None,
        max_length=2048,
        description="Hint used when a request declares otari_web_search without one of its own",
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        description="Results are kept only from these domains; intersected with any list the request sends",
    )
    blocked_domains: list[str] | None = Field(
        default=None,
        description="Results from these domains are dropped; added to any list the request sends",
    )
    provider_options: dict[str, Any] | None = Field(
        default=None,
        description="Provider-specific knobs forwarded to the search backend; a request's own keys win",
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


@dataclass(frozen=True)
class ResolvedWebSearchConfig:
    """What the request path reads off a stored configuration.

    A value type rather than the ORM row, so the admission check cannot lazily
    touch the session after it has moved on, and so the tool context carries no
    ORM identity into a streaming response that outlives the request handler.
    """

    enabled: bool
    max_results: int | None
    purpose_hint: str | None
    allowed_domains: tuple[str, ...] | None
    blocked_domains: tuple[str, ...] | None
    provider_options: dict[str, Any] | None


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
        allowed_domains=_as_tuple(config.allowed_domains),
        blocked_domains=_as_tuple(config.blocked_domains),
        provider_options=config.provider_options,
    )


def narrow_web_search_tool_entry(
    tool_entry: dict[str, Any],
    config: ResolvedWebSearchConfig,
    *,
    baseline_max_results: int,
) -> dict[str, Any]:
    """Compose a workspace's configuration onto the request's tool entry.

    Returns a new entry rather than mutating the caller's, which is the one
    reachable from the request body it was extracted from.

    Assumes ``config.enabled``; the veto is the caller's to raise, because only
    the caller knows which error shape the request format wants.

    ``baseline_max_results`` is how many results this request would get without
    a workspace row at all (``routes/_tools.web_search_max_results_baseline``:
    the deployment's own setting, or the backend's built-in). The workspace
    ceiling is floored against it and not merely written in, because writing it
    in would let a workspace whose ceiling sits above the operator's *raise* the
    operator's number, which is the one thing the narrowing rule forbids.

    Raises :class:`WorkspaceWebSearchDomainsExcludedError` when the request
    names an allow-list that overlaps the workspace's nowhere (see
    :func:`_intersect` for what overlapping means when the entries are domain
    suffixes rather than hosts). The
    alternative is an empty effective allow-list, which
    ``_build_web_search_backend`` reads as *no* allow-list because an empty list
    is falsy, and that turns the narrowest possible policy into no policy at
    all. Refusing also tells the caller something a silent zero-result search
    would not.
    """
    narrowed = dict(tool_entry)

    if config.max_results is not None:
        requested_max = narrowed.get("max_results")
        # ``bool`` is an ``int`` subclass, so exclude it: a JSON ``true`` must
        # not be read as a one-result ceiling.
        if not isinstance(requested_max, int) or isinstance(requested_max, bool) or requested_max <= 0:
            requested_max = baseline_max_results
        narrowed["max_results"] = min(requested_max, config.max_results)

    if config.blocked_domains:
        # Union: a workspace block a request could drop by sending a block-list
        # of its own would be a guardrail that fails open.
        narrowed["blocked_domains"] = _union(_entry_domains(narrowed.get("blocked_domains")), config.blocked_domains)

    if config.allowed_domains:
        requested_allowed = _entry_domains(narrowed.get("allowed_domains"))
        if requested_allowed is None:
            narrowed["allowed_domains"] = list(config.allowed_domains)
        else:
            both = _intersect(requested_allowed, config.allowed_domains)
            if not both:
                raise WorkspaceWebSearchDomainsExcludedError()
            narrowed["allowed_domains"] = both

    # A hint informs the model, it does not permit anything, so the request's
    # own wins and the workspace's fills a gap.
    if not narrowed.get("purpose_hint") and config.purpose_hint:
        narrowed["purpose_hint"] = config.purpose_hint

    if config.provider_options:
        request_options = narrowed.get("provider_options")
        narrowed["provider_options"] = (
            {**config.provider_options, **request_options}
            if isinstance(request_options, dict)
            else dict(config.provider_options)
        )

    return narrowed


def _as_tuple(value: list[str] | None) -> tuple[str, ...] | None:
    """Read a stored JSON list back as a tuple of hosts, or ``None`` if it holds none.

    Defensive about the element type because the column is JSON: a row written
    by something other than this service could hold anything, and a non-string
    would otherwise reach the backend's domain comparison.
    """
    if not value:
        return None
    hosts = tuple(str(host).strip().lower() for host in value if str(host).strip())
    return hosts or None


def _entry_domains(value: Any) -> list[str] | None:
    """Read a domain list off a request's tool entry, normalized like a stored one.

    ``None`` for anything that is not a non-empty list, so a malformed or absent
    field reads as "the request named no list" rather than as an empty one.
    """
    if not isinstance(value, list):
        return None
    hosts = [str(host).strip().lower() for host in value if str(host).strip()]
    return hosts or None


def _union(requested: list[str] | None, workspace: tuple[str, ...]) -> list[str]:
    """Every domain either side named, in request-then-workspace order, de-duplicated."""
    merged: dict[str, None] = {}
    for host in (*(requested or ()), *workspace):
        merged.setdefault(host, None)
    return list(merged)


def _intersect(requested: list[str], workspace: tuple[str, ...]) -> list[str]:
    """The domains both sides permit, in the request's order.

    Not a set intersection, because an entry in either list is a *suffix* and not
    a host: ``WebSearchBackend._apply_domain_filters`` keeps a result when its hostname
    equals an entry or ends in ``"." + entry``. So ``example.com`` on the
    workspace's list already covers ``docs.example.com``, and a request naming
    the subdomain is asking for strictly less than the workspace permits rather
    than for something outside it. Whichever side is the narrower of an
    overlapping pair is the one that survives; genuinely disjoint lists still
    intersect to nothing, which is what the caller refuses.
    """
    kept: dict[str, None] = {}
    for host in dict.fromkeys(requested):
        for allowed in workspace:
            if _covers(allowed, host):
                kept.setdefault(host, None)
            elif _covers(host, allowed):
                kept.setdefault(allowed, None)
    return list(kept)


def _covers(suffix: str, host: str) -> bool:
    """Whether a domain-list entry admits a host, the way the search backend decides it."""
    return host == suffix or host.endswith(f".{suffix}")


class WorkspaceWebSearchService:
    """Read, upsert and clear one workspace's web-search configuration."""

    def __init__(self, db: AsyncSession, *, web_search_configured: bool):
        self.db = db
        self.organizations = OrganizationService(db)
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
        handling as ``workspace_code_execution_policy_service``, for the same
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
