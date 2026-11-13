"""Request-body model for the gateway-managed ``guardrails`` field.

Guardrails are *not* a model-callable tool. Unlike ``otari_code_execution`` /
``otari_web_search`` (which the model decides to invoke inside the tool-use
loop), a guardrail is a request-level policy the **caller** opts into: it runs
on the request regardless of what the model decides, and the model never sees
it. So it lives in its own top-level ``guardrails`` field — modelled like
``mcp_servers`` (see :mod:`gateway.models.mcp`) — rather than inside ``tools``.

The gateway extracts this field, runs the configured checks against the
operator-controlled guardrails service (which exposes ``POST /validate``),
and strips the field before forwarding the
request upstream. Omit the field entirely → no guardrail runs.

Also holds the organization guardrail tables.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field, SecretStr, model_validator
from sqlalchemy import JSON, CheckConstraint, ForeignKey, ForeignKeyConstraint, Text, UniqueConstraint, Uuid, true
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime

GuardrailDirection = Literal["input", "tool_results", "tool_calls", "output"]
"""What a guardrail reads: the caller's input, tool results, the tool calls the model makes, or its answer."""

GuardrailMode = Literal["block", "monitor"]
"""What a guardrail does when it fires: record it, or refuse or withhold the content."""

DEFAULT_DIRECTIONS: tuple[GuardrailDirection, ...] = ("input",)


def _default_directions() -> list[GuardrailDirection]:
    return list(DEFAULT_DIRECTIONS)


def distinct_directions(value: list[GuardrailDirection]) -> list[GuardrailDirection]:
    """``value`` with each direction once, in the order first named."""
    return list(dict.fromkeys(value))


class GuardrailConfig(BaseModel):
    """A single guardrail check the caller wants the gateway to enforce.

    URL safety: when ``url`` is supplied it is validated by
    :func:`gateway.services.guardrails.check_guardrail_urls` (not here at parse
    time — the check does a DNS lookup that must be awaited) with the same
    SSRF guard used for MCP server URLs (loopback allowed by default for
    same-host sidecars; gated by ``OTARI_MCP_ALLOW_LOOPBACK`` /
    ``OTARI_MCP_ALLOW_PRIVATE_HOSTS``). Most deployments omit ``url`` and rely
    on the operator-set ``OTARI_GUARDRAILS_URL`` instead.
    """

    profile: str = Field(min_length=1, max_length=128)
    """Profile name configured on the guardrails service (e.g. ``"alinia"``)."""

    url: str | None = Field(default=None, min_length=1)
    """Optional per-request override of the operator-set ``OTARI_GUARDRAILS_URL``."""

    on: Annotated[list[GuardrailDirection], AfterValidator(distinct_directions)] = Field(
        default_factory=_default_directions, min_length=1
    )
    """What the guardrail reads. ``input`` is the caller's input, ``tool_results``
    every tool result whoever ran the tool, ``tool_calls`` each call the model
    makes before it runs or reaches the client, and ``output`` the model's
    answer. Defaults to ``input`` alone."""

    mode: GuardrailMode = "monitor"
    """``monitor`` (default) → forward the request anyway and annotate the
    response with the verdict (shadow mode); good for observing without
    disrupting workflows on false positives. ``block`` → refuse flagged input or
    a flagged answer with a 403, and withhold a flagged tool result or tool call
    so the agent can recover."""

    on_unavailable: Literal["block", "monitor"] = "block"
    """What to do when the check cannot be evaluated: the guardrails service
    cannot be reached, or it answers with no verdict (``valid: null``), as
    opposed to reachable-and-flagging. Only meaningful with ``mode="block"``,
    since a ``monitor`` guardrail already fails open.

    ``block`` (default, and the pre-existing behavior) fails closed: an enforcing
    guardrail that could not run must not be silently skipped. The cost is that a
    guardrails outage rejects every request carrying this guardrail, ahead of any
    fallback chain, so an operator mandating one on a routing policy is choosing
    to make that service a hard dependency. ``monitor`` is the escape hatch: the
    request is served and the skipped check is recorded, trading enforcement for
    availability. A tool result that a failing-closed check cannot evaluate is
    withheld rather than refused, as a flagged one is."""

    validate_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra kwargs for the check itself. Forwarded to the guardrails service
    ``/validate`` call, where they merge on top of the profile's own
    ``validate_kwargs`` server-side, or handed to a guardrail this gateway holds
    when an organization's definition serves the profile."""

    # The caller's own. Mandates reuse this model and never set it: theirs travels
    # beside the config (``ResolvedOrganizationGuardrail``), and the merge drops a
    # caller's credential for any profile a mandate claims.
    credential: SecretStr | None = Field(default=None, min_length=1, exclude=True)
    """Sent to ``url`` as ``Authorization: Bearer``. Requires ``url``, which must
    then be ``https``, so it never reaches the deployment's guardrails service."""

    @model_validator(mode="after")
    def _credential_needs_url(self) -> GuardrailConfig:
        if self.credential is not None and self.url is None:
            msg = "guardrail 'credential' requires 'url': it is sent only to an endpoint the entry names"
            raise ValueError(msg)
        return self


def _one_entry_per_credentialed_profile(guardrails: list[GuardrailConfig]) -> list[GuardrailConfig]:
    """Refuse a profile named twice when either entry carries a credential.

    Credentials reach the runner keyed by profile, so with two entries of one
    profile a token would go to both entries' endpoints, including one that
    falls back to the deployment's service.
    """
    seen: set[str] = set()
    credentialed: set[str] = set()
    for guardrail in guardrails:
        if guardrail.profile in seen and (guardrail.credential is not None or guardrail.profile in credentialed):
            msg = f"guardrail profile {guardrail.profile!r} appears more than once alongside a 'credential'"
            raise ValueError(msg)
        seen.add(guardrail.profile)
        if guardrail.credential is not None:
            credentialed.add(guardrail.profile)
    return guardrails


RequestedGuardrails = Annotated[
    list[GuardrailConfig], Field(max_length=8), AfterValidator(_one_entry_per_credentialed_profile)
]
"""The ``guardrails`` field of a completion request body."""


class OrganizationGuardrailDefinition(Base):
    """An API-hosted guardrail an organization configured for Otari to build itself.

    The mandate below says *where* a check runs and how hard it bites. This row
    says *what* the check is: which of the guardrails
    ``services/guardrail_catalog`` lists to construct, and the arguments to
    construct it with. So an organization can define a check rather than only
    name a profile some other service already serves.

    A table of its own rather than three more columns on the mandate, because
    the two have different keys. A mandate is identified by
    ``(organization_id, profile)`` and also carries ``mode``,
    ``on_unavailable`` and the workspace scope; the build arguments depend on
    ``guardrail_name`` alone. Fused, one guardrail placed under two policies
    would hold two copies of the same vendor key, take two writes to rotate it,
    and build the same vendor client twice per worker, and "configured but not
    mandated yet" would have nowhere to live. Fusing the two later is
    denormalizing; splitting them later needs a migration that dedupes N
    possibly-divergent copies with no safe automatic answer, and that asymmetry
    is what settles it.

    Organization-keyed, not deployment-global, so each tenant manages its own (#818).

    The runner builds these rows and holds the vendor clients ready
    (``services/tenancy/organization_guardrail_runner``), and a mandate pointing
    at one runs it in this process rather than posting the check anywhere.
    """

    __tablename__ = "organization_guardrail_definitions"
    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_org_guardrail_definitions_org_name"),
        # Covers (organization_id, id) so the mandate below can carry a composite
        # foreign key to it, pinning each mandate to a definition of *its own*
        # organization rather than trusting every write path to check. Same
        # reason, same shape as ``uq_org_provider_keys_org_id``
        # (``models/provider_keys.py``).
        UniqueConstraint("organization_id", "id", name="uq_org_guardrail_definitions_org_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # What a mandate points at, and the organization's own label. Not the
    # ``profile`` a caller sends and not the vendor's class name: an
    # organization may configure Lakera Guard twice under two names, with
    # different thresholds.
    name: Mapped[str] = mapped_column(nullable=False)
    # The ``any_guardrail`` class to construct, e.g. ``lakera_guard``. A plain
    # string rather than an enum: the set of guardrails is upstream's to grow,
    # and the catalog is what a write path checks the value against.
    guardrail_name: Mapped[str] = mapped_column(nullable=False)
    create_kwargs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    # One encrypted ``{name: value}`` map, not a column per credential: the
    # guardrails do not agree on how many they need (Bedrock 3, watsonx 2, most
    # 1, two none), so a column each would chase every guardrail upstream adds.
    # Encrypted as a single string through ``services/secret_box``, the way
    # ``SearchToolCredential.encrypted_api_key`` is.
    encrypted_create_secrets: Mapped[str | None] = mapped_column(Text, default=None)
    # The organization's kill switch for the definition itself: one write stops
    # the guardrail everywhere it is mandated, without losing the arguments and
    # the secrets it took to set up.
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
    # ``onupdate`` is load-bearing beyond bookkeeping. A runner holding built
    # guardrails needs a fingerprint to diff against, and ``(id, updated_at)``
    # is it, so an unchanged definition is not rebuilt on every refresh.
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class OrganizationGuardrail(Base):
    """A guardrail an organization runs over the requests of its workspaces.

    The plane *above* the deployment-wide guardrail settings, not a replacement
    for them: ``guardrails_url`` stays in ``runtime_settings`` and a deployment
    that configures no organization guardrails behaves exactly as it did
    (otari#654). A row here is a check the organization mandates; it is merged
    into the effective guardrail list at admission by ``prepare_gateway_tools``
    the same way a routing policy's mandate already is, so an organization can
    only ever add a check or tighten one a caller asked for.

    That is what keeps this inside the rule ``src/gateway/AGENTS.md`` records
    from #655/#678: a mandated guardrail can only make *fewer* requests succeed,
    never more, whichever endpoint it names. Which is also why the entry may
    carry its own ``url`` and credential where a workspace code-execution policy
    may not: the sandbox is a capability a workspace would be acquiring, and a
    guardrail is a restriction the organization is accepting. A caller can
    already point a request-body guardrail at a URL of their own
    (``models/guardrails.GuardrailConfig.url``, SSRF-checked on the request
    path), so storing one here grants nothing that was not already reachable.

    ``profile`` is unique per organization rather than a nickname being unique,
    which is where this parts company with the hosted
    ``organization_guardrail_key`` (unique on ``(organization_id, nickname)``,
    so one profile may be configured twice). The effective guardrail set on this
    request path is keyed by profile, because ``merge_guardrail_layers`` has
    always merged that way; two rows of one profile could therefore never both
    run, and one would silently win.
    """

    __tablename__ = "organization_guardrails"
    __table_args__ = (
        UniqueConstraint("organization_id", "profile", name="uq_organization_guardrails_org_profile"),
        # Composite on purpose. A plain ``definition_id`` foreign key would let
        # one organization mandate another's definition, and its vendor key with
        # it; carrying ``organization_id`` into the reference means the database
        # refuses that whatever the write path checked.
        ForeignKeyConstraint(
            ["organization_id", "definition_id"],
            [
                "organization_guardrail_definitions.organization_id",
                "organization_guardrail_definitions.id",
            ],
            name="fk_organization_guardrails_definition",
            # RESTRICT, not CASCADE: dropping a definition a mandate still names
            # would silently stop a guardrail running, the one outcome worth
            # refusing outright. Deleting the *organization* still removes both,
            # because each table cascades from ``organization`` within the one
            # statement, and RESTRICT is checked at the end of it.
            ondelete="RESTRICT",
        ),
        # Not an XOR. Both NULL is today's ordinary row, the one falling back to
        # the deployment's ``guardrails_url``, so an XOR would refuse every
        # existing mandate. This only says the two backends are exclusive: a row
        # cannot both POST to a remote service and be built here.
        CheckConstraint(
            "NOT (url IS NOT NULL AND definition_id IS NOT NULL)",
            name="ck_organization_guardrails_single_backend",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization.id", ondelete="CASCADE"), nullable=False, index=True
    )
    profile: Mapped[str] = mapped_column(nullable=False)
    # NULL means "use the deployment's guardrails_url", which is the ordinary
    # case: an organization that runs its own any-guardrail deployment names it
    # here, and then the credential below is what authenticates to it.
    url: Mapped[str | None] = mapped_column(default=None)
    encrypted_credential: Mapped[str | None] = mapped_column(Text, default=None)
    # NULL means "this mandate names a profile on a service"; set means Otari
    # builds and runs the check itself from the definition. Nullable with no
    # backfill, because every row that predates this column is the former.
    definition_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, default=None)
    mode: Mapped[str] = mapped_column(default="monitor", nullable=False)
    on_unavailable: Mapped[str] = mapped_column(default="block", nullable=False)
    # What the mandate reads, as `GuardrailConfig.on` names it. The column is not
    # called `on`, which SQL reserves.
    on: Mapped[list[str]] = mapped_column(
        "directions", JSON, default=_default_directions, server_default='["input"]', nullable=False
    )
    validate_kwargs: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    # The organization's own kill switch. A disabled entry runs nowhere,
    # whatever its scope says, so an organization can stop a guardrail without
    # losing the credential and the workspace list it took to set up.
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    # The inheritance rule otari#654 asks for, and otari-ai's
    # ``is_org_default`` under a name that says what it does: true means every
    # workspace of the organization runs this, including one created tomorrow,
    # and the scope rows below are not consulted. False means it runs only in
    # the workspaces named there, and a new workspace inherits nothing.
    applies_to_all_workspaces: Mapped[bool] = mapped_column(default=False, nullable=False)
    # Whether it also checks what agents propose to agent memory in those workspaces, before the consolidator
    # keeps it. On by default: what is kept reaches other members' prompts through recall. Off is for a check
    # that misreads memory notes, such as a prompt-injection model that takes "do X before Y" for an injection.
    applies_to_memory: Mapped[bool] = mapped_column(default=True, server_default=true(), nullable=False)
    # Gotcha: a plain DateTime(timezone=True) reads back naive on SQLite. The dashboard
    # then shows it as local time.
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class OrganizationGuardrailWorkspace(Base):
    """One workspace an organization guardrail is scoped to.

    Membership only: a row means "this guardrail runs in this workspace", and
    its absence means it does not. otari-ai instead carries a
    ``disabled`` flag on the equivalent row and admits three states, two of
    which resolve to off; there is nothing here for a third state to record,
    because the scope is the organization's to set and a workspace has no veto
    over it (a veto would widen what succeeds, which #655/#678 does not allow).

    Ignored entirely when the guardrail's ``applies_to_all_workspaces`` is set,
    so rows left behind by flipping that on are inert rather than contradictory.

    Both sides cascade: the pairing has no meaning once either end is gone.
    """

    __tablename__ = "organization_guardrail_workspaces"

    organization_guardrail_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("organization_guardrails.id", ondelete="CASCADE"), primary_key=True
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))
