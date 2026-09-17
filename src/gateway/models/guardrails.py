"""Request-body model for the gateway-managed ``guardrails`` field.

Guardrails are *not* a model-callable tool. Unlike ``otari_code_execution`` /
``otari_web_search`` (which the model decides to invoke inside the tool-use
loop), a guardrail is a request-level policy the **caller** opts into: it runs
on the request regardless of what the model decides, and the model never sees
it. So it lives in its own top-level ``guardrails`` field — modelled like
``mcp_servers`` (see :mod:`gateway.models.mcp`) — rather than inside ``tools``.

The gateway extracts this field, runs the configured checks against the
operator-controlled guardrails service (``otari-anyguardrails-container``,
which exposes ``POST /validate``), and strips the field before forwarding the
request upstream. Omit the field entirely → no guardrail runs.

Also holds the stored guardrail definitions and the organization guardrail tables.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import JSON, DateTime, ForeignKey, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime
from gateway.models.secret_fields import REDACTED_VALUE, redact_secret_like_values

GuardrailDirection = Literal["input", "output"]


def _default_directions() -> list[GuardrailDirection]:
    return ["input"]


class GuardrailConfig(BaseModel):
    """A single guardrail check the caller wants the gateway to enforce.

    URL safety: when ``url`` is supplied it is validated by
    :func:`gateway.services.guardrails.run_input_guardrails` (not here at parse
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

    on: list[GuardrailDirection] = Field(default_factory=_default_directions)
    """Which directions to check. v1 enforces ``input`` only; ``output`` is
    accepted but not yet enforced (the response-direction check is a planned
    follow-up that needs streaming handling)."""

    mode: Literal["block", "monitor"] = "monitor"
    """``monitor`` (default) → forward the request anyway and annotate the
    response with the verdict (shadow mode); good for observing without
    disrupting workflows on false positives. ``block`` → reject the request
    with a 403 and never call the provider when the guardrail flags it."""

    on_unavailable: Literal["block", "monitor"] = "block"
    """What to do when the guardrails service cannot be reached at all, as
    opposed to reachable-and-flagging. Only meaningful with ``mode="block"``,
    since a ``monitor`` guardrail already fails open.

    ``block`` (default, and the pre-existing behavior) fails closed: an enforcing
    guardrail that could not run must not be silently skipped. The cost is that a
    guardrails outage rejects every request carrying this guardrail, ahead of any
    fallback chain, so an operator mandating one on a routing policy is choosing
    to make that service a hard dependency. ``monitor`` is the escape hatch: the
    request is served and the skipped check is recorded, trading enforcement for
    availability."""

    validate_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra kwargs forwarded to the guardrails service ``/validate`` call,
    merged on top of the profile's own ``validate_kwargs`` server-side."""


class GuardrailCredential(Base):
    """A guardrail defined in Otari rather than in a sidecar's YAML.

    ``name`` is the ``profile`` a caller sends, and ``guardrail_name`` is the
    ``any_guardrail`` class the catalog offered. The arguments that build it are
    split across two columns rather than typed as their own, because the
    guardrails a hosted API reaches do not share a secret shape: Bedrock carries
    three secret constructor arguments, watsonx two, most one, and two carry
    none. A column per credential would have to chase every guardrail upstream
    adds. So every secret goes into one ``{name: value}`` map encrypted as a
    single string, and the split is made by the catalog's own ``secret`` flag
    (``services/guardrail_credential_service.split_create_kwargs``), which needs
    no per-guardrail knowledge.

    Nothing on the request path reads this yet. Standalone and hosted only,
    never the hybrid platform path.
    """

    __tablename__ = "guardrail_credentials"

    name: Mapped[str] = mapped_column(primary_key=True)
    guardrail_name: Mapped[str] = mapped_column()
    create_kwargs: Mapped[dict[str, Any]] = mapped_column("create_kwargs", JSON, default=dict)
    encrypted_create_secrets: Mapped[str | None] = mapped_column(Text, default=None)
    validate_kwargs: Mapped[dict[str, Any]] = mapped_column("validate_kwargs", JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    # What to do when the guardrail flags the input: "block" refuses the request,
    # "monitor" serves it and reports the verdict. Spelled as
    # ``GuardrailConfig.mode`` is, because it becomes one.
    mode: Mapped[str] = mapped_column(default="block", nullable=False)
    # What Otari does when no verdict came back at all: "block" or "allow", not
    # the "monitor" ``GuardrailConfig.on_unavailable`` spells. The translation,
    # and the reason, are in ``guardrail_credential_service.stored_guardrail_config``.
    on_unavailable: Mapped[str] = mapped_column(default="block", nullable=False)
    # True means every workspace runs this, including one created tomorrow, and
    # the scope rows below are not consulted. False means only the workspaces
    # named there, and a new workspace inherits nothing. The rule, and the
    # wording, are ``OrganizationGuardrail``'s.
    applies_to_all_workspaces: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def to_public_dict(self, *, secret_names: Collection[str] = ()) -> dict[str, Any]:
        """Serialize for the API. Never includes a secret, only the names of the stored ones.

        ``secret_names`` comes from the caller, because the service is the only
        layer that may decrypt. Passing none is what a row whose map will not
        decrypt reports, so an unreadable credential costs the operator the names
        and not the listing.

        ``validate_kwargs`` is masked by key name the way
        ``organization_guardrails`` masks its own: a guardrail class can take its
        vendor key as a per-call argument, so the column held in clear is as much
        a credential as the encrypted one. That mask reaches top-level keys only,
        a gap this inherits along with the pattern and #1125 tracks.
        """
        return {
            "name": self.name,
            "guardrail_name": self.guardrail_name,
            "create_kwargs": dict(self.create_kwargs or {}),
            "create_secrets": {name: REDACTED_VALUE for name in sorted(secret_names)},
            "validate_kwargs": redact_secret_like_values(self.validate_kwargs) or {},
            "enabled": self.enabled,
            "mode": self.mode,
            "on_unavailable": self.on_unavailable,
            "applies_to_all_workspaces": self.applies_to_all_workspaces,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class GuardrailCredentialWorkspace(Base):
    """One workspace a stored guardrail definition runs in.

    Membership only, the way ``OrganizationGuardrailWorkspace`` is: a row means
    "this definition checks this workspace's requests", and its absence means it
    does not. Ignored entirely when the definition's
    ``applies_to_all_workspaces`` is set, so rows left behind by flipping that on
    are inert rather than contradictory.

    Scoped by workspace and not by organization, although the definition is
    deployment-wide and the credential in it belongs to the operator. An
    organization can already mandate a check over its own workspaces through
    ``organization_guardrails``, with an endpoint and a credential of its own;
    what an operator needs here is the other thing, one vendor account they hold
    pointed at whichever workspaces they choose, which an organization-keyed
    scope could not express.

    Both sides cascade: the pairing has no meaning once either end is gone.
    """

    __tablename__ = "guardrail_credential_workspaces"

    credential_name: Mapped[str] = mapped_column(
        ForeignKey("guardrail_credentials.name", ondelete="CASCADE"), primary_key=True
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=lambda: datetime.now(UTC))


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
    __table_args__ = (UniqueConstraint("organization_id", "profile", name="uq_organization_guardrails_org_profile"),)

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
    mode: Mapped[str] = mapped_column(default="monitor", nullable=False)
    on_unavailable: Mapped[str] = mapped_column(default="block", nullable=False)
    validate_kwargs: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    # The organization's own kill switch. A disabled entry runs nowhere,
    # whatever its scope says, so an organization can stop a guardrail without
    # losing the credential and the workspace list it took to set up.
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    # The inheritance rule otari#654 asks for, and the hosted plane's
    # ``is_org_default`` under a name that says what it does: true means every
    # workspace of the organization runs this, including one created tomorrow,
    # and the scope rows below are not consulted. False means it runs only in
    # the workspaces named there, and a new workspace inherits nothing.
    applies_to_all_workspaces: Mapped[bool] = mapped_column(default=False, nullable=False)
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
    its absence means it does not. The hosted plane instead carries a
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
