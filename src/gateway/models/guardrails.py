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

Also holds the organization guardrail tables.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import JSON, ForeignKey, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime

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
