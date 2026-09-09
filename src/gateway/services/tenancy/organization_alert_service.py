"""Organization-scoped alert rules: CRUD, and the on-demand test send.

Where a tenant says how it wants to be told that one of its budgets is running
out. The rules themselves are acted on by
:mod:`gateway.services.alerts.evaluator`, which runs on a timer and never
touches this module; the two meet only at the ``alert_rules`` table.

**Why this lives with the tenancy services.** It shares their role gate
(``require_active_organization_management_access``) and their error family, so a
route over it stays thin and raises nothing, exactly as
`organization_guardrail_service` next door does.

**What a rule may watch.** The ceilings whose ``budgets`` row carries this
organization's id. A budget with a NULL ``organization_id`` is the deployment's
own, and `entities.Budget` records that the organization-scoped surface never
lists, offers or repoints one, so a tenant's rule cannot reach it and is not
meant to. Alerting on those is the deployment operator's plane and is
deliberately not in this surface; nothing in the schema has to change to add it,
because the dedupe key is per rule.

**The destination is a credential.** ``slack://`` embeds a bot token,
``discord://`` a webhook token, ``json://`` optional basic-auth. It is encrypted
with ``OTARI_SECRET_KEY`` before it is stored and never returned; reads get
``redact_alert_destination`` output, which masks path segments as well as
userinfo because Apprise puts its tokens in the path.

**SSRF.** Only the webhook-shaped schemas carry an operator-chosen host, so only
those are address-checked (:func:`_validate_destination`). The check runs at
write time, which is a real but partial guarantee: Apprise re-resolves DNS when
it sends, so a name that was safe when stored can move afterwards. That is the
same TOCTOU the MCP and web-search write paths carry, and it is why the gate is
fail-closed by default with a startup-only override
(``OTARI_ALERT_ALLOW_PRIVATE_HOSTS``) rather than a dashboard toggle.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Final
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import AlertRule
from gateway.models.tenancy import User
from gateway.services.alerts.dispatcher import (
    UnsupportedAlertDestinationError,
    parse_destination,
    send_alert,
)
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.tenancy.errors import (
    AlertRuleAlreadyExistsError,
    AlertRuleInertError,
    AlertRuleLimitReachedError,
    AlertRuleNotFoundError,
    AlertRuleUnsafeDestinationError,
    AlertRuleUnsupportedDestinationError,
    SecretBoxUnavailableTenancyError,
)
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.url_safety import (
    ALERT_SCHEME_TRANSPORT,
    UnsafeURLError,
    redact_alert_destination,
    validate_alert_destination_url,
)

# What one organization may configure. Every rule that matches a crossing is one
# more outbound send the evaluator's tick waits on, so this is a fan-out bound
# rather than a storage one, chosen near `MAX_GUARDRAILS_PER_ORGANIZATION` for
# the same reason: an operator wanting to notify more places than this wants a
# fan-out service on the other end of one webhook, not ten rules here.
MAX_ALERT_RULES_PER_ORGANIZATION: Final = 10

_MAX_LIST_LIMIT: Final = 1000


class AlertRuleCreate(BaseModel):
    """Request body for a new alert rule.

    ``destination`` is an Apprise URL. It is validated for parseability at this
    boundary rather than at send time, so an operator learns their URL is
    unusable while the form is still open instead of discovering it from an
    alert that never arrived.
    """

    # The generated sample would otherwise be ``"destination": "string"``, which
    # Apprise refuses, and generated samples reach a reader through the OpenAPI
    # spec and the Postman collection (`CreateSearchToolRequest` carries one for
    # the same reason).
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Platform team Slack",
                "destination": "slack://xoxb-token/C0123456789",
                "warn_at_percent": 80,
                "notify_on_exceeded": True,
            }
        }
    )

    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)] = Field(
        description="How this rule is identified in the dashboard and in the alert body, unique per organization",
    )
    destination: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = Field(
        max_length=4096,
        description=(
            "Apprise destination URL, for example slack://token/channel, "
            "discord://webhook_id/webhook_token, pagerduty://key@apikey, or json://host/path for a "
            "plain webhook. Encrypted at rest and never returned"
        ),
    )
    warn_at_percent: int | None = Field(
        default=80,
        gt=0,
        lt=100,
        description=(
            "Send a warning once a ceiling reaches this percent of any of its limits. "
            "Null sends no warning, leaving only the exceeded alert"
        ),
    )
    notify_on_exceeded: bool = Field(
        default=True,
        description="Send an alert when a ceiling reaches a limit and starts refusing requests",
    )
    enabled: bool = Field(default=True, description="False stops this rule sending without discarding it")

    @model_validator(mode="after")
    def _reject_a_rule_that_can_never_fire(self) -> AlertRuleCreate:
        """Refuse a rule with no warning threshold and no exceeded alert.

        Storable and meaningless: it would sit in the list looking like
        configured alerting while being incapable of producing a message. The
        way to turn a rule off is ``enabled``, which keeps saying what it means.
        """
        if self.warn_at_percent is None and not self.notify_on_exceeded:
            raise ValueError(
                "A rule must do something: set warn_at_percent, or notify_on_exceeded, or both. "
                "Use enabled=false to stop a rule without discarding it"
            )
        return self


class AlertRuleUpdate(BaseModel):
    """Partial update. Only the fields the caller sets are applied.

    ``destination`` has two states rather than three, unlike the credential
    fields on `organization_guardrail_service`: the column is NOT NULL, so there
    is nothing to clear it to. Omit it to keep the stored destination, send a
    value to replace it.

    ``warn_at_percent`` is the opposite: an explicit ``null`` is meaningful
    there and means "stop warning, alert only on refusal", so it is the one
    field here where null is accepted as a value.
    """

    model_config = ConfigDict(json_schema_extra={"example": {"warn_at_percent": 90, "enabled": False}})

    # ``SkipJsonSchema[None]`` on the fields backing NOT NULL columns, matching
    # `OrganizationGuardrailUpdate`: ``None`` is this schema's "not sent" marker
    # and the validator below refuses it as a value, so the published schema
    # must not advertise ``null`` as accepted or the generated TypeScript client
    # will tell a caller that ``{"name": null}`` is valid.
    name: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)] | SkipJsonSchema[None]
    ) = None
    destination: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] | SkipJsonSchema[None] = Field(
        default=None, max_length=4096
    )
    warn_at_percent: int | None = Field(default=None, gt=0, lt=100)
    notify_on_exceeded: bool | SkipJsonSchema[None] = None
    enabled: bool | SkipJsonSchema[None] = None

    @model_validator(mode="after")
    def _reject_explicit_nulls(self) -> AlertRuleUpdate:
        """Refuse an explicit ``null`` for a column that cannot hold one.

        Caught here rather than at the flush for the reason
        `OrganizationGuardrailUpdate` catches it: a NOT NULL violation arrives
        as the same ``IntegrityError`` the unique index raises, and the service
        would report a name collision that never happened.

        ``warn_at_percent`` is deliberately absent from the list: null is a
        value there, not an omission.
        """
        nulled = [
            field
            for field in ("name", "destination", "notify_on_exceeded", "enabled")
            if field in self.model_fields_set and getattr(self, field) is None
        ]
        if nulled:
            raise ValueError(f"{', '.join(nulled)} cannot be null; omit the field to leave it unchanged")
        return self


class AlertRulePublic(BaseModel):
    """The API-facing shape. Carries the redacted destination, never the real one."""

    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    # Scheme and host only; see `url_safety.redact_alert_destination`.
    destination: str
    warn_at_percent: int | None
    notify_on_exceeded: bool
    enabled: bool
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, rule: AlertRule) -> AlertRulePublic:
        return cls(
            id=rule.id,
            organization_id=rule.organization_id,
            name=rule.name,
            destination=rule.redacted_destination,
            warn_at_percent=rule.warn_at_percent,
            notify_on_exceeded=rule.notify_on_exceeded,
            enabled=rule.enabled,
            created_at=rule.created_at.isoformat(),
            updated_at=rule.updated_at.isoformat(),
        )


class AlertRulesPublic(BaseModel):
    data: list[AlertRulePublic]
    count: int


class AlertRuleTestResult(BaseModel):
    """The outcome of one on-demand send.

    ``detail`` carries the dispatcher's reason on failure, which names what went
    wrong without naming the destination, since the caller already knows which
    rule they asked about.
    """

    delivered: bool
    detail: str | None = None


def _transport_url(destination: str) -> str | None:
    """The http(s) URL a webhook-shaped destination will actually be posted to.

    Returns None for a vendor schema, whose endpoint is compiled into its
    Apprise plugin and is therefore not an operator-chosen address to check.
    """
    parts = urlsplit(destination)
    transport = ALERT_SCHEME_TRANSPORT.get(parts.scheme.lower())
    if transport is None:
        return None
    return urlunsplit((transport, parts.netloc, parts.path, parts.query, ""))


async def _validate_destination(destination: str) -> None:
    """Refuse a destination Apprise cannot use, or one pointed inside the deployment.

    Two separate checks with two separate errors, because they mean different
    things to whoever is filling in the form: the first is "this is not a URL I
    can deliver to", the second is "this is a URL I will not deliver to".
    """
    try:
        parse_destination(destination)
    except UnsupportedAlertDestinationError as exc:
        raise AlertRuleUnsupportedDestinationError(str(exc)) from exc

    transport = _transport_url(destination)
    if transport is None:
        return
    try:
        await validate_alert_destination_url(transport)
    except UnsafeURLError as exc:
        raise AlertRuleUnsafeDestinationError(str(exc)) from exc


def _encrypted(destination: str) -> str:
    try:
        return encrypt_secret(destination)
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError("alert rule destinations") from None


class OrganizationAlertService:
    """CRUD for the caller's organization's alert rules. Management-gated throughout."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.organizations = OrganizationService(db)

    async def _manageable_organization_id(self, user: User) -> uuid.UUID:
        """The caller's organization, having checked they may manage its alerts.

        One gate for reads and writes alike, matching
        `organization_guardrail_service` rather than the pricing overrides that
        any member may read: a rule names an external endpoint this gateway
        posts to, which is the same reason the MCP server list is gated. Where
        an organization's money goes is not a secret from the people spending
        it; where its alerts go is not something every member needs.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(
            user=user,
            organization=organization,
        )
        return organization.id

    async def _get_or_404(self, organization_id: uuid.UUID, rule_id: uuid.UUID) -> AlertRule:
        rule = await self.db.get(AlertRule, rule_id)
        if rule is None or rule.organization_id != organization_id:
            raise AlertRuleNotFoundError(rule_id)
        return rule

    async def list_rules(self, *, user: User, skip: int = 0, limit: int = 100) -> AlertRulesPublic:
        """One page of the organization's alert rules, and the total."""
        organization_id = await self._manageable_organization_id(user)
        limit = min(limit, _MAX_LIST_LIMIT)
        total = (
            await self.db.execute(
                select(func.count()).select_from(AlertRule).where(AlertRule.organization_id == organization_id)
            )
        ).scalar_one()
        rows = list(
            (
                await self.db.execute(
                    select(AlertRule)
                    .where(AlertRule.organization_id == organization_id)
                    .order_by(AlertRule.name)
                    .offset(skip)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return AlertRulesPublic(data=[AlertRulePublic.from_model(row) for row in rows], count=total)

    async def create_rule(self, *, user: User, request: AlertRuleCreate) -> AlertRulePublic:
        """Add a rule, encrypting its destination before it is stored."""
        organization_id = await self._manageable_organization_id(user)

        existing = (
            await self.db.execute(
                select(func.count()).select_from(AlertRule).where(AlertRule.organization_id == organization_id)
            )
        ).scalar_one()
        if existing >= MAX_ALERT_RULES_PER_ORGANIZATION:
            raise AlertRuleLimitReachedError(MAX_ALERT_RULES_PER_ORGANIZATION)

        await _validate_destination(request.destination)

        rule = AlertRule(
            organization_id=organization_id,
            name=request.name,
            encrypted_destination=_encrypted(request.destination),
            redacted_destination=redact_alert_destination(request.destination),
            warn_at_percent=request.warn_at_percent,
            notify_on_exceeded=request.notify_on_exceeded,
            enabled=request.enabled,
        )
        self.db.add(rule)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            await self.db.rollback()
            raise AlertRuleAlreadyExistsError(request.name) from exc
        await self.db.commit()
        await self.db.refresh(rule)
        return AlertRulePublic.from_model(rule)

    async def update_rule(self, *, user: User, rule_id: uuid.UUID, request: AlertRuleUpdate) -> AlertRulePublic:
        """Change a rule's name, destination, thresholds, or enabled flag."""
        organization_id = await self._manageable_organization_id(user)
        rule = await self._get_or_404(organization_id, rule_id)

        fields = request.model_fields_set
        if request.destination is not None:
            await _validate_destination(request.destination)
            rule.encrypted_destination = _encrypted(request.destination)
            rule.redacted_destination = redact_alert_destination(request.destination)
        if request.name is not None:
            rule.name = request.name
        # Read from ``model_fields_set`` rather than from the value, because
        # null is a meaningful value here and "not sent" is not.
        if "warn_at_percent" in fields:
            rule.warn_at_percent = request.warn_at_percent
        if request.notify_on_exceeded is not None:
            rule.notify_on_exceeded = request.notify_on_exceeded
        if request.enabled is not None:
            rule.enabled = request.enabled

        if rule.warn_at_percent is None and not rule.notify_on_exceeded:
            # The create body's own rule, re-checked against the merged row: a
            # PATCH can reach the same dead state by sending only one half.
            raise AlertRuleInertError()

        try:
            await self.db.flush()
        except IntegrityError as exc:
            await self.db.rollback()
            raise AlertRuleAlreadyExistsError(request.name) from exc
        await self.db.commit()
        await self.db.refresh(rule)
        return AlertRulePublic.from_model(rule)

    async def delete_rule(self, *, user: User, rule_id: uuid.UUID) -> None:
        """Discard a rule and its delivery history.

        The ``alert_deliveries`` rows cascade, which is what makes a deleted and
        recreated rule alert again on a ceiling that is still over its threshold
        rather than staying silent for the rest of the period.
        """
        organization_id = await self._manageable_organization_id(user)
        rule = await self._get_or_404(organization_id, rule_id)
        await self.db.delete(rule)
        await self.db.commit()

    async def test_rule(self, *, user: User, rule_id: uuid.UUID) -> AlertRuleTestResult:
        """Send a sample alert to this rule's destination now, and report the outcome.

        The counterpart of ``POST /v1/settings/mail/test``, and here for the same
        reason: a destination that silently accepts nothing is
        indistinguishable from a budget that never crossed a threshold, so an
        operator needs a way to prove the delivery path works while they are
        still setting it up rather than during the incident it was meant to
        warn about.

        Writes no ``alert_deliveries`` row. A test is not an alert about a
        ceiling and has no period to dedupe within, and claiming a key here
        would suppress the real alert it is rehearsing for.
        """
        organization_id = await self._manageable_organization_id(user)
        rule = await self._get_or_404(organization_id, rule_id)
        try:
            destination = decrypt_secret(rule.encrypted_destination)
        except SecretDecryptionError:
            # Reported as a failed test rather than raised: the caller asked
            # whether this rule can deliver, and "the stored destination will
            # not decrypt under the current OTARI_SECRET_KEY" is an answer to
            # that question, not a server fault.
            return AlertRuleTestResult(
                delivered=False,
                detail="The stored destination could not be decrypted; re-save the rule's destination",
            )
        result = await send_alert(
            destination,
            title=f"Otari test alert: {rule.name}",
            body=(
                f"This is a test of the alert rule '{rule.name}'.\n\n"
                "If you are reading it, budget alerts for this organization will reach here."
            ),
        )
        return AlertRuleTestResult(delivered=result.delivered, detail=result.detail)
