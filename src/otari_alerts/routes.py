"""The caller's organization's budget alert rules.

Thin composition over :mod:`otari_alerts.service`: resolve the caller's
identity, call the service, return its typed result. The role gate and the
destination checks live there, and the domain errors it raises carry their own
statuses (see :mod:`otari_alerts.errors`), so nothing here catches them.

Mounted by :func:`otari_alerts.register` as an ungated ``RouterContribution``,
which inherits Otari's ``/api/v1`` aggregate prefix, so these paths read
``/api/v1/organizations/me/alert-rules``. Ungated because the surface is
present exactly when the plugin is installed; there is no licensing decision
for an entitlement to answer. Authentication is a separate question and is
answered the way Otari's own management routers answer it, on the router.

Scoped to ``/me`` for the reason ``routes/organization_guardrails.py`` and
``routes/organization_pricing.py`` are: a standalone deployment has exactly one
organization and the caller's identity already points at it, so a request
cannot name one.

What these rules are evaluated against is the organization's own budget
ceilings, on the timer in :mod:`otari_alerts.evaluator`. Nothing on this router
sends an alert except ``POST /{rule_id}/test``, which is explicitly a
rehearsal.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_db, verify_master_key
from otari_alerts.service import (
    AlertRuleCreate,
    AlertRulePublic,
    AlertRulesPublic,
    AlertRuleTestResult,
    AlertRuleUpdate,
    OrganizationAlertService,
)


class Message(BaseModel):
    """A one-line acknowledgement, matching the shape Otari's own deletes return."""

    message: str


# Master key on the router, as every standalone management router declares it.
# A contributed router gets no credential by default, so it declares its own.
# The role gate is a separate question answered in the service: the credential
# says a request is the operator's, the membership says whether that identity
# may change where this organization's alerts are sent.
router = APIRouter(
    prefix="/organizations/me/alert-rules",
    tags=["organization-alerts"],
    dependencies=[Depends(verify_master_key)],
)


def get_organization_alert_service(db: Annotated[AsyncSession, Depends(get_db)]) -> OrganizationAlertService:
    """Build the service on the request's session."""
    return OrganizationAlertService(db)


OrganizationAlertServiceDep = Annotated[OrganizationAlertService, Depends(get_organization_alert_service)]


@router.get("")
async def list_alert_rules(
    service: OrganizationAlertServiceDep,
    current_identity: CurrentIdentity,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> AlertRulesPublic:
    """List where the caller's organization sends its budget alerts.

    Organization owners and admins only. Each rule's destination is returned
    redacted to its scheme and host: an Apprise URL carries its credentials in
    the path, so the stored value is never echoed back.
    """
    return await service.list_rules(user=current_identity, skip=skip, limit=limit)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_alert_rule(
    service: OrganizationAlertServiceDep,
    current_identity: CurrentIdentity,
    body: AlertRuleCreate,
) -> AlertRulePublic:
    """Send this organization's budget alerts somewhere. Organization owners and admins only.

    ``destination`` is an Apprise URL, so one field covers Slack, Discord,
    PagerDuty, Telegram, mail and a plain JSON webhook. It is validated for
    parseability here, and a webhook-shaped destination is additionally checked
    against the same SSRF rules the MCP and web-search write paths apply.

    ``warn_at_percent`` defaults to 80, which is the useful half of this
    feature: a refusal is already too late to act on. Set it to null to alert
    only when a ceiling starts refusing requests.
    """
    return await service.create_rule(user=current_identity, request=body)


@router.patch("/{rule_id}")
async def update_alert_rule(
    service: OrganizationAlertServiceDep,
    current_identity: CurrentIdentity,
    rule_id: uuid.UUID,
    body: AlertRuleUpdate,
) -> AlertRulePublic:
    """Change a rule's name, destination, thresholds, or enabled flag.

    Organization owners and admins only. Omitted fields are left as they are.
    ``warn_at_percent`` is the one field where an explicit null is a value
    rather than an omission: it means stop warning and alert only on refusal.
    """
    return await service.update_rule(user=current_identity, rule_id=rule_id, request=body)


@router.delete("/{rule_id}")
async def delete_alert_rule(
    service: OrganizationAlertServiceDep,
    current_identity: CurrentIdentity,
    rule_id: uuid.UUID,
) -> Message:
    """Discard a rule and its delivery history. Organization owners and admins only.

    Use ``enabled: false`` instead to stop the alerts while keeping the
    destination. Deleting drops the record of what has already been sent, so a
    ceiling that is still over its threshold alerts again once a rule is
    recreated.
    """
    await service.delete_rule(user=current_identity, rule_id=rule_id)
    return Message(message="Alert rule deleted")


@router.post("/{rule_id}/test")
async def test_alert_rule(
    service: OrganizationAlertServiceDep,
    current_identity: CurrentIdentity,
    rule_id: uuid.UUID,
) -> AlertRuleTestResult:
    """Send a sample alert to this rule's destination now. Organization owners and admins only.

    The counterpart of ``POST /api/v1/settings/mail/test``: a destination that
    silently accepts nothing looks exactly like a budget that never crossed a
    threshold, so proving the path works belongs at setup time rather than
    during the incident it was meant to warn about.

    Records nothing. A test is not an alert about a ceiling, and claiming a
    dedupe key here would suppress the real alert it is rehearsing for.
    """
    return await service.test_rule(user=current_identity, rule_id=rule_id)
