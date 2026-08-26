"""Outgoing mail settings, and the operator's way to prove they work.

Two endpoints under ``/v1/settings``, master-key gated and standalone-only like
the rest of the management API:

* ``GET /v1/settings/mail`` reports which transport (if any) this deployment
  would send through, what it would send as, and, when it cannot send, exactly
  which settings are missing. Naming them is the point: "mail is unavailable" is
  only honest if it says what would make it available.
* ``POST /v1/settings/mail/test`` renders and sends a real templated message, so
  an operator can confirm the configuration end to end rather than discovering
  it the first time someone is invited. On a deployment with no transport it
  refuses with 503 up front and never pretends to have sent anything.

The refusal is the reusable half. A surface with no non-mail fallback (the
password-reset flow to come) gates on the same ``mail_ready`` the dashboard
reads from ``/v1/bootstrap`` and refuses the same way, rather than accepting a
request it will silently drop.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from gateway.api.deps import get_config, require_deployment_operator
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.mail import Mailer, MailNotConfiguredError, normalized_address

router = APIRouter(prefix="/v1/settings/mail", tags=["settings"])


class MailSettings(BaseModel):
    """What this deployment can send, and what stands in the way if it cannot."""

    transport: str = Field(
        description="The transport a send would use: 'smtp', 'console' (logged, not delivered), or 'none'."
    )
    enabled: bool = Field(description="Whether a transport is configured at all.")
    ready: bool = Field(
        description=(
            "Whether a message carrying a link back to this deployment can be sent, which is "
            "what every message the control plane sends needs. Matches 'mail_ready' on "
            "/v1/bootstrap."
        )
    )
    from_email: str | None = Field(description="The 'From' address on outgoing mail, if one is configured.")
    from_name: str = Field(description="The 'From' display name on outgoing mail.")
    public_base_url: str | None = Field(
        description="This deployment's own externally-reachable URL, used to build links in outgoing mail."
    )
    missing: list[str] = Field(
        description=(
            "Settings that must be set before mail works, in config order. Empty exactly when "
            "'ready' is true, so the dashboard can name what to configure rather than only "
            "reporting that mail is off."
        )
    )


class SendTestMailRequest(BaseModel):
    """Where to send the test message."""

    # Not ``EmailStr``: that would pull in email-validator for one field, the
    # same trade tenancy's address fields already declined. The format hint
    # still reaches the generated client, so the form validates it, and the
    # handler applies the shape check the rest of the codebase uses.
    to: str = Field(
        max_length=255,
        description="Recipient of the test message.",
        json_schema_extra={"format": "email"},
    )


class SendTestMailResponse(BaseModel):
    """The outcome of one test send.

    ``reason`` carries the transport's own error text, which is the whole value
    of a test button (an operator needs to know it was a refused login rather
    than a wrong port). Safe here and only here: this endpoint is master-key
    gated, unlike the public error paths that must not leak internals.
    """

    ok: bool
    transport: str
    reason: str | None = None


def _settings(config: GatewayConfig) -> MailSettings:
    mailer = Mailer(config)
    return MailSettings(
        transport=mailer.transport_name,
        enabled=mailer.is_configured,
        ready=mailer.can_send_links,
        from_email=config.mail_from_email,
        from_name=config.mail_from_name,
        public_base_url=config.public_base_url,
        missing=list(mailer.missing_settings),
    )


@router.get("", dependencies=[Depends(require_deployment_operator)])
async def get_mail_settings(config: Annotated[GatewayConfig, Depends(get_config)]) -> MailSettings:
    """Report the effective outgoing-mail configuration."""
    return _settings(config)


@router.post("/test", dependencies=[Depends(require_deployment_operator)])
async def send_test_mail(
    request: SendTestMailRequest,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> SendTestMailResponse:
    """Send one templated test message to prove the configuration works.

    Refuses with 503 when the deployment cannot send a linked message, naming
    the missing settings; the dashboard disables the control in that state, so
    reaching this is a direct API call or a race with a configuration change.

    The recipient is the only caller-supplied value: the body is a fixed
    template, so this cannot be used to put chosen text in someone's inbox from
    the deployment's own address, and the message says outright that no account
    was created for whoever receives it.
    """
    recipient = normalized_address(request.to)
    if recipient is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Not a valid email address: {request.to!r}",
        )

    mailer = Mailer(config)
    try:
        mailer.require_ready()
    except MailNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from None

    delivery = await mailer.send_template(
        "mail_test",
        to=recipient,
        subject="Otari test message",
        values={
            # public_base_url is what makes the mailer ready, so it is set here.
            "PUBLIC_BASE_URL": config.public_base_url or "",
            "TRANSPORT": mailer.transport_name,
        },
    )
    logger.info("Operator sent a test email via %s: delivered=%s", delivery.transport, delivery.delivered)
    return SendTestMailResponse(ok=delivery.delivered, transport=delivery.transport, reason=delivery.reason)
