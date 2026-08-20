"""Mail configuration and the stdlib SMTP sender.

No real SMTP server is exercised: ``mail_enabled``/``invitation_mail_ready``
are pure config properties, and ``send_mail`` is tested for the two things
that must never raise (unconfigured, and a send that fails), not for actually
delivering mail.
"""

from unittest.mock import MagicMock, patch

from gateway.core.config import GatewayConfig
from gateway.services.mail_service import send_mail
from gateway.services.tenancy.invitation_email import render_invitation_email


def test_mail_is_disabled_with_nothing_configured() -> None:
    config = GatewayConfig()
    assert config.mail_enabled is False
    assert config.invitation_mail_ready is False


def test_mail_enabled_needs_both_a_host_and_a_from_address() -> None:
    host_only = GatewayConfig(smtp_host="smtp.example.com")
    from_only = GatewayConfig(mail_from_email="otari@example.com")
    both = GatewayConfig(smtp_host="smtp.example.com", mail_from_email="otari@example.com")

    assert host_only.mail_enabled is False
    assert from_only.mail_enabled is False
    assert both.mail_enabled is True


def test_invitation_mail_ready_also_needs_a_public_base_url() -> None:
    configured = GatewayConfig(smtp_host="smtp.example.com", mail_from_email="otari@example.com")
    with_url = GatewayConfig(
        smtp_host="smtp.example.com",
        mail_from_email="otari@example.com",
        public_base_url="https://otari.example.com",
    )

    assert configured.invitation_mail_ready is False
    assert with_url.invitation_mail_ready is True


def test_send_mail_is_a_no_op_when_unconfigured() -> None:
    config = GatewayConfig()
    sent = send_mail(config, to="ada@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")
    assert sent is False


def test_send_mail_reports_failure_rather_than_raising() -> None:
    """An unreachable host must not turn a mail failure into a request failure."""
    config = GatewayConfig(
        smtp_host="127.0.0.1",
        smtp_port=1,  # nothing listens here
        mail_from_email="otari@example.com",
    )
    sent = send_mail(config, to="ada@example.com", subject="Hi", html="<p>Hi</p>", text="Hi")
    assert sent is False


def test_send_mail_never_raises_even_on_a_malformed_header() -> None:
    """Defense in depth: message serialization runs before any socket I/O.

    ``message.as_string()`` is evaluated as a plain argument expression before
    ``client.sendmail(...)`` is even called, so this raises the same
    ``HeaderParseError`` it would with a real server, with no network needed
    to prove ``send_mail`` still reports ``False`` instead of propagating it.
    """
    config = GatewayConfig(smtp_host="smtp.example.com", mail_from_email="otari@example.com")
    with patch("gateway.services.mail_service.smtplib.SMTP") as mock_smtp:
        mock_smtp.return_value.__enter__.return_value = MagicMock()
        sent = send_mail(
            config,
            to="ada@example.com",
            subject="Broken\r\nheader: injected",
            html="<p>Hi</p>",
            text="Hi",
        )
    assert sent is False


def test_render_invitation_email_fills_every_placeholder() -> None:
    subject, html, text = render_invitation_email(
        organization_name="Acme",
        inviter_name="Ada",
        role="admin",
        accept_link="https://gw.example.com/#/accept-invitation?token=abc",
        expiry_hours=168,
    )

    assert "Acme" in subject
    for rendered in (html, text):
        assert "Acme" in rendered
        assert "Ada" in rendered
        assert "admin" in rendered
        assert "https://gw.example.com/#/accept-invitation?token=abc" in rendered
        assert "7 days" in rendered
    assert "{{ORGANIZATION_NAME}}" not in html
    assert "{{ACCEPT_LINK}}" not in html


def test_render_invitation_email_escapes_operator_set_strings_in_html_only() -> None:
    _, html, text = render_invitation_email(
        organization_name="<script>alert(1)</script>",
        inviter_name="Robert</b>",
        role="member",
        accept_link="https://gw.example.com/#/accept-invitation?token=abc",
        expiry_hours=168,
    )

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    # The plain-text variant has no markup to break out of, so it is not escaped.
    assert "<script>alert(1)</script>" in text


def test_render_invitation_email_does_not_let_one_value_collide_with_another_placeholder() -> None:
    """A bare-word placeholder scheme would let ``ROLE`` corrupt an org named after it.

    Delimiting placeholders (``{{ROLE}}``) is what stops a later substitution
    pass from rewriting text an earlier one just inserted.
    """
    _, html, text = render_invitation_email(
        organization_name="Role Corp",
        inviter_name="Ada",
        role="member",
        accept_link="https://gw.example.com/#/accept-invitation?token=abc",
        expiry_hours=168,
    )

    assert "Role Corp" in html
    assert "Role Corp" in text


def test_render_invitation_email_never_overstates_the_expiry() -> None:
    """Neither rounding direction may claim more time than the link actually has."""
    _, half_day_html, half_day_text = render_invitation_email(
        organization_name="Acme",
        inviter_name="Ada",
        role="member",
        accept_link="https://gw.example.com/#/accept-invitation?token=abc",
        expiry_hours=12,
    )
    # Rounding down ("1 day") would overstate a 12-hour link.
    assert "12 hours" in half_day_html
    assert "12 hours" in half_day_text

    _, one_hour_html, _ = render_invitation_email(
        organization_name="Acme",
        inviter_name="Ada",
        role="member",
        accept_link="https://gw.example.com/#/accept-invitation?token=abc",
        expiry_hours=1,
    )
    assert "1 hour" in one_hour_html
    assert "1 hours" not in one_hour_html

    _, day_and_a_half_html, _ = render_invitation_email(
        organization_name="Acme",
        inviter_name="Ada",
        role="member",
        accept_link="https://gw.example.com/#/accept-invitation?token=abc",
        expiry_hours=36,
    )
    # Rounding up ("2 days") would overstate a 36-hour link just as much;
    # staying in hours for anything that isn't an exact day count is what
    # avoids overstating in either direction.
    assert "36 hours" in day_and_a_half_html
    assert "2 days" not in day_and_a_half_html

    _, exact_two_days_html, _ = render_invitation_email(
        organization_name="Acme",
        inviter_name="Ada",
        role="member",
        accept_link="https://gw.example.com/#/accept-invitation?token=abc",
        expiry_hours=48,
    )
    assert "2 days" in exact_two_days_html


def test_render_invitation_email_strips_newlines_from_the_subject_line() -> None:
    """A newline in the subject is either a malformed header or a header-injection vector."""
    subject, _, _ = render_invitation_email(
        organization_name="Acme\r\nBcc: attacker@example.com",
        inviter_name="Ada",
        role="member",
        accept_link="https://gw.example.com/#/accept-invitation?token=abc",
        expiry_hours=168,
    )

    assert "\r" not in subject
    assert "\n" not in subject
