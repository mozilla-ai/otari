"""The copy of the organization invitation email.

The rendering mechanics (layout, escaping, one-pass fill, subject
sanitization) are covered in ``test_mail.py``; what is asserted here is this
message's own values and the wording it produces.
"""

from gateway.services.tenancy.invitation_email import render_invitation_email

ACCEPT_LINK = "https://gw.example.com/#/accept-invitation?token=abc"


def _render(**overrides: object) -> tuple[str, str, str]:
    message = render_invitation_email(
        **{  # type: ignore[arg-type]
            "organization_name": "Acme",
            "inviter_name": "Ada",
            "role": "member",
            "accept_link": ACCEPT_LINK,
            "expiry_hours": 168,
            **overrides,
        }
    )
    return message.subject, message.html, message.text


def test_render_invitation_email_fills_every_placeholder() -> None:
    subject, html, text = _render(role="admin")

    assert "Acme" in subject
    for rendered in (html, text):
        assert "Acme" in rendered
        assert "Ada" in rendered
        assert "admin" in rendered
        assert ACCEPT_LINK in rendered
        assert "7 days" in rendered
    assert "{{ORGANIZATION_NAME}}" not in html
    assert "{{ACCEPT_LINK}}" not in html


def test_render_invitation_email_escapes_operator_set_strings_in_html_only() -> None:
    _, html, text = _render(organization_name="<script>alert(1)</script>", inviter_name="Robert</b>")

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    # The plain-text variant has no markup to break out of, so it is not escaped.
    assert "<script>alert(1)</script>" in text


def test_render_invitation_email_does_not_let_one_value_collide_with_another_placeholder() -> None:
    """A bare-word placeholder scheme would let ``ROLE`` corrupt an org named after it."""
    _, html, text = _render(organization_name="Role Corp")

    assert "Role Corp" in html
    assert "Role Corp" in text


def test_render_invitation_email_never_overstates_the_expiry() -> None:
    """Neither rounding direction may claim more time than the link actually has."""
    _, half_day_html, half_day_text = _render(expiry_hours=12)
    # Rounding down ("1 day") would overstate a 12-hour link.
    assert "12 hours" in half_day_html
    assert "12 hours" in half_day_text

    _, one_hour_html, _ = _render(expiry_hours=1)
    assert "1 hour" in one_hour_html
    assert "1 hours" not in one_hour_html

    # Rounding up ("2 days") would overstate a 36-hour link just as much;
    # staying in hours for anything that isn't an exact day count is what
    # avoids overstating in either direction.
    _, day_and_a_half_html, _ = _render(expiry_hours=36)
    assert "36 hours" in day_and_a_half_html
    assert "2 days" not in day_and_a_half_html

    _, exact_two_days_html, _ = _render(expiry_hours=48)
    assert "2 days" in exact_two_days_html


def test_render_invitation_email_strips_newlines_from_the_subject_line() -> None:
    """A newline in the subject is either a malformed header or a header-injection vector."""
    subject, _, _ = _render(organization_name="Acme\r\nBcc: attacker@example.com")

    assert "\r" not in subject
    assert "\n" not in subject
