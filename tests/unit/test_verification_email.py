"""The copy of the email-verification message.

The rendering mechanics (layout, escaping, one-pass fill, subject
sanitization) are covered in ``test_mail.py``; what is asserted here is this
message's own values and the wording it produces.
"""

from gateway.services.tenancy.verification_email import render_verification_email

VERIFY_LINK = "https://gw.example.com/#/verify-email?token=abc"


def _render(**overrides: object) -> tuple[str, str, str]:
    message = render_verification_email(
        **{  # type: ignore[arg-type]
            "verify_link": VERIFY_LINK,
            "expiry_hours": 48,
            **overrides,
        }
    )
    return message.subject, message.html, message.text


def test_render_verification_email_fills_every_placeholder() -> None:
    subject, html, text = _render()

    assert "verify" in subject.lower()
    for rendered in (html, text):
        assert VERIFY_LINK in rendered
        assert "2 days" in rendered
    assert "{{VERIFY_LINK}}" not in html
    assert "{{VALID_PERIOD}}" not in html


def test_render_verification_email_never_overstates_the_expiry() -> None:
    _, html, _ = _render(expiry_hours=12)
    assert "12 hours" in html
    assert "1 day" not in html

    _, exact_two_days_html, _ = _render(expiry_hours=48)
    assert "2 days" in exact_two_days_html
