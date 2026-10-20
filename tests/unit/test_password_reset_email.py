"""The copy of the password-reset message.

The rendering mechanics (layout, escaping, one-pass fill, subject
sanitization) are covered in ``test_mail.py``; what is asserted here is this
message's own values and the wording it produces.
"""

import pytest

from gateway.services.tenancy.password_reset_email import render_account_exists_email, render_password_reset_email

RESET_LINK = "https://gw.example.com/#/reset-password?token=abc"


def _render(**overrides: object) -> tuple[str, str, str]:
    message = render_password_reset_email(
        **{  # type: ignore[arg-type]
            "reset_link": RESET_LINK,
            "expiry_hours": 2,
            "asset_root": None,
            **overrides,
        }
    )
    return message.subject, message.html, message.text


def test_render_password_reset_email_fills_every_placeholder() -> None:
    subject, html, text = _render()

    assert "reset" in subject.lower()
    for rendered in (html, text):
        assert RESET_LINK in rendered
        assert "2 hours" in rendered
        assert "once" in rendered
    assert "{{RESET_LINK}}" not in html
    assert "{{VALID_PERIOD}}" not in html


def test_render_password_reset_email_never_overstates_the_expiry() -> None:
    _, half_day_html, _ = _render(expiry_hours=12)
    assert "12 hours" in half_day_html
    assert "1 day" not in half_day_html

    _, exact_two_days_html, _ = _render(expiry_hours=48)
    assert "2 days" in exact_two_days_html


@pytest.mark.parametrize(
    ("has_password", "provider", "expected"),
    [
        (True, None, "Forgot your password?"),
        (False, "GitHub", "signs in with GitHub"),
        (False, None, "with a provider or a passkey"),
    ],
)
def test_render_account_exists_email_says_how_the_account_signs_in(
    has_password: bool, provider: str | None, expected: str
) -> None:
    message = render_account_exists_email(
        sign_in_link="https://gw.example.com/", has_password=has_password, provider=provider, asset_root=None
    )

    assert message.subject == "You already have an Otari account"
    assert expected in message.text
    assert "https://gw.example.com/" in message.html
    assert "token=" not in message.text
