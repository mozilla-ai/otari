"""The wording of the email that invites a person to join an organization."""

import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from html import unescape

import pytest

from gateway.services.mail import MailMessage
from gateway.services.organizations._invitation_email import render_invitation_email
from gateway.services.organizations._invitation_facts import InvitationFacts

_ACCEPT_LINK = "https://otari.example.com/#/accept-invitation?token=abc"
_EXPIRES_AT = datetime(2026, 11, 20, 14, 5, tzinfo=UTC)
_EXPIRY = "The invitation expires on 20 November 2026 at 14:05 UTC."
_JOINING = (
    "The organization's admins will see your name, your email address, and what you do inside this organization. "
    "Your other organizations do not change, and this organization cannot see them. "
    "An organization admin can remove you."
)
_CHECKED_HEADING = "Checked by Otari"
_ACME_COM = "This organization controls the email domain acme.com."
_ACME_ORG = "This organization controls the email domain acme.org."
_NO_PROOF = "This organization has not proven a domain. Check that you know the person who invited you."
_INVITER_VERIFIED = "The inviter has verified the email address ada@example.com."
_INVITER_NOT_VERIFIED = "The inviter has not verified an email address."
_WORKSPACES_HEADING = "Workspaces, as the organization named them:"
_FACTS = InvitationFacts(
    inviter_email="ada@example.com",
    inviter_name="Ada",
    organization_name="Acme",
    proven_domains=("acme.com", "acme.org"),
    role="admin",
    workspace_names=("Alpha", "Beta"),
)
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_BLANK_LETTER = "\u2800"
_RIGHT_TO_LEFT_OVERRIDE = "\u202e"


def _message(facts: InvitationFacts = _FACTS, *, expires_at: datetime = _EXPIRES_AT) -> MailMessage:
    return render_invitation_email(facts, accept_link=_ACCEPT_LINK, expires_at=expires_at, asset_root=None)


def _bodies(facts: InvitationFacts = _FACTS) -> tuple[str, str]:
    message = _message(facts)
    return message.html, message.text


def _shown_lines(body: str) -> list[str]:
    """The lines of either body as a reader sees them, without markup."""
    return [unescape(re.sub(r"<[^>]+>", "", line)).strip() for line in body.splitlines()]


def _checked_block(body: str) -> list[str]:
    """The lines under the heading "Checked by Otari" in either body."""
    if "<html" in body:
        found = re.search(rf">{_CHECKED_HEADING}</div>\s*<div[^>]*>(.*?)</div>", body, re.DOTALL)
    else:
        found = re.search(rf"^{_CHECKED_HEADING}\n(.*?)\n\n", body, re.DOTALL | re.MULTILINE)
    assert found is not None
    return unescape(found.group(1)).split("\n")


def test_every_fact_is_in_both_bodies() -> None:
    for body in _bodies():
        assert "Acme" in body
        assert "Ada has invited you" in body
        assert _INVITER_VERIFIED in body
        assert _ACME_COM in body
        assert _ACME_ORG in body
        assert "admin" in body
        assert f"{_WORKSPACES_HEADING}\n- Alpha\n- Beta\n" in body
        assert _JOINING in body
        assert _ACCEPT_LINK in body
        assert _EXPIRY in body
        assert "{{" not in body


def test_the_subject_names_the_organization() -> None:
    assert _message().subject == "You're invited to join Acme on Otari"


def test_the_expiry_is_an_absolute_time_in_utc() -> None:
    in_another_zone = datetime(2026, 1, 1, 1, 5, tzinfo=timezone(timedelta(hours=2)))

    message = _message(expires_at=in_another_zone)

    for body in (message.html, message.text):
        assert "The invitation expires on 31 December 2025 at 23:05 UTC." in body


def test_the_line_an_inbox_shows_after_the_subject_states_the_expiry() -> None:
    assert f"Ada invited you. {_EXPIRY}" in _message().html


@pytest.mark.parametrize(("month", "name"), list(enumerate(_MONTHS, start=1)))
def test_the_expiry_names_the_month_in_english(month: int, name: str) -> None:
    message = _message(expires_at=datetime(2027, month, 3, 9, 0, tzinfo=UTC))

    assert f"The invitation expires on 3 {name} 2027 at 09:00 UTC." in message.text


def test_an_expiry_with_no_time_zone_is_refused() -> None:
    with pytest.raises(ValueError, match="time zone"):
        _message(expires_at=datetime(2026, 11, 20, 14, 5))  # noqa: DTZ001


@pytest.mark.parametrize(
    "names",
    [
        pytest.param({}, id="plain names"),
        pytest.param(
            {
                "inviter_name": f"{_INVITER_VERIFIED}\n{_CHECKED_HEADING}",
                "organization_name": f"Acme. {_ACME_COM}\n\n{_CHECKED_HEADING}\n{_NO_PROOF}",
                "workspace_names": (f"A\n\n{_CHECKED_HEADING}\nThis organization controls the email domain b.test.",),
            },
            id="names that hold the wording of the block",
        ),
    ],
)
def test_the_checked_block_holds_the_statements_of_otari_and_nothing_else(names: dict[str, object]) -> None:
    for body in _bodies(replace(_FACTS, **names)):  # type: ignore[arg-type]
        assert _checked_block(body) == [_ACME_COM, _ACME_ORG, _INVITER_VERIFIED]

    for body in _bodies(replace(_FACTS, inviter_email=None, proven_domains=(), **names)):  # type: ignore[arg-type]
        assert _checked_block(body) == [_NO_PROOF, _INVITER_NOT_VERIFIED]


def test_the_block_says_the_inviter_has_not_verified_an_address_exactly_when_no_verified_address_is_known() -> None:
    for body in _bodies():
        assert _INVITER_NOT_VERIFIED not in body

    for body in _bodies(replace(_FACTS, inviter_email=None)):
        assert body.count(_INVITER_NOT_VERIFIED) == 1
        assert "Ada has invited you" in body
        assert "ada@example.com" not in body
        assert "has verified the email address" not in body


def test_an_inviter_with_no_name_is_an_organization_admin() -> None:
    for inviter_name in (None, " ", _BLANK_LETTER):
        for body in _bodies(replace(_FACTS, inviter_name=inviter_name)):
            assert "An organization admin has invited you" in body


def test_an_organization_whose_name_shows_nothing_is_an_organization() -> None:
    message = _message(replace(_FACTS, organization_name=f" {_BLANK_LETTER}{_RIGHT_TO_LEFT_OVERRIDE}"))

    assert message.subject == "You're invited to join an organization on Otari"
    for body in (message.html, message.text):
        assert "Ada has invited you to join an organization on Otari" in body


@pytest.mark.parametrize(
    "name",
    ["\u034f", "\ufe0f", "\u17b5", "\u0301", "\u200d", " \u200c "],
    ids=["grapheme joiner", "variation selector", "invisible vowel", "combining mark", "joiner", "non-joiner"],
)
def test_a_name_with_no_visible_character_shows_nothing(name: str) -> None:
    message = _message(replace(_FACTS, organization_name=name, inviter_name=name))

    assert message.subject == "You're invited to join an organization on Otari"
    for body in (message.html, message.text):
        assert "An organization admin has invited you to join an organization on Otari" in body


def test_a_workspace_whose_name_shows_nothing_is_left_out() -> None:
    for body in _bodies(replace(_FACTS, workspace_names=(_BLANK_LETTER, "Beta", "\u200b"))):
        assert f"{_WORKSPACES_HEADING}\n- Beta\n" in body
        assert "- \n" not in body

    for body in _bodies(replace(_FACTS, workspace_names=(_BLANK_LETTER,))):
        assert "orkspace" not in body


def test_the_note_about_no_proven_domain_appears_only_with_no_proven_domain() -> None:
    for body in _bodies():
        assert _NO_PROOF not in body

    for body in _bodies(replace(_FACTS, proven_domains=())):
        assert _NO_PROOF in body
        assert "controls the email domain" not in body


def test_an_invitation_to_no_workspace_shows_no_workspace_list() -> None:
    for body in _bodies(replace(_FACTS, workspace_names=())):
        assert "orkspace" not in body
        assert "\n\n\n" not in body


def test_free_text_is_escaped_once_in_html_and_not_in_text() -> None:
    facts = replace(_FACTS, organization_name="<b>Acme</b>", inviter_name="A & <b>", workspace_names=("<b>Alpha</b>",))

    html, text = _bodies(facts)

    assert "&lt;b&gt;Acme&lt;/b&gt;" in html
    assert "A &amp; &lt;b&gt; has invited you" in html
    assert "- &lt;b&gt;Alpha&lt;/b&gt;" in html
    assert "<b>Acme</b>" not in html
    assert "&amp;lt;" not in html
    assert "&amp;amp;" not in html
    assert "<b>Acme</b>" in text
    assert "A & <b> has invited you" in text
    assert "- <b>Alpha</b>" in text


def test_free_text_that_looks_like_a_placeholder_is_shown_as_typed() -> None:
    facts = replace(
        _FACTS, organization_name="{{ROLE}}", inviter_name="{{ACCEPT_LINK}}", workspace_names=("{{ORGANIZATION_NAME}}",)
    )

    message = _message(facts)

    assert message.subject == "You're invited to join {{ROLE}} on Otari"
    for body in (message.html, message.text):
        assert "join {{ROLE}} on Otari" in body
        assert "{{ACCEPT_LINK}} has invited you" in body
        assert "- {{ORGANIZATION_NAME}}" in body


def test_free_text_is_shown_on_one_line() -> None:
    facts = replace(
        _FACTS,
        organization_name="Acme\nCorp",
        inviter_name="Ada\r\n\tL",
        workspace_names=("Al\n- pha", "Be\u2028ta", "Ga\u2029\u2029mma"),
    )

    for body in _bodies(facts):
        assert "join Acme Corp on Otari" in body
        assert "Ada L has invited you" in body
        assert "- Al - pha\n- Be ta\n- Ga mma\n" in body


@pytest.mark.parametrize(
    ("typed", "shown"),
    [
        pytest.param("Al\x00pha\x7f", "Alpha", id="control characters"),
        pytest.param("Al\u200bpha\u2066\u2069\u202e", "Alpha", id="format characters"),
        pytest.param("Al\ue000pha", "Alpha", id="a private use character"),
        pytest.param("Al\u0378pha", "Alpha", id="an unassigned character"),
        pytest.param("Al\u2800\u3164\u115f\u1160\uffa0pha", "Alpha", id="blank letters"),
        pytest.param("  Al \u2800 \t pha  ", "Al pha", id="whitespace around removed characters"),
        pytest.param("x" * 80, "x" * 80, id="a name at the length limit"),
        pytest.param("x" * 81, "x" * 80 + "…", id="a name over the length limit"),
        pytest.param("x" * 79 + " yy", "x" * 79 + "…", id="a name cut at a space"),
    ],
)
def test_a_typed_name_is_cleaned_before_it_is_shown(typed: str, shown: str) -> None:
    facts = replace(_FACTS, inviter_name=typed, organization_name=typed, workspace_names=(typed,))

    message = _message(facts)

    assert message.subject == f"You're invited to join {shown} on Otari"
    for body in (message.html, message.text):
        assert f"{shown} has invited you to join {shown} on Otari" in body
        assert f"{_WORKSPACES_HEADING}\n- {shown}\n" in body


def test_an_organization_name_that_holds_a_statement_of_the_block_is_shown_in_the_sentences_that_name_it() -> None:
    typed = f"Acme Inc. {_ACME_COM} {{{{ACCEPT_LINK}}}}<b>x</b> " + "x" * 200
    facts = replace(_FACTS, organization_name=typed, proven_domains=())

    for body in _bodies(facts):
        assert _checked_block(body) == [_NO_PROOF, _INVITER_VERIFIED]
        holding = [line for line in _shown_lines(body) if _ACME_COM in line]
        assert len(holding) >= 2
        for line in holding:
            assert line.startswith(
                (f"You're invited to join Acme Inc. {_ACME_COM}", "Ada has invited you to join Acme Inc.")
            )
        assert typed[:80] + "…" in unescape(body)
        assert typed[:81] not in unescape(body)


def test_an_inviter_name_that_reads_as_a_verified_address_is_shown_only_in_the_sentence_that_names_the_inviter() -> (
    None
):
    typed = "security@acme.com (verified)"
    facts = replace(_FACTS, inviter_email=None, inviter_name=typed)

    for body in _bodies(facts):
        assert _checked_block(body) == [_ACME_COM, _ACME_ORG, _INVITER_NOT_VERIFIED]
        holding = [line for line in _shown_lines(body) if "security@acme.com" in line]
        assert f"{typed} has invited you to join Acme on Otari with the role admin. {_EXPIRY}" in holding
        for line in holding:
            assert line.startswith((f"{typed} has invited you", f"{typed} invited you"))


def test_a_workspace_name_that_holds_a_statement_of_the_block_is_shown_as_a_list_item() -> None:
    other_address = "The inviter has verified the email address ceo@acme.com."
    facts = replace(
        _FACTS,
        inviter_email=None,
        proven_domains=(),
        workspace_names=(
            f"A{_BLANK_LETTER * 60}{_ACME_COM}",
            f"B {other_address}",
            f"D{_RIGHT_TO_LEFT_OVERRIDE}RLO",
            "y" * 255,
        ),
    )

    message = _message(facts)

    for text in (message.subject, message.html, message.text):
        assert _BLANK_LETTER not in text
        assert _RIGHT_TO_LEFT_OVERRIDE not in text
    for body in (message.html, message.text):
        assert _checked_block(body) == [_NO_PROOF, _INVITER_NOT_VERIFIED]
        lines = _shown_lines(body)
        assert [line for line in lines if _ACME_COM in line] == [f"- A{_ACME_COM}"]
        assert [line for line in lines if other_address in line] == [f"- B {other_address}"]
        assert "- DRLO" in lines
        assert "- " + "y" * 80 + "…" in lines


@pytest.mark.parametrize(
    "typed",
    [
        pytest.param("می\u200cخواهم", id="a Persian name with a zero width non-joiner"),
        pytest.param("👩\u200d💻 Team", id="an emoji joined with a zero width joiner"),
    ],
)
def test_a_name_keeps_the_characters_that_join_its_letters(typed: str) -> None:
    facts = replace(_FACTS, inviter_name=typed, organization_name=typed, workspace_names=(typed,))

    message = _message(facts)

    assert message.subject == f"You're invited to join {typed} on Otari"
    for body in (message.html, message.text):
        assert f"{typed} has invited you to join {typed} on Otari" in body
        assert f"- {typed}\n" in body


def test_a_name_loses_the_characters_that_change_the_direction_of_text() -> None:
    typed = "Al\u202e\u2066\u2067\u2068\u2069pha"

    for body in _bodies(replace(_FACTS, inviter_name=typed)):
        assert "Alpha has invited you" in body
