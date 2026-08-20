"""Mail configuration, transport selection, the mailer, and the template renderer.

No real SMTP server is exercised. The transports are covered for the two things
that must never happen (a send that raises into the request that triggered it,
and a deployment discovering at send time that it has no mail), and the console
transport is what proves a *templated message actually sends* without one.
"""

import ast
import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.core.config import MAIL_TRANSPORT_SETTINGS, GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.services import mail as mail_package
from gateway.services.mail import (
    ConsoleTransport,
    MailEnvelope,
    Mailer,
    MailMessage,
    MailNotConfiguredError,
    MailTemplateError,
    SmtpTransport,
    normalized_address,
    render_email,
    select_transport,
)
from gateway.services.mail.templates import _load
from gateway.services.mail.transports import build_mime

MESSAGE = MailMessage(subject="Hi", html="<p>Hi</p>", text="Hi")


@pytest.fixture(autouse=True)
def _mail_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every test here to its own inputs.

    ``GatewayConfig()`` layers ``OTARI_*`` env vars (and a ``.env``) over its
    defaults, so a developer with ``OTARI_SMTP_HOST`` exported would flip the
    assertions about an unconfigured deployment, which are most of this file.
    """
    for name in (
        "OTARI_MAIL_TRANSPORT",
        "OTARI_SMTP_HOST",
        "OTARI_SMTP_PORT",
        "OTARI_SMTP_USER",
        "OTARI_SMTP_PASSWORD",
        "OTARI_MAIL_FROM_EMAIL",
        "OTARI_MAIL_FROM_NAME",
        "OTARI_PUBLIC_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def gateway_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Route the ``gateway`` logger (which does not propagate) into caplog."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)

SMTP_CONFIGURED = {"smtp_host": "smtp.example.com", "mail_from_email": "otari@example.com"}


def _ready(**overrides: object) -> GatewayConfig:
    """A config that can send a message carrying a link back to itself."""
    return GatewayConfig(**{**SMTP_CONFIGURED, "public_base_url": "https://otari.example.com", **overrides})  # type: ignore[arg-type]


# --- Configuration: which transport, and is it ready ------------------------


def test_mail_is_off_with_nothing_configured() -> None:
    """The state of every deployment that never heard of mail, and it is not an error."""
    config = GatewayConfig()
    assert config.effective_mail_transport == "none"
    assert config.mail_enabled is False
    assert config.mail_ready is False


def test_auto_selects_smtp_only_with_both_a_host_and_a_from_address() -> None:
    """A host with no from-address sends mail no recipient could trust; the reverse has nothing to send through."""
    assert GatewayConfig(smtp_host="smtp.example.com").effective_mail_transport == "none"
    assert GatewayConfig(mail_from_email="otari@example.com").effective_mail_transport == "none"
    assert GatewayConfig(**SMTP_CONFIGURED).effective_mail_transport == "smtp"  # type: ignore[arg-type]


def test_mail_ready_also_needs_a_public_base_url() -> None:
    """Every message the control plane sends carries a link back into this deployment."""
    configured = GatewayConfig(**SMTP_CONFIGURED)  # type: ignore[arg-type]
    assert configured.mail_enabled is True
    assert configured.mail_ready is False
    assert _ready().mail_ready is True


def test_none_turns_mail_off_even_where_smtp_is_configured() -> None:
    config = _ready(mail_transport="none")
    assert config.effective_mail_transport == "none"
    assert config.mail_enabled is False
    assert config.missing_mail_settings == ("mail_transport",)


def test_missing_settings_name_what_would_turn_mail_on() -> None:
    """"Unavailable" is only honest if it says what to set."""
    assert GatewayConfig().missing_mail_settings == ("smtp_host", "mail_from_email", "public_base_url")
    assert GatewayConfig(smtp_host="smtp.example.com").missing_mail_settings == (
        "mail_from_email",
        "public_base_url",
    )
    assert GatewayConfig(**SMTP_CONFIGURED).missing_mail_settings == ("public_base_url",)  # type: ignore[arg-type]
    assert _ready().missing_mail_settings == ()


def test_readiness_never_depends_on_a_validator_having_run() -> None:
    """A config that was never validated must still answer truthfully.

    An explicit ``smtp`` with nothing behind it is refused at startup, but the
    refusal is not what makes the answer here correct: ``effective_mail_transport``
    reports what a send would actually use, so it agrees with
    ``select_transport`` in every state, validated or not. A readiness answer
    that is only truthful because a check ran somewhere else is the shape of bug
    this design exists to rule out.
    """
    half_configured = GatewayConfig(mail_transport="smtp", smtp_host="smtp.example.com")

    assert half_configured.effective_mail_transport == "none"
    assert half_configured.mail_enabled is False
    assert half_configured.mail_ready is False
    assert select_transport(half_configured) is None
    assert Mailer(half_configured).can_send_links is False
    # And it still names what is missing rather than reporting nothing to fix.
    assert half_configured.missing_mail_settings == ("mail_from_email", "public_base_url")


def test_config_and_transport_never_disagree_about_whether_mail_exists() -> None:
    """The two readers of "is mail configured" must not be able to drift apart.

    ``/v1/bootstrap`` reads the config property and the invitation path asks the
    mailer; if those could differ, the dashboard would offer an affordance the
    request path refuses. Exhaustive over the settings that decide it, because
    the states that broke this before were the ones nobody thinks to write a
    case for.
    """
    for transport in MAIL_TRANSPORT_SETTINGS:
        for host in (None, "smtp.example.com"):
            for sender in (None, "otari@example.com"):
                for url in (None, "https://otari.example.com"):
                    config = GatewayConfig(
                        mail_transport=transport,
                        smtp_host=host,
                        mail_from_email=sender,
                        public_base_url=url,
                    )
                    state = (transport, host, sender, url)
                    assert config.mail_enabled == (select_transport(config) is not None), state
                    assert config.mail_ready == Mailer(config).can_send_links, state
                    # "Empty exactly when ready" is the contract GET /v1/settings/mail
                    # publishes, so it holds for every state and not just the tidy ones.
                    assert (config.missing_mail_settings == ()) == config.mail_ready, state


def test_an_unknown_transport_is_refused_at_load() -> None:
    with pytest.raises(ValueError, match="mail_transport must be one of"):
        GatewayConfig(mail_transport="sendgrid")


def test_asking_for_smtp_without_smtp_settings_fails_at_startup_not_at_send_time() -> None:
    config = GatewayConfig(mail_transport="smtp", smtp_host="smtp.example.com")
    with pytest.raises(ValueError, match="mail_from_email"):
        config.validate_mail_transport()


def test_selecting_console_warns_that_it_writes_tokens_to_the_log(
    gateway_logs: pytest.LogCaptureFixture,
) -> None:
    """The one sanctioned exception to never-log-a-token, so it says so out loud.

    A console "delivery" writes the message body, and an invitation body carries
    the accept token. The transport is opt-in per deployment and useless without
    the link, so the trade is announced at startup rather than redacted away.
    """
    GatewayConfig(mail_transport="console").validate_mail_transport()

    assert "token-bearing" in gateway_logs.text


def test_a_real_transport_warns_about_nothing(gateway_logs: pytest.LogCaptureFixture) -> None:
    GatewayConfig(**SMTP_CONFIGURED).validate_mail_transport()  # type: ignore[arg-type]
    GatewayConfig().validate_mail_transport()

    assert gateway_logs.text == ""


def test_the_auto_default_validates_clean_with_nothing_configured() -> None:
    """No mail is the ordinary state of a self-hosted deployment, not a misconfiguration."""
    GatewayConfig().validate_mail_transport()


# --- Transport selection ----------------------------------------------------


def test_no_transport_is_an_absent_object_not_a_disabled_one() -> None:
    """A surface asks whether mail exists; it never has to interpret a flag on a sender."""
    assert select_transport(GatewayConfig()) is None
    assert isinstance(select_transport(GatewayConfig(**SMTP_CONFIGURED)), SmtpTransport)  # type: ignore[arg-type]
    assert isinstance(select_transport(GatewayConfig(mail_transport="console")), ConsoleTransport)


def test_console_is_a_transport_without_any_smtp_settings() -> None:
    mailer = Mailer(GatewayConfig(mail_transport="console", public_base_url="https://otari.example.com"))
    assert mailer.transport_name == "console"
    assert mailer.is_configured is True
    assert mailer.can_send_links is True


# --- The mailer -------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_reports_the_no_transport_case_rather_than_raising() -> None:
    delivery = await Mailer(GatewayConfig()).send(to="ada@example.com", message=MESSAGE)
    assert delivery.delivered is False
    assert delivery.transport == "none"
    assert delivery.reason is not None


@pytest.mark.asyncio
async def test_send_reports_a_failure_rather_than_raising() -> None:
    """An unreachable host must not turn a mail failure into a request failure."""
    # Patched rather than dialing a closed port: on a network that black-holes
    # loopback instead of refusing, a real connect would wait out
    # SMTP_TIMEOUT_SECONDS before this assertion could pass.
    config = _ready()
    with patch("gateway.services.mail.transports.smtplib.SMTP", side_effect=OSError("connection refused")):
        delivery = await Mailer(config).send(to="ada@example.com", message=MESSAGE)
    assert delivery.delivered is False
    assert delivery.transport == "smtp"
    assert delivery.reason is not None


@pytest.mark.asyncio
async def test_a_failure_reason_never_carries_the_recipient_into_the_logs(
    gateway_logs: pytest.LogCaptureFixture,
) -> None:
    config = _ready()
    with patch("gateway.services.mail.transports.smtplib.SMTP", side_effect=OSError("connection refused")):
        await Mailer(config).send(to="ada@example.com", message=MESSAGE)
    assert "ada@example.com" not in gateway_logs.text
    assert "a***@example.com" in gateway_logs.text


@pytest.mark.asyncio
async def test_send_never_raises_even_on_a_malformed_header() -> None:
    """Defense in depth: message serialization runs before any socket I/O.

    ``as_string()`` is evaluated as a plain argument expression before
    ``client.sendmail(...)`` is even called, so this raises the same
    ``HeaderParseError`` a real server would, with no network needed to prove
    the mailer still reports a failure instead of propagating it.
    """
    message = MailMessage(subject="Broken\r\nheader: injected", html="<p>Hi</p>", text="Hi")
    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        mock_smtp.return_value.__enter__.return_value = MagicMock()
        # Bypasses build_mime's own sanitization by handing the header a value
        # that only email.header rejects once assembled.
        with patch("gateway.services.mail.transports.sanitize_header_value", side_effect=lambda value: value):
            delivery = await Mailer(_ready()).send(to="ada@example.com", message=message)
    assert delivery.delivered is False


@pytest.mark.asyncio
async def test_send_never_raises_on_a_crlf_in_an_address() -> None:
    """smtplib puts the envelope addresses on the wire itself, so sanitizing the header is not enough.

    ``build_mime`` sanitized the ``To`` header while ``sendmail`` received the
    raw value, so a CR/LF in a recipient left the message clean and the wire
    dirty: smtplib refused it with ``ValueError``, which was outside this
    transport's except clause and propagated out of ``Mailer.send`` into the
    caller's request. That is the never-raises contract broken, and it was
    reachable by any caller that had not validated its address first (both of
    today's had, which is precisely why nothing caught it).
    """
    config = _ready()
    injected = "ada@example.com>\r\nRCPT TO:<attacker@evil.example"

    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        client = MagicMock()
        mock_smtp.return_value.__enter__.return_value = client
        delivery = await Mailer(config).send(to=injected, message=MESSAGE)

    # Did not raise, which is the contract.
    assert delivery.transport == "smtp"
    # And the addresses smtplib was handed carry no newline, which is why it
    # cannot raise: asserting on the call rather than on the delivery outcome,
    # because a mocked server would accept anything and prove nothing.
    sender, recipients, _body = client.sendmail.call_args.args
    assert "\r" not in sender and "\n" not in sender
    for recipient in recipients:
        assert "\r" not in recipient, recipient
        assert "\n" not in recipient, recipient


def test_build_mime_puts_no_newline_in_any_header() -> None:
    """The header half of the same guarantee, asserted on the assembled message."""
    envelope = MailEnvelope(
        to="ada@example.com>\r\nBcc: attacker@evil.example",
        sender_email="otari@example.com\r\nBcc: attacker@evil.example",
        sender_name="Otari\r\nBcc: attacker@evil.example",
        message=MESSAGE,
    )

    assembled = build_mime(envelope)

    for header in ("From", "To", "Subject"):
        assert "\r" not in assembled[header], header
        assert "\n" not in assembled[header], header
    # And it serializes, which is what a raw newline would have prevented.
    assert assembled.as_string()


@pytest.mark.asyncio
async def test_send_survives_a_crlf_in_the_configured_from_name() -> None:
    """mail_from_name is operator config, not caller input the subject path already sanitizes.

    Without stripping it, a stray CR/LF reaches ``as_string()`` unescaped and
    raises ``HeaderParseError`` on every send, which the mailer would report as
    a silent undelivered with nothing pointing at the From name as the cause.
    """
    config = _ready(mail_from_name="Otari\r\nBcc: attacker@example.com")
    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        mock_smtp.return_value.__enter__.return_value = MagicMock()
        delivery = await Mailer(config).send(to="ada@example.com", message=MESSAGE)
    assert delivery.delivered is True


@pytest.mark.asyncio
async def test_a_templated_message_sends_over_a_configured_transport(
    gateway_logs: pytest.LogCaptureFixture,
) -> None:
    """The definition of done, end to end, with no SMTP server to stand up."""
    mailer = Mailer(GatewayConfig(mail_transport="console", public_base_url="https://otari.example.com"))
    delivery = await mailer.send_template(
        "mail_test",
        to="ada@example.com",
        subject="Otari test message",
        values={"PUBLIC_BASE_URL": "https://otari.example.com", "TRANSPORT": "console"},
    )
    assert delivery.delivered is True
    assert delivery.transport == "console"
    assert "Your Otari mail settings work" in gateway_logs.text
    # The recipient is redacted here too: these lines land in the same stream.
    assert "ada@example.com" not in gateway_logs.text


def test_require_ready_refuses_and_names_what_is_missing() -> None:
    """What a surface with no non-mail fallback (password reset) raises."""
    with pytest.raises(MailNotConfiguredError) as excinfo:
        Mailer(GatewayConfig()).require_ready()
    assert excinfo.value.missing == ("smtp_host", "mail_from_email", "public_base_url")
    assert "smtp_host" in str(excinfo.value)


def test_require_ready_passes_once_a_linked_message_can_be_sent() -> None:
    Mailer(_ready()).require_ready()


def test_a_configured_transport_without_a_public_url_still_cannot_send_links() -> None:
    mailer = Mailer(GatewayConfig(**SMTP_CONFIGURED))  # type: ignore[arg-type]
    assert mailer.is_configured is True
    assert mailer.can_send_links is False
    with pytest.raises(MailNotConfiguredError):
        mailer.require_ready()


def test_link_is_absolute_when_the_deployment_knows_its_address_and_relative_otherwise() -> None:
    assert Mailer(_ready()).link("/#/accept-invitation?token=abc") == (
        "https://otari.example.com/#/accept-invitation?token=abc"
    )
    assert Mailer(GatewayConfig()).link("/#/accept-invitation?token=abc") == "/#/accept-invitation?token=abc"


def test_link_does_not_double_a_trailing_slash() -> None:
    assert Mailer(_ready(public_base_url="https://otari.example.com/")).link("/x") == "https://otari.example.com/x"


# --- The template renderer --------------------------------------------------


def test_render_wraps_a_body_in_the_shared_layout() -> None:
    message = render_email("mail_test", subject="Otari test message", values={"PUBLIC_BASE_URL": "u", "TRANSPORT": "t"})
    assert message.html.startswith("<!doctype html>")
    assert "<title>Otari test message</title>" in message.html
    assert "Your Otari mail settings work" in message.html
    assert message.text.rstrip().endswith("Otari")


def test_a_placeholder_with_no_value_fails_here_rather_than_in_an_inbox() -> None:
    with pytest.raises(MailTemplateError, match="TRANSPORT"):
        render_email("mail_test", subject="Otari test message", values={"PUBLIC_BASE_URL": "u"})


def test_every_shipped_template_loads() -> None:
    """The name constraint below must not reject a template this package ships.

    ``_layout.html`` starts with an underscore, so a stricter pattern would have
    broken every message while each body template still looked fine on its own.
    """
    package = Path(mail_package.__file__).parents[2] / "templates" / "email"
    shipped = sorted(path.name for path in package.iterdir() if path.suffix in {".html", ".txt"})
    assert shipped, "no templates found; this test would pass vacuously"
    for name in shipped:
        assert _load(name)


def test_a_template_name_that_is_not_ours_is_refused() -> None:
    """``name`` is interpolated into a resource path and the result is cached forever."""
    for name in ("../../../etc/passwd", "..%2fsecrets.html", "/etc/passwd", "nope.exe", ""):
        with pytest.raises(MailTemplateError, match="valid email template name"):
            _load(name)


def test_a_missing_template_is_named() -> None:
    with pytest.raises(MailTemplateError, match="password_reset"):
        render_email("password_reset", subject="x", values={})


def test_a_reserved_value_name_is_refused_rather_than_silently_overwritten() -> None:
    with pytest.raises(MailTemplateError, match="SUBJECT"):
        render_email("mail_test", subject="x", values={"SUBJECT": "mine", "PUBLIC_BASE_URL": "u", "TRANSPORT": "t"})


def test_a_value_is_never_rescanned_as_a_placeholder() -> None:
    """One pass, so a value that looks like a placeholder is emitted verbatim.

    A sequential replace-per-value would substitute into text an earlier pass
    had just inserted, which is how a free-text organization name becomes an
    injection into its own email.
    """
    message = render_email(
        "mail_test",
        subject="{{TRANSPORT}}",
        values={"PUBLIC_BASE_URL": "{{TRANSPORT}}", "TRANSPORT": "console"},
    )
    assert message.subject == "console"
    assert "{{TRANSPORT}}" in message.text


def test_the_subject_is_stripped_of_newlines_but_the_body_is_not() -> None:
    message = render_email(
        "mail_test",
        subject="Test {{TRANSPORT}}",
        values={"PUBLIC_BASE_URL": "u", "TRANSPORT": "smtp\r\nBcc: attacker@example.com"},
    )
    assert "\r" not in message.subject
    assert "\n" not in message.subject


# --- The shape of the guard -------------------------------------------------


def test_the_mail_package_gates_nothing_behind_an_assert() -> None:
    """No availability check here may be an ``assert``, at any depth.

    ``python -O`` strips assertions, so one guarding a send has two different
    wrong behaviors depending on how the process was started: a crash at send
    time unoptimized, and a silent no-op optimized. Those are precisely the two
    answers #648 rules out, from one line. otari.ai's ``send_email`` is the live
    example (``assert settings.emails_enabled``), which is why this is asserted
    structurally rather than trusted to review: the mail package's guarantees
    are ``return``s and typed exceptions, and a future edit must not quietly
    swap one for an assertion.
    """
    package = Path(mail_package.__file__).parent
    modules = sorted(package.glob("*.py"))
    assert modules, "no mail modules found; this test would pass vacuously"

    offenders = [
        f"{path.name}:{node.lineno}"
        for path in modules
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Assert)
    ]
    assert offenders == []


# --- Address handling -------------------------------------------------------


def test_normalized_address_lowercases_trims_and_refuses_a_non_address() -> None:
    assert normalized_address("  Ada@Example.COM ") == "ada@example.com"
    assert normalized_address("ada@example") is None
    assert normalized_address("not an address") is None
    assert normalized_address("") is None
