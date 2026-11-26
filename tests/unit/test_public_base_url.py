"""What this deployment accepts as its own externally-reachable address."""

import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig

_REFUSAL = "public_base_url must be an absolute http\\(s\\) URL"


def test_a_deployment_that_configured_no_address_is_not_misconfigured() -> None:
    assert GatewayConfig().public_base_url is None


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_address_reads_as_unset(value: str) -> None:
    assert GatewayConfig(public_base_url=value).public_base_url is None


@pytest.mark.parametrize(
    "url",
    ["https://otari.example.com", "http://otari.example.com:8443", "https://otari.example.com/gateway"],
)
def test_an_absolute_web_address_is_accepted_and_kept_verbatim(url: str) -> None:
    """A configured path and a trailing slash both survive, because only the ends are trimmed."""
    assert GatewayConfig(public_base_url=f" {url} ").public_base_url == url


@pytest.mark.parametrize(
    "url",
    [
        "otari.example.com",
        "//otari.example.com",
        "javascript:alert(1)",
        "https://",
        "https://:443",
        "https://otari.example.com:bad",
    ],
)
def test_an_address_a_browser_could_not_be_sent_to_is_refused(url: str) -> None:
    with pytest.raises(ValidationError, match=_REFUSAL):
        GatewayConfig(public_base_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://otari.example.com/#dashboard",
        "https://otari.example.com/gateway?region=eu#dashboard",
        # An empty delimiter breaks an appended path the same way a populated one does.
        "https://otari.example.com#",
        "https://otari.example.com/gateway#",
    ],
)
def test_an_address_carrying_a_fragment_is_refused(url: str) -> None:
    """A fragment belongs at the end of a URL, and this value is appended to."""
    with pytest.raises(ValidationError, match="public_base_url must carry no fragment"):
        GatewayConfig(public_base_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://otari.example.com?region=eu",
        "https://otari.example.com/gateway?region=eu",
        "https://otari.example.com?",
    ],
)
def test_an_address_carrying_a_query_string_is_refused(url: str) -> None:
    """Joining is plain concatenation here, so a query string would sit in the middle."""
    with pytest.raises(ValidationError, match="public_base_url must carry no query string"):
        GatewayConfig(public_base_url=url)


@pytest.mark.parametrize("url", ["https://otari\n.example.com", "https://otari\t.example.com", "https://a b.com"])
def test_an_address_carrying_whitespace_inside_it_is_refused(url: str) -> None:
    """Whitespace is dropped when a URL is parsed and kept when it is stored, so the two disagree.

    Surrounding whitespace is trimmed instead, because that is a transcription
    artifact rather than a different address.
    """
    with pytest.raises(ValidationError, match="public_base_url must carry no whitespace"):
        GatewayConfig(public_base_url=url)


def test_a_refusal_names_the_setting_and_not_the_value() -> None:
    """A rejected address may carry a credential, so the refusal names the setting only.

    NOTE: Asserted on the refusal this validator produces, not on the rendered
    ``ValidationError``. Pydantic appends the rejected input to that rendering,
    so the whole envelope does repeat the value whatever a validator says.
    """
    with pytest.raises(ValidationError) as refused:
        GatewayConfig(public_base_url="https://user:hunter2@otari.example.com:bad")

    assert "hunter2" not in refused.value.errors()[0]["msg"]


def test_an_address_carrying_a_credential_is_refused() -> None:
    with pytest.raises(ValidationError, match="public_base_url must carry no username or password"):
        GatewayConfig(public_base_url="https://user:secret@otari.example.com")


def test_an_address_missing_its_scheme_cannot_reach_a_running_deployment() -> None:
    """A complete mail configuration does not soften the rule."""
    with pytest.raises(ValidationError, match=_REFUSAL):
        GatewayConfig(
            public_base_url="otari.example.com",
            mail_transport="smtp",
            smtp_host="smtp.example.com",
            mail_from_email="otari@example.com",
        )
