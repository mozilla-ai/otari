"""The alert destination allowlist and its SSRF gate.

The two scheme groups are the gate, so what is in each one is worth asserting
directly. IP literals throughout: ``reject_internal_host`` takes the literal
branch and never resolves, so none of this needs DNS, unlike
``tests/unit/test_url_safety.py``.
"""

import pytest

from gateway.services.url_safety import UnsafeURLError
from otari_alerts.destinations import (
    ALERT_SCHEMES_WITH_FIXED_ENDPOINT,
    ALERT_SCHEMES_WITH_OPERATOR_HOST,
    SUPPORTED_ALERT_SCHEMES,
    validate_alert_destination,
)


@pytest.mark.parametrize(
    "scheme",
    [
        # The ones a self-hosted deployment reaches for, and the webhook-shaped
        # ones an earlier version of this gate covered on its own.
        "mailto",
        "mailtos",
        "gotify",
        "ntfy",
        "matrix",
        "rocket",
        "mmost",
        "ncloud",
        "json",
        "jsons",
        "xml",
        "form",
    ],
)
@pytest.mark.asyncio
async def test_a_schema_that_names_a_host_is_address_checked(scheme: str) -> None:
    """Every one of these dials the host in the URL, so every one is checked.

    ``mailto`` is the case that motivated the split: Apprise sends it to the
    SMTP server named in the URL, so a gate covering only the webhook-shaped
    schemas would let it through unchecked.
    """
    assert scheme in ALERT_SCHEMES_WITH_OPERATOR_HOST
    with pytest.raises(UnsafeURLError):
        await validate_alert_destination(scheme, "169.254.169.254")


@pytest.mark.parametrize("scheme", ["slack", "discord", "pagerduty", "tgram"])
@pytest.mark.asyncio
async def test_a_fixed_endpoint_schema_has_no_address_to_check(scheme: str) -> None:
    """The netloc is a token, so checking it would reject a working rule."""
    assert scheme in ALERT_SCHEMES_WITH_FIXED_ENDPOINT
    await validate_alert_destination(scheme, "10")


@pytest.mark.asyncio
async def test_a_public_host_is_accepted() -> None:
    await validate_alert_destination("json", "93.184.216.34")


@pytest.mark.asyncio
async def test_a_host_bearing_schema_with_no_host_is_refused() -> None:
    """Fail closed. A schema in the checked group must produce something to check."""
    with pytest.raises(UnsafeURLError):
        await validate_alert_destination("json", None)


@pytest.mark.asyncio
async def test_the_override_opens_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The self-hosted case: an internal chat server on the deployment's network."""
    monkeypatch.setenv("OTARI_ALERT_ALLOW_PRIVATE_HOSTS", "true")
    await validate_alert_destination("json", "10.0.0.5")


@pytest.mark.asyncio
async def test_another_gates_override_does_not_open_this_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each SSRF gate has its own override; turning one on must not widen another."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS", "true")
    monkeypatch.setenv("OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS", "true")
    with pytest.raises(UnsafeURLError):
        await validate_alert_destination("json", "10.0.0.5")


def test_the_two_scheme_groups_do_not_overlap() -> None:
    """A schema in both would be checked or skipped depending on read order."""
    assert not (ALERT_SCHEMES_WITH_OPERATOR_HOST & ALERT_SCHEMES_WITH_FIXED_ENDPOINT)
    assert SUPPORTED_ALERT_SCHEMES == ALERT_SCHEMES_WITH_OPERATOR_HOST | ALERT_SCHEMES_WITH_FIXED_ENDPOINT
