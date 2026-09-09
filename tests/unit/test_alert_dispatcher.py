"""The alert dispatcher, the destination redaction, and the evaluator's pure math.

No database and no network. Apprise is exercised for real where it is pure
(``instantiate`` parses a URL without connecting) and stubbed where it would
dial (``async_notify``).

The redaction cases are the security-relevant half of this file: an Apprise URL
carries its credentials in the path, so a redaction that only masked userinfo
and query would return a live bot token to anybody who can read the rule list.
"""

import asyncio
from collections.abc import Iterator
from decimal import Decimal

import pytest

from gateway.services.alerts.dispatcher import (
    SEND_TIMEOUT_SECONDS,
    SOCKET_CONNECT_TIMEOUT_SECONDS,
    SOCKET_READ_TIMEOUT_SECONDS,
    UnsupportedAlertDestinationError,
    _bound_sockets,
    parse_destination,
    send_alert,
)
from gateway.services.alerts.evaluator import (
    KIND_EXCEEDED,
    KIND_WARNING,
    _format_amount,
    _kind_for,
    _percent,
    _worst_axis,
)
from gateway.services.url_safety import redact_alert_destination


class _Rule:
    """The two fields `_kind_for` reads, without a database row behind them."""

    def __init__(self, *, warn_at_percent: int | None, notify_on_exceeded: bool) -> None:
        self.warn_at_percent = warn_at_percent
        self.notify_on_exceeded = notify_on_exceeded


class _Ceiling:
    def __init__(self, **counters: object) -> None:
        for axis in ("current_spend", "reserved_spend"):
            setattr(self, axis, counters.get(axis, 0))
        for axis in (
            "current_tokens",
            "reserved_tokens",
            "current_requests",
            "reserved_requests",
        ):
            setattr(self, axis, counters.get(axis, 0))


class _Budget:
    def __init__(self, **caps: object) -> None:
        self.max_budget = caps.get("max_budget")
        self.token_limit = caps.get("token_limit")
        self.request_limit = caps.get("request_limit")


# --------------------------------------------------------------------------
# parse_destination
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "destination",
    [
        "json://example.com/hook",
        "jsons://example.com/hook",
        "form://example.com/hook",
        "mailto://user:pass@example.com",
    ],
)
def test_parse_destination_accepts_a_schema_apprise_knows(destination: str) -> None:
    """A supported schema returns the plugin's service name rather than raising."""
    assert parse_destination(destination)


@pytest.mark.parametrize(
    "destination",
    [
        "",
        "   ",
        "not-a-url",
        "https://example.com/hook",  # a bare http(s) URL is not an Apprise schema
        "definitelynotascheme://example.com",
    ],
)
def test_parse_destination_rejects_what_apprise_cannot_deliver_to(destination: str) -> None:
    with pytest.raises(UnsupportedAlertDestinationError):
        parse_destination(destination)


def test_parse_destination_never_echoes_the_rejected_url() -> None:
    """The message must not contain the input: it may hold a pasted real token.

    The rejection is shown to whoever submitted the form and is also carried
    into an API error body, so quoting the value back would put a credential
    somewhere it was never meant to be.
    """
    secret = "definitelynotascheme://xoxb-super-secret-token"
    with pytest.raises(UnsupportedAlertDestinationError) as caught:
        parse_destination(secret)
    assert "xoxb-super-secret-token" not in str(caught.value)


# --------------------------------------------------------------------------
# send_alert
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_alert_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport that accepts the notification reports delivered with no detail.

    Stubbed rather than dialed: a test that made a real request would pass or
    fail on whether the runner has egress, which `AGENTS.md` calls out as the
    thing to avoid.
    """

    async def _accept(*_: object, **__: object) -> bool:
        return True

    monkeypatch.setattr("apprise.Apprise.async_notify", _accept)
    result = await send_alert("json://example.com/hook", title="t", body="b")
    assert result.delivered is True
    assert result.detail is None


@pytest.mark.asyncio
async def test_send_alert_treats_no_matching_server_as_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``async_notify`` returns None when nothing matched, which is not success."""

    async def _nothing_matched(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr("apprise.Apprise.async_notify", _nothing_matched)
    result = await send_alert("json://example.com/hook", title="t", body="b")
    assert result.delivered is False
    assert result.detail


@pytest.mark.asyncio
async def test_send_alert_reports_a_rejected_destination_without_raising() -> None:
    """A URL Apprise will not add is a failed result, not an exception.

    The evaluator calls this inside a loop over rules; one unusable destination
    must not stop the rules after it from being evaluated in the same tick.
    """
    result = await send_alert("definitelynotascheme://host", title="t", body="b")
    assert result.delivered is False
    assert result.detail


@pytest.mark.asyncio
async def test_send_alert_turns_a_transport_exception_into_a_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apprise surfaces a plugin's own exception; the caller must never see it."""

    async def _explode(*_: object, **__: object) -> bool:
        raise RuntimeError("the transport blew up")

    monkeypatch.setattr("apprise.Apprise.async_notify", _explode)
    result = await send_alert("json://example.com/hook", title="t", body="b")
    assert result.delivered is False
    assert "RuntimeError" in (result.detail or "")


@pytest.mark.asyncio
async def test_send_alert_does_not_leak_the_exception_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the exception's class name is reported, never its message.

    A plugin that echoes its own URL into the error would otherwise put the
    token into the delivery row and the gateway log.
    """

    async def _explode(*_: object, **__: object) -> bool:
        raise RuntimeError("failed posting to json://user:hunter2@example.com/hook")

    monkeypatch.setattr("apprise.Apprise.async_notify", _explode)
    result = await send_alert("json://example.com/hook", title="t", body="b")
    assert "hunter2" not in (result.detail or "")


@pytest.mark.asyncio
async def test_send_alert_times_out_rather_than_holding_a_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destination that never answers is bounded by SEND_TIMEOUT_SECONDS.

    The timeout is patched down rather than waited out: what is under test is
    that ``wait_for`` wraps the call at all, not the constant's value.
    """
    monkeypatch.setattr("gateway.services.alerts.dispatcher.SEND_TIMEOUT_SECONDS", 0.01)

    async def _hang(*_: object, **__: object) -> bool:
        await asyncio.sleep(10)
        return True

    monkeypatch.setattr("apprise.Apprise.async_notify", _hang)
    result = await send_alert("json://example.com/hook", title="t", body="b")
    assert result.delivered is False
    assert "imed out" in (result.detail or "")


@pytest.mark.asyncio
async def test_send_alert_reraises_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown must still cancel a send in flight.

    ``CancelledError`` is deliberately not swallowed by the broad handler, or a
    cancelled lifespan would never finish.
    """

    async def _cancelled(*_: object, **__: object) -> bool:
        raise asyncio.CancelledError

    monkeypatch.setattr("apprise.Apprise.async_notify", _cancelled)
    with pytest.raises(asyncio.CancelledError):
        await send_alert("json://example.com/hook", title="t", body="b")


def test_socket_timeouts_are_clamped_down_from_a_greedy_url() -> None:
    """An operator's ``?cto=&rto=`` must not park a shared executor thread.

    ``asyncio.wait_for`` cannot cancel a thread already inside a socket read, so
    the URL's own timeouts are the real bound on thread occupancy.
    """
    import apprise

    client = apprise.Apprise()
    assert client.add("json://example.com/hook?cto=600&rto=900")
    _bound_sockets(client)
    for server in client:
        assert server.socket_connect_timeout == SOCKET_CONNECT_TIMEOUT_SECONDS
        assert server.socket_read_timeout == SOCKET_READ_TIMEOUT_SECONDS


def test_a_shorter_socket_timeout_is_left_alone() -> None:
    """Clamped, not overwritten: an operator asking for less keeps it."""
    import apprise

    client = apprise.Apprise()
    assert client.add("json://example.com/hook?cto=1&rto=2")
    _bound_sockets(client)
    for server in client:
        assert server.socket_connect_timeout == 1.0
        assert server.socket_read_timeout == 2.0


def test_bounding_sockets_survives_a_plugin_without_the_attributes() -> None:
    """A custom plugin need not derive the timeouts; one must not fail the pass."""

    class _Bare:
        pass

    class _FakeClient:
        def __iter__(self) -> Iterator[object]:
            return iter([_Bare()])

    _bound_sockets(_FakeClient())  # type: ignore[arg-type]


def test_send_timeout_is_a_sane_bound() -> None:
    """A regression guard on the constant, which bounds an executor thread."""
    assert 0 < SEND_TIMEOUT_SECONDS <= 60


# --------------------------------------------------------------------------
# redact_alert_destination
# --------------------------------------------------------------------------


def test_redaction_masks_a_slack_token_in_the_path() -> None:
    """The case `redact_url_secrets` gets wrong, which is why this exists."""
    redacted = redact_alert_destination("slack://xoxb-A/xoxb-B/xoxb-C/#alerts")
    assert "xoxb-A" not in redacted
    assert "xoxb-B" not in redacted
    assert "xoxb-C" not in redacted
    assert redacted.startswith("slack://")


def test_redaction_masks_a_discord_webhook_token() -> None:
    redacted = redact_alert_destination("discord://123456789/abcdefTOKEN")
    assert "abcdefTOKEN" not in redacted
    assert "123456789" not in redacted
    assert redacted.startswith("discord://")


def test_redaction_keeps_the_host_of_a_webhook_destination() -> None:
    """A json:// host is operator-chosen and is what makes the row recognizable."""
    redacted = redact_alert_destination("json://hooks.internal.example/incoming/SECRET")
    assert "hooks.internal.example" in redacted
    assert "SECRET" not in redacted


def test_redaction_masks_userinfo_and_query() -> None:
    redacted = redact_alert_destination("json://user:hunter2@example.com/hook?token=abc123")
    assert "hunter2" not in redacted
    assert "abc123" not in redacted
    assert "user" not in redacted


def test_redaction_survives_junk() -> None:
    """A value that will not parse redacts to the mask rather than raising.

    The column is written on every create and update, so a redaction that
    raised would turn a bad destination into a 500 instead of a validation
    error.
    """
    assert redact_alert_destination("") == "***"
    assert redact_alert_destination("no-scheme-at-all") == "***"


# --------------------------------------------------------------------------
# the evaluator's pure math
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("used", "cap", "expected"),
    [
        (0, 100, 0),
        (50, 100, 50),
        (799, 1000, 79),  # floored: 79.9 percent does not fire an 80 warning
        (800, 1000, 80),
        (1000, 1000, 100),
        (2500, 1000, 250),
        (0, 0, 0),  # a zero cap cannot be divided into
    ],
)
def test_percent_floors_and_survives_a_zero_cap(used: int, cap: int, expected: int) -> None:
    assert _percent(Decimal(used), Decimal(cap)) == expected


def test_percent_saturates_rather_than_formatting_an_absurd_ratio() -> None:
    """A cap lowered under an already-spent ceiling is clamped, not reported raw."""
    assert _percent(Decimal(10**9), Decimal(1)) == 999


def test_worst_axis_returns_none_when_nothing_is_capped() -> None:
    """A ceiling whose budget caps no axis can never cross a threshold."""
    assert _worst_axis(_Ceiling(), _Budget()) is None  # type: ignore[arg-type]


def test_worst_axis_counts_reservations_alongside_committed_spend() -> None:
    """Utilization matches the gate in ``scoped_budget_service.reserve``.

    That gate admits on ``committed + held <= cap``, so a ceiling holding 90 of
    a 100 cap is already refusing arrivals and must not report 10 percent.
    """
    axis, used, cap, percent = _worst_axis(  # type: ignore[misc]
        _Ceiling(current_spend=Decimal(10), reserved_spend=Decimal(80)),  # type: ignore[arg-type]
        _Budget(max_budget=Decimal(100)),  # type: ignore[arg-type]
    )
    assert axis == "spend"
    assert used == Decimal(90)
    assert cap == Decimal(100)
    assert percent == 90


def test_worst_axis_picks_the_axis_closest_to_refusing() -> None:
    """Tokens at 95 percent beat dollars at 10 percent, because tokens refuse first."""
    axis, _, _, percent = _worst_axis(  # type: ignore[misc]
        _Ceiling(  # type: ignore[arg-type]
            current_spend=Decimal(10),
            current_tokens=950,
            current_requests=1,
        ),
        _Budget(max_budget=Decimal(100), token_limit=1000, request_limit=1000),  # type: ignore[arg-type]
    )
    assert axis == "tokens"
    assert percent == 95


@pytest.mark.parametrize(
    ("warn", "on_exceeded", "percent", "expected"),
    [
        (80, True, 10, None),
        (80, True, 79, None),
        (80, True, 80, KIND_WARNING),
        (80, True, 99, KIND_WARNING),
        (80, True, 100, KIND_EXCEEDED),
        (80, True, 250, KIND_EXCEEDED),
        # A rule that only wants the refusal stays quiet through the warning band.
        (None, True, 90, None),
        (None, True, 100, KIND_EXCEEDED),
        # A rule that only wants the warning keeps warning past the cap. Going
        # silent at 100 would lose the alert at the moment it matters, on the
        # deployment that routes refusals through its own error monitoring.
        (80, False, 90, KIND_WARNING),
        (80, False, 100, KIND_WARNING),
        (80, False, 250, KIND_WARNING),
    ],
)
def test_kind_for_picks_at_most_one_alert(
    warn: int | None, on_exceeded: bool, percent: int, expected: str | None
) -> None:
    """Exceeded wins over the warning, and neither fires below the threshold."""
    rule = _Rule(warn_at_percent=warn, notify_on_exceeded=on_exceeded)
    assert _kind_for(rule, percent) == expected  # type: ignore[arg-type]


def test_format_amount_renders_each_unit_the_way_it_reads() -> None:
    """Dollars keep cents; counts are whole, not Decimal-tailed."""
    assert _format_amount("spend", Decimal("12.5")) == "$12.50"
    assert _format_amount("tokens", Decimal(1000)) == "1,000"
    assert _format_amount("requests", Decimal(7)) == "7"
