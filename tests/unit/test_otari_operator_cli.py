"""`otari operator handover`, against a stand-in gateway."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from otari_agent.cli import cli

_ACCOUNT_ID = "bbbbbbbb-0000-0000-0000-000000000000"
# The keyword arguments each client the command built was given.
_CLIENT_OPTIONS: list[dict[str, Any]] = []


def _gateway(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.Request]:
    """Route every client the command builds through ``handler``, recording each request."""
    seen: list[httpx.Request] = []
    _CLIENT_OPTIONS.clear()
    real_client = httpx.Client

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def client(*args: Any, **kwargs: Any) -> httpx.Client:
        _CLIENT_OPTIONS.append(kwargs)
        return real_client(*args, transport=httpx.MockTransport(record), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    return seen


def _accounts(*emails: str) -> dict[str, Any]:
    return {
        "data": [
            {"id": _ACCOUNT_ID if index == 0 else f"other-{index}", "email": email}
            for index, email in enumerate(emails)
        ],
        "count": len(emails),
    }


def _ok(request: httpx.Request) -> httpx.Response:
    if request.method == "GET":
        return httpx.Response(200, json=_accounts("grace@example.com", "grace@example.com.au"))
    return httpx.Response(204)


def test_hands_the_role_to_the_account_with_that_exact_address(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _gateway(monkeypatch, _ok)

    result = CliRunner().invoke(
        cli,
        ["operator", "handover", "Grace@Example.com", "--url", "https://otari.test/", "--yes"],
        env={"OTARI_MASTER_KEY": "otari-mk-test"},
    )

    assert result.exit_code == 0, result.output
    assert "grace@example.com is now the bootstrap operator" in result.output
    assert "rotate it" in result.output
    listed, posted = seen
    assert listed.url.path == "/api/v1/admin/users"
    assert listed.url.params["search"] == "Grace@Example.com"
    assert posted.method == "POST"
    assert posted.url.path == f"/api/v1/admin/users/{_ACCOUNT_ID}/bootstrap-operator"
    assert posted.headers["Authorization"] == "Bearer otari-mk-test"


def test_asks_before_handing_over_and_sends_nothing_when_declined(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _gateway(monkeypatch, _ok)

    result = CliRunner().invoke(
        cli,
        ["operator", "handover", "grace@example.com"],
        env={"OTARI_MASTER_KEY": "otari-mk-test"},
        input="n\n",
    )

    assert result.exit_code == 1
    assert [request.method for request in seen] == ["GET"]


def test_an_unknown_address_is_reported_and_nothing_is_posted(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _gateway(monkeypatch, lambda request: httpx.Response(200, json=_accounts("grace@example.com.au")))

    result = CliRunner().invoke(
        cli, ["operator", "handover", "grace@example.com", "--yes"], env={"OTARI_MASTER_KEY": "otari-mk-test"}
    )

    assert result.exit_code == 1
    assert "No account on this deployment signs in as grace@example.com" in result.output
    assert [request.method for request in seen] == ["GET"]


def test_the_gateway_reason_for_a_refusal_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_accounts("grace@example.com"))
        return httpx.Response(400, json={"detail": "Make this account an operator before handing it the role"})

    _gateway(monkeypatch, refuse)

    result = CliRunner().invoke(
        cli, ["operator", "handover", "grace@example.com", "--yes"], env={"OTARI_MASTER_KEY": "otari-mk-test"}
    )

    assert result.exit_code == 1
    assert "(400): Make this account an operator" in result.output


def test_a_wrong_master_key_is_named_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    _gateway(monkeypatch, lambda request: httpx.Response(401, text=json.dumps({"detail": "Invalid master key"})))

    result = CliRunner().invoke(
        cli, ["operator", "handover", "grace@example.com", "--yes"], env={"OTARI_MASTER_KEY": "wrong"}
    )

    assert result.exit_code == 1
    assert "refused the master key" in result.output


def test_the_master_key_is_required() -> None:
    result = CliRunner().invoke(cli, ["operator", "handover", "grace@example.com"], env={"OTARI_MASTER_KEY": ""})

    assert result.exit_code == 2
    assert "master key is required" in result.output


def test_refuses_to_send_the_master_key_over_plain_http_to_another_host(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _gateway(monkeypatch, _ok)

    result = CliRunner().invoke(
        cli,
        ["operator", "handover", "grace@example.com", "--url", "http://otari.example.com", "--yes"],
        env={"OTARI_MASTER_KEY": "otari-mk-test"},
    )

    assert result.exit_code == 2
    assert "without HTTPS" in result.output
    assert seen == []


def test_a_redirect_is_reported_rather_than_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    def redirect(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_accounts("grace@example.com"))
        return httpx.Response(307, headers={"Location": "https://elsewhere.test/"}, text="")

    seen = _gateway(monkeypatch, redirect)

    result = CliRunner().invoke(
        cli, ["operator", "handover", "grace@example.com", "--yes"], env={"OTARI_MASTER_KEY": "otari-mk-test"}
    )

    assert result.exit_code == 1
    assert "(307)" in result.output
    assert len(seen) == 2


def test_addresses_differing_only_in_case_are_refused_as_ambiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _gateway(
        monkeypatch, lambda request: httpx.Response(200, json=_accounts("grace@example.com", "Grace@example.com"))
    )

    result = CliRunner().invoke(
        cli, ["operator", "handover", "grace@example.com", "--yes"], env={"OTARI_MASTER_KEY": "otari-mk-test"}
    )

    assert result.exit_code == 1
    assert "More than one account" in result.output
    assert [request.method for request in seen] == ["GET"]


def test_plain_http_to_this_machine_ignores_proxies_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A proxy would otherwise carry the master key off the machine in cleartext."""
    _gateway(monkeypatch, _ok)

    result = CliRunner().invoke(
        cli,
        ["operator", "handover", "grace@example.com", "--yes"],
        env={"OTARI_MASTER_KEY": "otari-mk-test", "HTTP_PROXY": "http://proxy.example:8080"},
    )

    assert result.exit_code == 0, result.output
    assert _CLIENT_OPTIONS[0]["trust_env"] is False


def test_a_search_with_more_matches_than_one_page_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def truncated(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**_accounts("grace@example.com"), "count": 5000})

    seen = _gateway(monkeypatch, truncated)

    result = CliRunner().invoke(
        cli, ["operator", "handover", "grace@example.com", "--yes"], env={"OTARI_MASTER_KEY": "otari-mk-test"}
    )

    assert result.exit_code == 1
    assert "accounts match" in result.output
    assert [request.method for request in seen] == ["GET"]
