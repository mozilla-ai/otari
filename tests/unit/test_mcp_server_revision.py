"""The derived stored-server revision (R-DISC-3, R-RES-3).

The revision is what binds an application's authorization decision to the
configuration Otari executed against. It has to be a pure function of the four
execution-relevant fields so that every worker and replica derives the same
value, and it has to ignore the two fields an operator may retitle without
invalidating a pending approval.
"""

from __future__ import annotations

import re
import uuid

import pytest

from gateway.models.mcp import ResolvedMcpServer

REVISION_FORMAT = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

SERVER_ID = uuid.UUID("2c948a61-dc96-4cd8-96bb-8e1434bf424e")


def _server(**overrides: object) -> ResolvedMcpServer:
    fields: dict[str, object] = {
        "id": SERVER_ID,
        "name": "github",
        "url": "https://mcp.example.com/mcp",
        "authorization_token": "server-secret",
        "enabled": True,
        "allowed_tools": ["create_issue", "list_issues"],
    }
    fields.update(overrides)
    return ResolvedMcpServer(**fields)  # type: ignore[arg-type]


def test_revision_matches_the_published_format() -> None:
    assert REVISION_FORMAT.match(_server().revision)


def test_repeated_resolution_of_an_unchanged_server_derives_one_revision() -> None:
    assert _server().revision == _server().revision


@pytest.mark.parametrize(
    "change",
    [
        {"url": "https://other.example.com/mcp"},
        {"authorization_token": "rotated-secret"},
        {"authorization_token": None},
        {"enabled": False},
        {"allowed_tools": ["create_issue"]},
        {"allowed_tools": None},
        {"allowed_tools": []},
    ],
)
def test_execution_relevant_change_moves_the_revision(change: dict[str, object]) -> None:
    assert _server(**change).revision != _server().revision


def test_allowlist_order_does_not_move_the_revision() -> None:
    assert _server(allowed_tools=["list_issues", "create_issue"]).revision == _server().revision


def test_absent_and_empty_allowlists_are_different_revisions() -> None:
    """The two are opposite policies (R-RES-4), so they must not collide."""
    assert _server(allowed_tools=None).revision != _server(allowed_tools=[]).revision


@pytest.mark.parametrize("change", [{"name": "github-enterprise"}, {"purpose_hint": "issues"}])
def test_display_only_change_leaves_the_revision_alone(change: dict[str, object]) -> None:
    assert _server(**change).revision == _server().revision


def test_revision_does_not_disclose_the_credential() -> None:
    assert "server-secret" not in _server().revision
