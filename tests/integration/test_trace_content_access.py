"""Who may read captured content and change a workspace's capture, and the record every read leaves.

Content is its session's own user's. An organization admin reads it only where the workspace allows
it, and the deployment operator only by breaking glass with a reason. A member who did not run the
session reads its metadata but is refused its content, and is refused the capture settings. Another
organization's admin finds nothing: their ids answer 404, like ones that do not exist.
"""

import logging
import time
import uuid
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.tenancy import Organization, User, Workspace
from gateway.models.traces import Trace, TraceContentAccess, TraceSpanContent
from gateway.models.users import User as BillingUser
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME
from gateway.services.secret_box import generate_secret_key

from .conftest import MODEL_NAME, build_test_client
from .test_trace_read_api import _identity

_SECRET = "the merger closes on friday"
_MESSAGES = [
    {"role": "user", "content": "x"},
    {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "Bash", "arguments": '{"command": "cat notes"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": _SECRET},
]


@pytest.fixture
def client(
    test_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[TestClient]:
    """A client on a deployment that permits capture, whatever the default ceiling is."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield from build_test_client(
        test_config.model_copy(update={"trace_content_capture_max": "full", "files_local_dir": str(tmp_path)})
    )


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="c",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[
            Choice(index=0, message=ChatCompletionMessage(role="assistant", content="done"), finish_reason="stop")
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


@pytest.fixture
def captured(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> dict[str, str]:
    """One tool span with captured content in a session its owner ran, and a member, an admin and an outsider."""
    workspace_id = client.get(f"{API_ROOT}/keys/{api_key_obj['id']}", headers=master_key_header).json()["workspace_id"]
    enabled = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/trace-settings",
        json={"content_capture": "tool_io"},
        headers=master_key_header,
    )
    assert enabled.status_code == status.HTTP_200_OK, enabled.text
    with patch("gateway.api.routes.chat.acompletion", return_value=_completion()):
        sent = client.post(
            f"{API_ROOT}/chat/completions", json={"model": MODEL_NAME, "messages": _MESSAGES}, headers=api_key_header
        )
    assert sent.status_code == status.HTTP_200_OK, sent.text

    deadline = time.monotonic() + 10
    while True:
        with db_session_factory() as session:
            rows = list(session.scalars(select(TraceSpanContent)))
        if rows or time.monotonic() > deadline:
            break
        time.sleep(0.1)
    [row] = rows

    with db_session_factory() as session:
        workspace = session.get(Workspace, uuid.UUID(workspace_id))
        assert workspace is not None
        organization_id = workspace.organization_id
        other = Organization(name="Other", slug="other-c")
        session.add(other)
        session.commit()
        admin = _identity(session, email="a@c.test", organization_id=organization_id, role="admin")
        owner = _identity(
            session, email="w@c.test", organization_id=organization_id, role="member", workspace_ids=(workspace.id,)
        )
        ids = {user.email: user.id for user in session.scalars(select(User))}
        # A member's own key bills to their identity's id, which is how a session names its owner.
        session.add(BillingUser(user_id=str(ids["w@c.test"])))
        session.flush()
        trace = session.get(Trace, (uuid.UUID(workspace_id), row.trace_id))
        assert trace is not None
        trace.user_id = str(ids["w@c.test"])
        session.commit()
        return {
            "admin_id": str(ids["a@c.test"]),
            "owner_id": str(ids["w@c.test"]),
            "owner": owner,
            "workspace_id": workspace_id,
            "trace_id": row.trace_id,
            "span_id": row.span_id,
            "member": _identity(
                session,
                email="m@c.test",
                organization_id=organization_id,
                role="member",
                workspace_ids=(workspace.id,),
            ),
            "admin": admin,
            "outsider": _identity(session, email="o@c.test", organization_id=other.id, role="owner"),
        }


@contextmanager
def _signed_in(client: TestClient, cookie: str) -> Iterator[TestClient]:
    client.cookies.set(SESSION_COOKIE_NAME, cookie)
    try:
        yield client
    finally:
        client.cookies.clear()


@contextmanager
def _audit_lines() -> Iterator[list[str]]:
    """The gateway logger's records while the block runs; it does not propagate, so caplog cannot see them."""
    lines: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    gateway_logger = logging.getLogger("gateway")
    handler, level = _Collect(level=logging.INFO), gateway_logger.level
    gateway_logger.addHandler(handler)
    gateway_logger.setLevel(logging.INFO)
    try:
        yield lines
    finally:
        gateway_logger.removeHandler(handler)
        gateway_logger.setLevel(level)


def _content_url(captured: dict[str, str], prefix: str = "/organizations/me/traces") -> str:
    return f"{API_ROOT}{prefix}/{captured['trace_id']}/spans/{captured['span_id']}/content"


def _reads(factory: Callable[[], Session]) -> list[TraceContentAccess]:
    with factory() as session:
        return list(session.scalars(select(TraceContentAccess)))


def test_the_sessions_own_user_reads_its_content_and_the_read_is_recorded(
    client: TestClient, captured: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    with _signed_in(client, captured["owner"]):
        response = client.get(_content_url(captured))

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["fields"]["result"] == _SECRET
    [read] = _reads(db_session_factory)
    assert (read.reader_kind, read.reader, read.reason) == ("owner", f"user:{captured['owner_id']}", None)


def test_a_member_who_did_not_run_the_session_sees_it_but_not_what_it_said(
    client: TestClient, captured: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    with _signed_in(client, captured["member"]):
        detail = client.get(f"{API_ROOT}/organizations/me/traces/{captured['trace_id']}")
        content = client.get(_content_url(captured))

    assert detail.status_code == status.HTTP_200_OK, detail.text
    assert content.status_code == status.HTTP_403_FORBIDDEN
    assert _SECRET not in content.text
    assert _reads(db_session_factory) == []


def test_an_admin_reads_content_only_where_the_workspace_allows_it(
    client: TestClient, captured: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    settings = f"{API_ROOT}/workspaces/{captured['workspace_id']}/trace-settings"
    with _signed_in(client, captured["admin"]):
        refused = client.get(_content_url(captured))
        allowed = client.put(settings, json={"admin_content_access": True})
        read = client.get(_content_url(captured))
        log = client.get(f"{settings}/content-access")

    assert refused.status_code == status.HTTP_403_FORBIDDEN
    assert allowed.json()["admin_content_access"] is True
    assert allowed.json()["content_capture"] == "tool_io", "changing one setting leaves the other"
    assert read.status_code == status.HTTP_200_OK, read.text
    [entry] = log.json()["items"]
    assert (entry["reader_kind"], entry["reader"]) == ("admin", f"user:{captured['admin_id']}")


def test_another_organizations_admin_finds_no_content(client: TestClient, captured: dict[str, str]) -> None:
    with _signed_in(client, captured["outsider"]):
        response = client.get(_content_url(captured))

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert _SECRET not in response.text


def test_the_operator_reads_content_only_by_breaking_glass_with_a_reason(
    client: TestClient,
    master_key_header: dict[str, str],
    captured: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    url = _content_url(captured, "/traces")
    plain = client.get(url, headers=master_key_header)
    no_reason = client.post(f"{url}/break-glass", json={"reason": "why"}, headers=master_key_header)
    with _audit_lines() as lines:
        broken = client.post(
            f"{url}/break-glass", json={"reason": "Legal hold, case 2026-114"}, headers=master_key_header
        )

    assert plain.status_code in (status.HTTP_404_NOT_FOUND, status.HTTP_405_METHOD_NOT_ALLOWED)
    assert no_reason.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert broken.status_code == status.HTTP_200_OK, broken.text
    assert broken.json()["fields"]["result"] == _SECRET
    [read] = _reads(db_session_factory)
    assert (read.reader_kind, read.reader, read.reason) == ("break_glass", "master_key", "Legal hold, case 2026-114")
    assert any(line.startswith("Break-glass trace content read") for line in lines)


def test_only_the_workspaces_admins_see_who_read_its_content(client: TestClient, captured: dict[str, str]) -> None:
    url = f"{API_ROOT}/workspaces/{captured['workspace_id']}/trace-settings/content-access"
    with _signed_in(client, captured["member"]):
        member = client.get(url)
    with _signed_in(client, captured["outsider"]):
        outsider = client.get(url)

    assert (member.status_code, outsider.status_code) == (403, 404)


@pytest.mark.parametrize(("who", "expected"), [("member", 403), ("outsider", 404)])
def test_only_an_admin_of_the_workspace_reads_or_changes_its_capture(
    client: TestClient,
    captured: dict[str, str],
    db_session_factory: Callable[[], Session],
    who: str,
    expected: int,
) -> None:
    url = f"{API_ROOT}/workspaces/{captured['workspace_id']}/trace-settings"
    with _signed_in(client, captured[who]):
        read = client.get(url)
        changed = client.put(url, json={"content_capture": "full"})
        purged = client.post(f"{url}/purge-content")

    assert (read.status_code, changed.status_code, purged.status_code) == (expected, expected, expected)
    with db_session_factory() as session:
        assert len(list(session.scalars(select(TraceSpanContent)))) == 1, "a refused purge deletes nothing"
    with _signed_in(client, captured["admin"]):
        assert client.get(url).json()["content_capture"] == "tool_io", "a refused change changes nothing"
