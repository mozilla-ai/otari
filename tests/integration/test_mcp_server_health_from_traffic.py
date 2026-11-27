"""What tool discovery against a stored MCP server records in its health.

A server that fails every other discovery must not read healthy because the last one happened to
answer, a deadline that cuts a listing short is the server's failure, and a wait for one of the
gateway's own slots is not.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp.shared.exceptions import MCPError
from mcp.types import ListToolsResult, PaginatedRequestParams
from mcp.types import Tool as MCPTool
from sqlalchemy.orm import Session

from gateway.api.routes import mcp as mcp_route
from gateway.core.config import API_ROOT
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import Organization, Workspace
from gateway.models.tools import WorkspaceMcpServer
from gateway.services import mcp_stateless
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.tools import McpServerContacts

PUBLIC_URL = "https://93.184.216.34/mcp"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture
def contacts(monkeypatch: pytest.MonkeyPatch) -> McpServerContacts:
    """A recorder of its own, so no other test's throttle holds a write back."""
    recorder = McpServerContacts()
    monkeypatch.setattr(mcp_route, "mcp_server_contacts", recorder)
    return recorder


class _Session:
    async def list_tools(self, *, params: PaginatedRequestParams | None = None) -> ListToolsResult:
        return ListToolsResult(tools=[MCPTool(name="search", input_schema={"type": "object"})])


def _answers(monkeypatch: pytest.MonkeyPatch, outcomes: list[str]) -> None:
    """Open a session that, in turn, answers (``ok``), errors (``error``) or never answers (``hang``)."""

    @asynccontextmanager
    async def open_session(*args: Any, **kwargs: Any) -> AsyncIterator[_Session]:
        outcome = outcomes.pop(0)
        if outcome == "error":
            raise MCPError(-32603, "Internal error: upstream said sk-live-secret")
        if outcome == "hang":
            await asyncio.sleep(30)
        yield _Session()

    monkeypatch.setattr(mcp_stateless, "open_session", open_session)


def _store_server(test_db: Session, api_key_id: str, *, workspace_id: uuid.UUID | None = None) -> WorkspaceMcpServer:
    key = test_db.get(APIKey, api_key_id)
    assert key is not None and key.workspace_id is not None
    workspace = test_db.get(Workspace, workspace_id or key.workspace_id)
    assert workspace is not None
    row = WorkspaceMcpServer(
        organization_id=workspace.organization_id,
        workspace_id=workspace.id,
        name="apify",
        url=PUBLIC_URL,
        encrypted_token=encrypt_secret("apify_api_token"),
    )
    test_db.add(row)
    test_db.commit()
    test_db.refresh(row)
    return row


def _discover(client: TestClient, headers: dict[str, str], server_id: uuid.UUID) -> int:
    return client.get(f"{API_ROOT}/mcp/servers/{server_id}/tools", headers=headers).status_code


def _revalidate(client: TestClient, headers: dict[str, str], server_id: uuid.UUID) -> int:
    return client.get(
        f"{API_ROOT}/mcp/servers/{server_id}/tools", headers={**headers, "Cache-Control": "no-cache"}
    ).status_code


def _health(client: TestClient, contacts: McpServerContacts, test_db: Session, row: WorkspaceMcpServer) -> Any:
    assert client.portal is not None
    client.portal.call(contacts.drain)
    test_db.expire_all()
    stored = test_db.get(WorkspaceMcpServer, row.id)
    assert stored is not None
    return stored.health


def test_a_server_that_answers_between_failures_reads_degraded_with_its_last_failure(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    contacts: McpServerContacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _store_server(test_db, api_key_obj["id"])
    _answers(monkeypatch, ["error", "ok"])

    assert _discover(client, api_key_header, row.id) == 502
    failing = _health(client, contacts, test_db, row)
    assert failing["status"] == "failing" and failing["error"] == "not_mcp"

    assert _discover(client, api_key_header, row.id) == 200
    health = _health(client, contacts, test_db, row)
    assert health["status"] == "degraded"
    assert health["error"] is None
    assert health["tools"] == ["search"]
    assert health["last_failure"] == {"failed_at": failing["checked_at"], "error": "not_mcp", "http_status": None}
    for stored in (failing, health):
        # A cause by name only: never the server's own message, its address or its credential.
        text = json.dumps(stored)
        for secret in ("sk-live-secret", "Internal error", "93.184.216.34", "apify_api_token"):
            assert secret not in text, secret


def test_a_kept_listing_records_nothing_and_a_failed_revalidation_records_the_failure(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    contacts: McpServerContacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _store_server(test_db, api_key_obj["id"])
    outcomes = ["ok", "error"]
    _answers(monkeypatch, outcomes)

    assert _discover(client, api_key_header, row.id) == 200
    assert _discover(client, api_key_header, row.id) == 200
    assert outcomes == ["error"], "the second discovery should be served from the kept listing"
    assert _health(client, contacts, test_db, row)["status"] == "healthy"

    assert _revalidate(client, api_key_header, row.id) == 502

    health = _health(client, contacts, test_db, row)
    assert health["status"] == "failing" and health["last_failure"]["error"] == "not_mcp"


def test_discovery_of_another_workspace_s_server_records_nothing_on_it(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    contacts: McpServerContacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    organization = test_db.query(Organization).first()
    assert organization is not None
    other = Workspace(name="Other", organization_id=organization.id)
    test_db.add(other)
    test_db.commit()
    row = _store_server(test_db, api_key_obj["id"], workspace_id=other.id)
    _answers(monkeypatch, ["error"])

    assert _discover(client, api_key_header, row.id) == 404

    assert _health(client, contacts, test_db, row) is None


def test_a_listing_the_route_deadline_cuts_short_is_recorded_as_a_timeout(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    contacts: McpServerContacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _store_server(test_db, api_key_obj["id"])
    _answers(monkeypatch, ["hang"])
    monkeypatch.setattr(mcp_stateless, "DISCOVERY_TOTAL_TIMEOUT_S", 0.5)

    assert _discover(client, api_key_header, row.id) == 502

    health = _health(client, contacts, test_db, row)
    assert health["status"] == "failing" and health["error"] == "timeout"
    assert health["last_failure"]["error"] == "timeout"


@pytest.mark.parametrize(
    ("admission_timeout_s", "total_timeout_s", "status"),
    [
        pytest.param(0.05, 15.0, 503, id="refused-for-capacity"),
        pytest.param(5.0, 0.3, 502, id="deadline-spent-waiting-for-a-slot"),
    ],
)
def test_a_wait_for_the_gateway_s_own_capacity_says_nothing_about_the_server(
    client: TestClient,
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    contacts: McpServerContacts,
    monkeypatch: pytest.MonkeyPatch,
    admission_timeout_s: float,
    total_timeout_s: float,
    status: int,
) -> None:
    row = _store_server(test_db, api_key_obj["id"])
    _answers(monkeypatch, ["ok"])
    full = mcp_stateless.ConcurrencyGate(limit=1, admission_timeout_s=admission_timeout_s)
    full._semaphore = asyncio.Semaphore(0)
    monkeypatch.setattr(mcp_stateless, "DISCOVERY_GATE", full)
    monkeypatch.setattr(mcp_stateless, "DISCOVERY_TOTAL_TIMEOUT_S", total_timeout_s)

    assert _discover(client, api_key_header, row.id) == status

    assert _health(client, contacts, test_db, row) is None
