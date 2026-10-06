"""A hosted control plane starts each new workspace with code execution on.

There a workspace with no code-execution policy may not run code, so every path
that creates a workspace stages an enabled policy beside it. A standalone
deployment reads no policy as on already and stages nothing.
"""

import uuid
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.tenancy import Workspace
from gateway.models.tools import WorkspaceCodeExecutionPolicy


def _policies(db_session_factory: Callable[[], Session]) -> dict[uuid.UUID, bool]:
    session = db_session_factory()
    try:
        rows = session.execute(select(WorkspaceCodeExecutionPolicy)).scalars().all()
        return {row.workspace_id: row.enabled for row in rows}
    finally:
        session.close()


def _workspaces_of(db_session_factory: Callable[[], Session], organization_id: str) -> set[uuid.UUID]:
    session = db_session_factory()
    try:
        rows = session.execute(
            select(col(Workspace.id)).where(col(Workspace.organization_id) == uuid.UUID(organization_id))
        )
        return set(rows.scalars().all())
    finally:
        session.close()


@pytest.fixture
def hosted(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(test_config, "mode", "hosted")
    assert test_config.is_hosted_mode


def _create_workspace(client: TestClient, master_key_header: dict[str, str]) -> uuid.UUID:
    response = client.post(f"{API_ROOT}/workspaces", json={"name": "Research"}, headers=master_key_header)
    assert response.status_code == 201, response.text
    return uuid.UUID(response.json()["id"])


@pytest.mark.usefixtures("hosted")
def test_a_new_workspace_starts_with_code_execution_on(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    workspace_id = _create_workspace(client, master_key_header)

    assert _policies(db_session_factory).get(workspace_id) is True


@pytest.mark.usefixtures("hosted")
def test_a_new_organization_s_workspace_starts_with_code_execution_on(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    response = client.post(f"{API_ROOT}/organizations", json={"name": "Acme"}, headers=master_key_header)
    assert response.status_code == 201, response.text

    workspaces = _workspaces_of(db_session_factory, response.json()["id"])
    policies = _policies(db_session_factory)
    assert workspaces
    assert all(policies.get(workspace) is True for workspace in workspaces)


@pytest.mark.usefixtures("hosted")
def test_a_self_serve_signup_s_workspace_starts_with_code_execution_on(
    client: TestClient,
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory: Callable[[], Session],
) -> None:
    monkeypatch.setattr(test_config, "open_signup", True)
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")
    before = _policies(db_session_factory)

    response = client.post(
        f"{API_ROOT}/auth/signup",
        json={"email": "new-tenant@example.com", "password": "a-real-password"},  # pragma: allowlist secret
    )
    assert response.status_code == 200, response.text

    added = {workspace: enabled for workspace, enabled in _policies(db_session_factory).items() if workspace not in before}
    assert list(added.values()) == [True]


def test_standalone_stages_no_policy_for_a_new_workspace(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    workspace_id = _create_workspace(client, master_key_header)

    assert workspace_id not in _policies(db_session_factory)
