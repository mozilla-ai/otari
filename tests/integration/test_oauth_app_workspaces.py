"""Consent searches writable workspaces within the current organization."""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT
from gateway.models.tenancy import Workspace, WorkspaceMember
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME

from .test_oauth_apps import test_config as test_config
from .test_organization_member_keys import _World
from .test_organization_member_keys import world as world

PATH = f"{API_ROOT}/oauth/apps/workspaces"


@pytest.mark.parametrize(
    "who,expected",
    [
        ("alpha_owner", {"alpha_one", "alpha_two"}),
        ("alpha_member", {"alpha_one"}),
        ("alpha_viewer", set()),
        ("alpha_workspace_viewer", set()),
        ("alpha_viewer_promoted", set()),
        ("alpha_newcomer", set()),
    ],
)
def test_only_writable_workspaces_are_offered(client: TestClient, world: _World, who: str, expected: set[str]) -> None:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    response = client.get(PATH)
    assert response.status_code == 200, response.text
    body = response.json()
    assert {row["id"] for row in body["data"]} == {str(world.workspaces[name]) for name in expected}
    assert body["count"] == len(expected)


def test_search_and_count_are_scoped_before_pagination(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    with db_session_factory() as db:
        db.add_all(
            [
                Workspace(name="Research A", organization_id=world.alpha),
                Workspace(name="Research B", organization_id=world.alpha),
                Workspace(name="Research foreign", organization_id=world.beta),
                Workspace(name="100%_literal", organization_id=world.alpha),
            ]
        )
        db.commit()
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_owner"])
    first = client.get(PATH, params={"search": "rEsEaRcH", "limit": 1}).json()
    second = client.get(PATH, params={"search": "rEsEaRcH", "limit": 1, "skip": 1}).json()
    assert first["count"] == second["count"] == 2
    assert [first["data"][0]["name"], second["data"][0]["name"]] == ["Research A", "Research B"]
    literal = client.get(PATH, params={"search": "%_"}).json()
    assert literal["count"] == 1
    assert literal["data"][0]["name"] == "100%_literal"


def test_suspended_workspace_membership_is_not_selectable(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    with db_session_factory() as db:
        db.execute(
            update(WorkspaceMember)
            .where(col(WorkspaceMember.user_id) == world.users["alpha_member"])
            .values(status="suspended")
        )
        db.commit()
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    response = client.get(PATH)
    assert response.status_code == 200
    assert response.json() == {"data": [], "count": 0}


def test_workspace_picker_requires_authentication(client: TestClient) -> None:
    assert client.get(PATH).status_code == 401


@pytest.mark.parametrize("params", [{"limit": 1001}, {"skip": -1}, {"search": "x" * 201}])
def test_workspace_picker_bounds_queries(client: TestClient, world: _World, params: dict[str, int | str]) -> None:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_owner"])
    assert client.get(PATH, params=params).status_code == 422
