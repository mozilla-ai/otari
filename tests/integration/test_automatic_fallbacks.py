"""``GET /api/v1/organizations/me/automatic-fallbacks``: the models an organization used, and the fallback each gets.

Two organizations with traffic in both. Alpha offers Claude on its own Fireworks
key as well, so its view orders Claude's offerings differently from the
deployment's. Each assertion names what must be absent as well as what is there.
"""

import uuid
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.models.usage import UsageLog
from gateway.services import catalog as selectors
from gateway.services import selector_index_service
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

from .conftest import build_test_client

_PATH = f"{API_ROOT}/organizations/me/automatic-fallbacks"

_CLAUDE = "anthropic/claude-x"
_GEMINI = "google/gemini-x"
_ANTHROPIC = "anthropic:claude-x-1"
_BEDROCK = "bedrock:anthropic.claude-x-1"
_FIREWORKS = "fireworks:accounts/fireworks/models/claude-x"
_GEMINI_OFFERING = "gemini:gemini-x"

_DEPLOYMENT_ROWS = [
    selectors.OfferingRow(_ANTHROPIC, "anthropic", "anthropic", 3.0),
    selectors.OfferingRow(_BEDROCK, "bedrock", "bedrock", 2.0),
    selectors.OfferingRow(_GEMINI_OFFERING, "gemini", "gemini", 1.0),
]
_IDENTITIES: dict[str, tuple[str, tuple[str, ...]]] = {
    _CLAUDE: ("claudex", (_ANTHROPIC, _BEDROCK)),
    _GEMINI: ("geminix", (_GEMINI_OFFERING,)),
}
_NAMES = {_CLAUDE: "Claude X", _GEMINI: "Gemini X"}


@dataclass
class _World:
    workspaces: dict[str, uuid.UUID] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)


def _identity(
    session: Session,
    *,
    email: str,
    organization_id: uuid.UUID,
    role: str,
    workspace_ids: tuple[uuid.UUID, ...] = (),
) -> str:
    user = User(email=email, full_name=email.split("@")[0].title(), active_organization_id=organization_id)
    session.add(user)
    session.commit()
    session.refresh(user)
    session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role=role, status="active"))
    for workspace_id in workspace_ids:
        session.add(WorkspaceMember(workspace_id=workspace_id, user_id=user.id, role="member", status="active"))
    token = f"otari-sess-{email}"
    session.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=user.id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=12),
        )
    )
    session.commit()
    return token


def _row(
    workspace_id: uuid.UUID,
    selector: str,
    *,
    sent: str | None,
    status_: str = "success",
    attempt: int | None = None,
    source: str = "gateway",
    age: timedelta = timedelta(hours=1),
) -> UsageLog:
    provider, _, model = selector.partition(":")
    return UsageLog(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        model=model,
        provider=provider,
        endpoint="/v1/chat/completions",
        source=source,
        status=status_,
        policy_name=sent,
        attempt_position=attempt,
        timestamp=datetime.now(UTC) - age,
    )


@pytest.fixture
def fallbacks_client(
    test_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    """A client whose selector index is the one the test sets, not one the refresher rebuilds over it."""

    async def _keep(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(selector_index_service, "rebuild_selector_index", _keep)
    yield from build_test_client(test_config)


@pytest.fixture
def world(
    fallbacks_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> _World:
    assert (
        fallbacks_client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code
        == status.HTTP_200_OK
    )
    session = db_session_factory()
    try:
        alpha = Organization(name="Alpha", slug="alpha")
        beta = Organization(name="Beta", slug="beta")
        session.add_all([alpha, beta])
        session.commit()
        alpha_one = Workspace(name="Alpha one", organization_id=alpha.id)
        alpha_two = Workspace(name="Alpha two", organization_id=alpha.id)
        beta_one = Workspace(name="Beta one", organization_id=beta.id)
        session.add_all([alpha_one, alpha_two, beta_one])
        session.commit()

        session.add_all(
            [
                # Two requests by catalog ID, served first time.
                _row(alpha_one.id, _FIREWORKS, sent=_CLAUDE, attempt=1),
                _row(alpha_one.id, _FIREWORKS, sent=_CLAUDE, attempt=1),
                # One that fell back: the absorbed attempt is not a request of its own.
                _row(alpha_one.id, _FIREWORKS, sent=_CLAUDE, status_="absorbed", attempt=1),
                _row(alpha_one.id, _BEDROCK, sent=_CLAUDE, attempt=2),
                # One that named Bedrock and so had no fallback.
                _row(alpha_one.id, _BEDROCK, sent=None),
                # Outside the window, and imported usage: neither counts.
                _row(alpha_one.id, _ANTHROPIC, sent=_CLAUDE, age=timedelta(days=40)),
                _row(alpha_one.id, _ANTHROPIC, sent=None, source="claude_code"),
                # The sibling workspace a member does not belong to.
                _row(alpha_two.id, _GEMINI_OFFERING, sent=_GEMINI),
                # Another organization's traffic.
                *(_row(beta_one.id, _ANTHROPIC, sent=_CLAUDE, attempt=1) for _ in range(5)),
            ]
        )
        session.commit()

        built = _World(workspaces={"alpha_one": alpha_one.id, "alpha_two": alpha_two.id, "beta_one": beta_one.id})
        built.sessions = {
            "alpha_owner": _identity(session, email="owner@alpha.test", organization_id=alpha.id, role="owner"),
            "alpha_member": _identity(
                session,
                email="member@alpha.test",
                organization_id=alpha.id,
                role="member",
                workspace_ids=(alpha_one.id,),
            ),
            "beta_owner": _identity(session, email="owner@beta.test", organization_id=beta.id, role="owner"),
        }

        own = [selectors.OfferingRow(_FIREWORKS, "fireworks", "fireworks", 4.0)]
        alpha_identities = {**_IDENTITIES, _CLAUDE: ("claudex", (_ANTHROPIC, _BEDROCK, _FIREWORKS))}
        selectors.set_selector_index(
            selectors.build_selector_index(
                _DEPLOYMENT_ROWS,
                _IDENTITIES,
                organizations={
                    alpha.id: selectors.build_organization_selectors(_DEPLOYMENT_ROWS, own, alpha_identities, _NAMES)
                },
                names=_NAMES,
            )
        )
        return built
    finally:
        session.close()


def _read(client: TestClient, world: _World, who: str) -> dict[str, dict[str, Any]]:
    response = client.get(_PATH, cookies={SESSION_COOKIE_NAME: world.sessions[who]})
    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert body["window_days"] == 30
    return {model["id"]: model for model in body["models"]}


def test_an_owner_reads_their_organizations_fallbacks_in_its_own_order(
    fallbacks_client: TestClient, world: _World
) -> None:
    models = _read(fallbacks_client, world, "alpha_owner")

    assert list(models) == [_CLAUDE, _GEMINI]
    claude = models[_CLAUDE]
    assert claude["name"] == "Claude X"
    # Alpha's own key comes first for its callers, then the vendor's own provider, then the cheapest.
    assert [offering["selector"] for offering in claude["offerings"]] == [_FIREWORKS, _ANTHROPIC, _BEDROCK]
    assert [offering["provider"] for offering in claude["offerings"]] == ["fireworks", "anthropic", "bedrock"]
    assert (claude["requests"], claude["fell_back"], claude["pinned"]) == (4, 1, 1)
    assert claude["pinned_provider"] == "bedrock"
    tools = {offering["provider"]: offering["own_tools"] for offering in claude["offerings"]}
    assert "web_search" in tools["anthropic"]
    assert tools["bedrock"] == []

    gemini = models[_GEMINI]
    assert [offering["selector"] for offering in gemini["offerings"]] == [_GEMINI_OFFERING]
    assert (gemini["requests"], gemini["fell_back"], gemini["pinned"]) == (1, 0, 0)


def test_a_member_reads_only_the_workspaces_they_belong_to(fallbacks_client: TestClient, world: _World) -> None:
    models = _read(fallbacks_client, world, "alpha_member")
    assert list(models) == [_CLAUDE]
    assert _GEMINI not in models


def test_another_organization_sees_its_own_traffic_and_order(fallbacks_client: TestClient, world: _World) -> None:
    models = _read(fallbacks_client, world, "beta_owner")
    assert list(models) == [_CLAUDE]
    claude = models[_CLAUDE]
    assert (claude["requests"], claude["fell_back"], claude["pinned"]) == (5, 0, 0)
    # The deployment's order: Alpha's key is nowhere in it.
    assert [offering["selector"] for offering in claude["offerings"]] == [_ANTHROPIC, _BEDROCK]


def test_the_read_needs_a_signed_in_caller(fallbacks_client: TestClient, world: _World) -> None:
    assert fallbacks_client.get(_PATH).status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN)
