"""A deployment operator reads their active organization's rows, not every tenant's.

The operator routes behind the Spend, Usage, Activity, Routing, Traces and
Overview pages were deployment-wide queries, so an operator signed in to one
organization read every other organization's budgets, ceilings, usage, policies,
aliases, traces and key count. ``/api/v1/keys`` and ``/api/v1/users`` had already
settled that question (otari#817): a session acts in its active organization and
a header master key in the default one. This suite holds the other routes to the
same rule.

Rows that belong to no organization stay in reach: the deployment's own budgets
and the config-file aliases and policies.

Every assertion names the row that must be absent, for the reason
``test_users_organization_scope`` gives: a count matches by accident.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT
from gateway.models.api_keys import APIKey
from gateway.models.budgets import Budget, ScopedBudget
from gateway.models.providers import ModelAlias
from gateway.models.routing import RoutingPolicy
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, Workspace
from gateway.models.tenancy import User as TenancyUser
from gateway.models.traces import Trace
from gateway.models.usage import UsageLog
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

_NOW = datetime.now(UTC)
_WINDOW = {
    "start_date": (_NOW - timedelta(days=1)).isoformat(),
    "end_date": (_NOW + timedelta(days=1)).isoformat(),
}


@dataclass
class _World:
    workspaces: dict[str, uuid.UUID] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)
    ceilings: dict[str, str] = field(default_factory=dict)
    usage: dict[str, str] = field(default_factory=dict)


def _identity(session: Session, *, email: str, organization_id: uuid.UUID, role: str, operator: bool) -> str:
    """A live dashboard session for someone on one organization's roster."""
    person = TenancyUser(
        email=email,
        full_name=email.split("@")[0].title(),
        active_organization_id=organization_id,
        is_superuser=operator,
    )
    session.add(person)
    session.commit()
    session.refresh(person)
    session.add(OrganizationMember(organization_id=organization_id, user_id=person.id, role=role, status="active"))
    token = f"otari-sess-{email}"
    session.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=person.id,
            created_at=_NOW,
            expires_at=_NOW + timedelta(hours=12),
        )
    )
    session.commit()
    return token


def _tenant_rows(session: Session, world: _World, name: str, organization_id: uuid.UUID | None) -> None:
    """A key, a usage row, an imported usage row, a policy, an alias and a trace in one workspace."""
    workspace_id = world.workspaces[name]
    key_id = f"key-{name}"
    session.add(APIKey(id=key_id, workspace_id=workspace_id, key_hash=f"hash-{name}", key_prefix=f"gw-{name}"))
    session.flush()
    for kind, source, counts in (("served", "gateway", True), ("imported", "claude_code", False)):
        row_id = f"usage-{name}-{kind}"
        world.usage[f"{name}-{kind}"] = row_id
        session.add(
            UsageLog(
                id=row_id,
                workspace_id=workspace_id,
                api_key_id=key_id,
                model=f"model-{name}",
                provider="openai",
                endpoint="/v1/chat/completions",
                source=source,
                counts_toward_budget=counts,
                status="success",
                total_tokens=10,
                cost=Decimal("0.01"),
                timestamp=_NOW,
            )
        )
    session.add(
        RoutingPolicy(
            name=f"{name}-policy", spec={"select": [{"default": "openai:gpt-4o-mini"}]}, workspace_id=workspace_id
        )
    )
    session.add(ModelAlias(name=f"{name}-alias", target="openai:gpt-4o-mini", workspace_id=workspace_id))
    session.add(
        Trace(
            workspace_id=workspace_id,
            trace_id=f"trace-{name}",
            session_source="none",
            started_at=_NOW,
            last_activity_at=_NOW,
        )
    )
    if organization_id is not None:
        session.add(
            Budget(budget_id=f"{name}-budget", name=name, max_budget=Decimal(10), organization_id=organization_id)
        )
        session.flush()
        world.ceilings[f"{name}-workspace"] = f"ceiling-{name}-workspace"
        world.ceilings[f"{name}-key"] = f"ceiling-{name}-key"
        world.ceilings[f"{name}-organization"] = f"ceiling-{name}-organization"
        session.add_all(
            [
                ScopedBudget(
                    id=f"ceiling-{name}-workspace",
                    scope_type="workspace",
                    scope_id=str(workspace_id),
                    budget_id=f"{name}-budget",
                ),
                ScopedBudget(
                    id=f"ceiling-{name}-key", scope_type="api_token", scope_id=key_id, budget_id=f"{name}-budget"
                ),
                ScopedBudget(
                    id=f"ceiling-{name}-organization",
                    scope_type="organization",
                    scope_id=str(organization_id),
                    budget_id=f"{name}-budget",
                ),
            ]
        )
    session.commit()


@pytest.fixture
def world(client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]) -> _World:
    """The default organization and two tenants, each with one of every row these routes read."""
    me = client.get(f"{API_ROOT}/organizations/me", headers=master_key_header)
    assert me.status_code == status.HTTP_200_OK, me.text
    default_organization = uuid.UUID(me.json()["organization"]["id"])

    session = db_session_factory()
    try:
        alpha = Organization(name="Alpha", slug="alpha-op-scope")
        beta = Organization(name="Beta", slug="beta-op-scope")
        session.add_all([alpha, beta])
        session.commit()
        alpha_one = Workspace(name="Alpha one", organization_id=alpha.id)
        beta_one = Workspace(name="Beta one", organization_id=beta.id)
        session.add_all([alpha_one, beta_one])
        session.commit()

        built = _World()
        built.workspaces = {
            "default": session.execute(
                select(col(Workspace.id)).where(col(Workspace.organization_id) == default_organization).limit(1)
            ).scalar_one(),
            "alpha": alpha_one.id,
            "beta": beta_one.id,
        }
        session.add(Budget(budget_id="gateway-budget", name="Gateway", max_budget=Decimal(100)))
        session.commit()
        _tenant_rows(session, built, "default", None)
        _tenant_rows(session, built, "alpha", alpha.id)
        _tenant_rows(session, built, "beta", beta.id)

        built.sessions = {
            "alpha_operator": _identity(
                session, email="op@alpha.test", organization_id=alpha.id, role="owner", operator=True
            ),
            "alpha_admin": _identity(
                session, email="admin@alpha.test", organization_id=alpha.id, role="admin", operator=False
            ),
        }
        return built
    finally:
        session.close()


def _call(client: TestClient, world: _World, who: str, method: str, path: str, **kwargs: Any) -> tuple[int, Any]:
    """Call as ``who``: a session name from the world, or ``master`` for the header master key."""
    headers = kwargs.pop("headers", {})
    if who != "master":
        client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        response = client.request(method, f"{API_ROOT}{path}", headers=headers, **kwargs)
    finally:
        client.cookies.clear()
    body = response.json() if response.content else None
    return response.status_code, body


def _get(client: TestClient, world: _World, who: str, path: str, **kwargs: Any) -> Any:
    code, body = _call(client, world, who, "GET", path, **kwargs)
    assert code == status.HTTP_200_OK, body
    return body


@pytest.fixture
def as_master(master_key_header: dict[str, str]) -> dict[str, Any]:
    return {"headers": master_key_header}


# =============================================================================
# Budgets
# =============================================================================


def test_an_operator_lists_the_deployments_budgets_and_their_own_organizations(
    client: TestClient, world: _World
) -> None:
    listed = {row["budget_id"] for row in _get(client, world, "alpha_operator", "/budgets?limit=1000")}
    assert "gateway-budget" in listed, "a budget the deployment owns belongs to no tenant"
    assert "alpha-budget" in listed
    assert "beta-budget" not in listed


@pytest.mark.parametrize(
    ("method", "suffix", "body"),
    [("GET", "", None), ("GET", "/reset-logs", None), ("PATCH", "", {"name": "renamed"})],
)
def test_another_organizations_budget_is_not_found(
    client: TestClient, world: _World, method: str, suffix: str, body: dict[str, str] | None
) -> None:
    kwargs = {"json": body} if body is not None else {}
    code, _ = _call(client, world, "alpha_operator", method, f"/budgets/beta-budget{suffix}", **kwargs)
    assert code == status.HTTP_404_NOT_FOUND


def test_the_same_budget_routes_serve_the_operators_own_and_the_deployments(client: TestClient, world: _World) -> None:
    """The control: 404 has to mean out of scope, not broken."""
    for budget_id in ("alpha-budget", "gateway-budget"):
        assert _get(client, world, "alpha_operator", f"/budgets/{budget_id}")["budget_id"] == budget_id
        assert _get(client, world, "alpha_operator", f"/budgets/{budget_id}/reset-logs") == []


# =============================================================================
# Spend ceilings
# =============================================================================


def test_an_operator_lists_only_their_organizations_ceilings(client: TestClient, world: _World) -> None:
    listed = {row["id"] for row in _get(client, world, "alpha_operator", "/scoped-budgets?limit=1000")}
    assert {world.ceilings["alpha-workspace"], world.ceilings["alpha-key"], world.ceilings["alpha-organization"]} <= (
        listed
    ), "every scope kind inside the organization is matched"
    assert (
        not {
            world.ceilings["beta-workspace"],
            world.ceilings["beta-key"],
            world.ceilings["beta-organization"],
        }
        & listed
    )


@pytest.mark.parametrize(
    ("method", "body"),
    [("GET", None), ("PATCH", {"name": "renamed"}), ("DELETE", None)],
)
def test_another_organizations_ceiling_is_not_found(
    client: TestClient, world: _World, method: str, body: dict[str, str] | None
) -> None:
    kwargs = {"json": body} if body is not None else {}
    code, _ = _call(
        client, world, "alpha_operator", method, f"/scoped-budgets/{world.ceilings['beta-workspace']}", **kwargs
    )
    assert code == status.HTTP_404_NOT_FOUND
    assert _get(client, world, "alpha_operator", f"/scoped-budgets/{world.ceilings['alpha-workspace']}")


# =============================================================================
# Usage
# =============================================================================


def test_an_operator_reads_only_their_organizations_usage(client: TestClient, world: _World) -> None:
    rows = {row["id"] for row in _get(client, world, "alpha_operator", "/usage", params={"limit": 1000})}
    assert world.usage["alpha-served"] in rows
    assert world.usage["beta-served"] not in rows
    assert world.usage["default-served"] not in rows

    count = _get(client, world, "alpha_operator", "/usage/count")["total"]
    assert count == 2, "the paginator's total is scoped like the page"

    summary = _get(client, world, "alpha_operator", "/usage/summary", params={**_WINDOW, "dimensions": "model"})
    assert {row["key"] for row in summary["by_model"]} == {"model-alpha"}

    series = _get(client, world, "alpha_operator", "/usage/series", params={**_WINDOW, "group_by": "model"})
    assert {row["key"] for row in series["groups"]} == {"model-alpha"}


def test_a_workspace_filter_cannot_reach_another_organization(client: TestClient, world: _World) -> None:
    params = {"workspace_id": str(world.workspaces["beta"])}
    assert _get(client, world, "alpha_operator", "/usage", params=params) == []
    assert _get(client, world, "alpha_operator", "/usage/count", params=params)["total"] == 0


def test_a_bulk_delete_cannot_reach_another_organizations_rows(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    """The count an operator confirms is scoped, so the mutation it sizes has to be too."""
    code, body = _call(
        client,
        world,
        "alpha_operator",
        "DELETE",
        "/usage",
        json={"ids": [world.usage["beta-imported"], world.usage["alpha-imported"]]},
    )
    assert code == status.HTTP_200_OK, body
    assert body == {"deleted": 1}
    with db_session_factory() as session:
        assert session.get(UsageLog, world.usage["beta-imported"]) is not None
        assert session.get(UsageLog, world.usage["alpha-imported"]) is None


def test_a_bulk_reprice_cannot_reach_another_organizations_rows(client: TestClient, world: _World) -> None:
    code, body = _call(
        client,
        world,
        "alpha_operator",
        "POST",
        "/usage/set-price",
        json={
            "by_filter": True,
            "input_price_per_million": 1,
            "output_price_per_million": 1,
        },
    )
    assert code == status.HTTP_200_OK, body
    assert body["matched"] == 1, "alpha's imported row only"


# =============================================================================
# Routing policies and aliases
# =============================================================================


def test_an_operator_lists_only_their_organizations_policies_and_aliases(client: TestClient, world: _World) -> None:
    policies = {row["name"] for row in _get(client, world, "alpha_operator", "/routing/policies")}
    assert "alpha-policy" in policies
    assert "beta-policy" not in policies
    assert "default-policy" not in policies

    aliases = {row["name"] for row in _get(client, world, "alpha_operator", "/aliases")}
    assert "alpha-alias" in aliases
    assert "beta-alias" not in aliases
    assert "default-alias" not in aliases


def test_a_policy_or_alias_workspace_filter_cannot_reach_another_organization(
    client: TestClient, world: _World
) -> None:
    params = {"workspace_id": str(world.workspaces["beta"])}
    assert not [
        row for row in _get(client, world, "alpha_operator", "/routing/policies", params=params) if row["workspace_id"]
    ]
    assert not [row for row in _get(client, world, "alpha_operator", "/aliases", params=params) if row["workspace_id"]]


# =============================================================================
# Traces
# =============================================================================


def test_an_operator_reads_only_their_organizations_traces(client: TestClient, world: _World) -> None:
    listed = {row["trace_id"] for row in _get(client, world, "alpha_operator", "/traces")["items"]}
    assert listed == {"trace-alpha"}
    assert _get(client, world, "alpha_operator", "/traces/count")["count"] == 1

    code, _ = _call(client, world, "alpha_operator", "GET", "/traces/trace-beta")
    assert code == status.HTTP_404_NOT_FOUND


# =============================================================================
# Overview
# =============================================================================


def test_an_operators_key_count_is_their_organizations(client: TestClient, world: _World) -> None:
    """Counted the way an admin of the same organization counts it."""
    operator = _get(client, world, "alpha_operator", "/overview")
    admin = _get(client, world, "alpha_admin", "/overview")
    assert operator["active_keys"] == admin["active_keys"] == 1
    assert operator["budgets"] is not None, "the deployment strip is still the operator's"


# =============================================================================
# The header master key, and callers who do not operate the deployment
# =============================================================================


def test_the_header_master_key_acts_in_the_default_organization(
    client: TestClient, world: _World, as_master: dict[str, Any], db_session_factory: Callable[[], Session]
) -> None:
    budgets = {row["budget_id"] for row in _get(client, world, "master", "/budgets?limit=1000", **as_master)}
    assert "gateway-budget" in budgets
    assert not {"alpha-budget", "beta-budget"} & budgets

    ceilings = {row["id"] for row in _get(client, world, "master", "/scoped-budgets?limit=1000", **as_master)}
    assert not {value for value in world.ceilings.values()} & ceilings

    usage = {row["id"] for row in _get(client, world, "master", "/usage", params={"limit": 1000}, **as_master)}
    assert world.usage["default-served"] in usage
    assert not {world.usage["alpha-served"], world.usage["beta-served"]} & usage

    policies = {row["name"] for row in _get(client, world, "master", "/routing/policies", **as_master)}
    assert "default-policy" in policies
    assert not {"alpha-policy", "beta-policy"} & policies

    aliases = {row["name"] for row in _get(client, world, "master", "/aliases", **as_master)}
    assert "default-alias" in aliases
    assert not {"alpha-alias", "beta-alias"} & aliases

    traces = {row["trace_id"] for row in _get(client, world, "master", "/traces", **as_master)["items"]}
    assert traces == {"trace-default"}

    with db_session_factory() as session:
        default_keys = session.scalar(
            select(func.count())
            .select_from(APIKey)
            .join(Workspace, col(Workspace.id) == APIKey.workspace_id)
            .where(
                APIKey.is_active.is_(True),
                col(Workspace.organization_id)
                == select(col(Workspace.organization_id))
                .where(col(Workspace.id) == world.workspaces["default"])
                .scalar_subquery(),
            )
        )
    assert _get(client, world, "master", "/overview", **as_master)["active_keys"] == default_keys


def test_an_admin_who_does_not_operate_the_deployment_is_still_refused(client: TestClient, world: _World) -> None:
    """The role gates are unchanged: scoping the rows is not a looser gate."""
    for path in ("/budgets", "/scoped-budgets", "/usage", "/routing/policies", "/aliases", "/traces"):
        code, _ = _call(client, world, "alpha_admin", "GET", path)
        assert code == status.HTTP_403_FORBIDDEN, path
