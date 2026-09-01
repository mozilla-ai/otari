"""Deployment-wide management routes admit only a deployment operator.

`api/deps.verify_master_key` answers *authenticated*, not *authorized*: a
dashboard session clears it for any active identity. Two families of router
declare it, and only one of them re-checks the caller afterwards. The
tenant-scoped family (organizations, workspaces, org provider keys, `/v1/admin`)
resolves `CurrentIdentity` and asks a service whether that identity may act on
the organization, workspace or deployment named; the deployment-wide family does
not, so clearing the credential check *was* the whole authorization there.

That gap is otari-ai#1880: a member of one organization, invited to nothing else,
held master-key authority over the whole process. It mints a key into another
organization's workspace (which then resolves that organization's BYO provider
credential and bills it), rotates the deployment master key, freezes sign-ins,
and drives the three credential-test endpoints at a URL of its choosing.

Both halves are asserted here, and the second half is the point: the control
cases at the bottom are what catches a fix that swung too wide and took the
tenant-scoped routers with it.

What the gate does over HTTP is this file. Where it is declared, and that it
covers every route on every deployment-wide router rather than the roster of
paths probed below, is `tests/unit/test_operator_gate_declarations.py`.
"""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.models.entities import DashboardSession
from gateway.models.tenancy import Organization, OrganizationMember, User
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

# One probe per deployment-wide router family, each the cheapest request that
# reaches its handler. A GET where the router has one, so a member's refusal is
# never confused with a validation error; a write where every route on the
# router is one (`pricing` reads are deliberately open to any API key, so only
# its writes belong here).
_DEPLOYMENT_WIDE_PROBES: list[tuple[str, str]] = [
    ("GET", "/v1/keys"),
    ("GET", "/v1/users"),
    ("GET", "/v1/budgets"),
    ("GET", "/v1/scoped-budgets"),
    ("GET", "/v1/usage"),
    ("GET", "/v1/agent-telemetry/summary"),
    ("GET", "/v1/aliases"),
    ("GET", "/v1/routing/policies"),
    ("GET", "/v1/routing/status"),
    ("GET", "/v1/models/discoverable"),
    ("GET", "/v1/provider-credentials"),
    ("GET", "/v1/search-tools"),
    ("GET", "/v1/tool-settings"),
    ("GET", "/v1/settings"),
    ("GET", "/v1/settings/mail"),
    ("GET", "/v1/settings/maintenance-mode"),
    ("POST", "/v1/pricing"),
]

# The subset whose reach is worse than reading somebody else's rows: two that
# take the deployment away from its operator, and the three that make the
# gateway issue an outbound request to an address the caller supplies.
_ESCALATION_PROBES: list[tuple[str, str]] = [
    ("POST", "/v1/settings/master-key/rotate"),
    ("PATCH", "/v1/settings/maintenance-mode"),
    ("POST", "/v1/provider-credentials/test"),
    ("POST", "/v1/tool-settings/web_search/test"),
    ("POST", "/v1/settings/mail/test"),
]

# The data plane: a provider is called with somebody's credentials and a usage
# row is written against somebody's budget, both resolved through the
# deployment's default workspace for a caller with no key row. A cookie must not
# reach any of it. 401, not 403: with the cookie ignored the request simply
# carries no credential this plane recognizes.
_DATA_PLANE_PROBES: list[tuple[str, str]] = [
    ("POST", "/v1/embeddings"),
    ("POST", "/v1/moderations"),
    ("POST", "/v1/rerank"),
    ("POST", "/v1/search"),
    ("POST", "/v1/images/generations"),
    ("POST", "/v1/usage/external-events"),
    ("GET", "/v1/files"),
    ("GET", "/v1/batches"),
]

# The exception, and the reason the data-plane dependency was split rather than
# just tightened: these describe the deployment instead of acting on it, and the
# dashboard's Models and Pricing pages are built on them.
_CATALOG_PROBES: list[tuple[str, str]] = [
    ("GET", "/v1/models"),
    ("GET", "/v1/pricing"),
    ("GET", "/v1/tools"),
]

# Routers that resolve the caller and check their standing themselves. A plain
# member reaches these by design, so a 403 from any of them means the gate was
# applied to the wrong family. Only the routes a *member* may reach are listed:
# several of these routers hold owner/admin routes as well (organization
# guardrails and pricing are entirely owner/admin, and the provider-keys list
# joined them in otari-ai#1944), and their own 403 is indistinguishable here
# from the one this file is about.
_TENANT_SCOPED_PROBES: list[tuple[str, str]] = [
    ("GET", "/v1/organizations/me"),
    ("GET", "/v1/workspaces"),
    ("GET", "/v1/admin/access"),
]


def _provision(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Make one master-key request, which provisions the tenancy root."""
    assert client.get("/v1/organizations/me", headers=master_key_header).status_code == 200


def _default_organization_id(session_factory: Callable[[], Session]) -> uuid.UUID:
    session = session_factory()
    try:
        return session.query(Organization).filter(col(Organization.slug) == "default").one().id
    finally:
        session.close()


def _session_for(
    session_factory: Callable[[], Session],
    *,
    organization_id: uuid.UUID,
    email: str,
    role: str = "member",
    is_superuser: bool = False,
) -> str:
    """An identity holding a live dashboard session, and the cookie that names it."""
    session = session_factory()
    try:
        user = User(
            email=email,
            full_name=email.split("@")[0].title(),
            active_organization_id=organization_id,
            is_superuser=is_superuser,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        session.add(
            OrganizationMember(organization_id=organization_id, user_id=user.id, role=role, status="active")
        )
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
    finally:
        session.close()


def _call(client: TestClient, method: str, path: str) -> int:
    # An empty JSON body on every write: the routes that take one answer 422 for
    # it, which is past the gate and therefore still tells these cases apart.
    response = client.request(method, path, json={} if method != "GET" else None)
    return response.status_code


# =============================================================================
# The gap
# =============================================================================


@pytest.mark.parametrize(("method", "path"), _DEPLOYMENT_WIDE_PROBES + _ESCALATION_PROBES)
def test_a_plain_member_session_is_refused_by_every_deployment_wide_route(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    method: str,
    path: str,
) -> None:
    """otari-ai#1880: this is the whole of the cross-organization breach.

    403 rather than 404: unlike `/v1/admin`, these routes are no secret, and the
    dashboard has to tell "you may not" apart from "there is nothing here" to
    decide whether to keep the caller signed in.
    """
    _provision(client, master_key_header)
    organization_id = _default_organization_id(db_session_factory)
    token = _session_for(db_session_factory, organization_id=organization_id, email="ada@example.com")

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        assert _call(client, method, path) == 403
    finally:
        client.cookies.clear()


def test_an_organization_owner_is_still_not_a_deployment_operator(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Owning an organization is authority inside it, not over the process.

    The distinction is the reason the check is not "is this caller an admin
    somewhere": an owner who cleared this gate would reach every other
    organization on the deployment, which is what an owner most plausibly is.
    """
    _provision(client, master_key_header)
    organization_id = _default_organization_id(db_session_factory)
    token = _session_for(
        db_session_factory,
        organization_id=organization_id,
        email="owner@example.com",
        role="owner",
    )

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        refused = _call(client, "GET", "/v1/keys")
        # ...while the organization they do own answers as before.
        own = _call(client, "GET", "/v1/organizations/me")
        # Probed at owner rather than member because the provider-keys list is
        # organization-management-gated (otari-ai#1944), which is what took it
        # out of `_TENANT_SCOPED_PROBES`. Its own 403 would be
        # indistinguishable there from this file's; here it says the
        # deployment-operator gate is still off that router.
        keys = _call(client, "GET", "/v1/organizations/me/provider-keys")
    finally:
        client.cookies.clear()

    assert refused == 403
    assert own == 200
    assert keys == 200


# =============================================================================
# Who still gets through
# =============================================================================


@pytest.mark.parametrize(("method", "path"), _DEPLOYMENT_WIDE_PROBES)
def test_a_superuser_session_reaches_every_deployment_wide_route(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    method: str,
    path: str,
) -> None:
    """The dashboard is the operator's own console and must keep working."""
    _provision(client, master_key_header)
    organization_id = _default_organization_id(db_session_factory)
    token = _session_for(
        db_session_factory,
        organization_id=organization_id,
        email="operator@example.com",
        is_superuser=True,
    )

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        assert _call(client, method, path) != 403
    finally:
        client.cookies.clear()


@pytest.mark.parametrize(("method", "path"), _DEPLOYMENT_WIDE_PROBES)
def test_the_master_key_reaches_every_deployment_wide_route(
    client: TestClient,
    master_key_header: dict[str, str],
    method: str,
    path: str,
) -> None:
    """The header credential is the deployment's own, so it is never gated out."""
    _provision(client, master_key_header)
    response = client.request(method, path, headers=master_key_header, json={} if method != "GET" else None)

    assert response.status_code != 403, response.text


def test_the_master_key_still_lists_keys(client: TestClient, master_key_header: dict[str, str]) -> None:
    """One probe asserted at its real status, so `!= 403` above is not the only claim."""
    _provision(client, master_key_header)
    assert client.get("/v1/keys", headers=master_key_header).status_code == 200


def test_an_unauthenticated_request_is_still_401_and_not_403(client: TestClient) -> None:
    """The gate runs after the credential check, so no-credential keeps its status."""
    assert client.get("/v1/keys").status_code == 401


# =============================================================================
# Control cases: the tenant-scoped family is untouched
# =============================================================================


@pytest.mark.parametrize(("method", "path"), _TENANT_SCOPED_PROBES)
def test_a_plain_member_session_still_reaches_the_tenant_scoped_routes(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    method: str,
    path: str,
) -> None:
    """These check the caller's standing themselves, so the gate must stay off them."""
    _provision(client, master_key_header)
    organization_id = _default_organization_id(db_session_factory)
    token = _session_for(db_session_factory, organization_id=organization_id, email="grace@example.com")

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        assert _call(client, method, path) == 200
    finally:
        client.cookies.clear()


def test_the_admin_router_keeps_its_own_404_rather_than_the_gate_403(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    test_config: GatewayConfig,
) -> None:
    """`/v1/admin` hides itself from a non-operator on purpose, and still does.

    Its refusal is a 404 so the surface does not confirm it exists, and
    `GET /access` answers 200 either way so the dashboard has something to gate
    its navigation on. Gating that router here would have replaced both.
    """
    _provision(client, master_key_header)
    organization_id = _default_organization_id(db_session_factory)
    token = _session_for(db_session_factory, organization_id=organization_id, email="alan@example.com")

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        listed = client.get("/v1/admin/users")
        access = client.get("/v1/admin/access")
    finally:
        client.cookies.clear()

    assert listed.status_code == 404, listed.text
    assert access.status_code == 200, access.text
    assert access.json() == {"granted": False}


# =============================================================================
# The data plane, which a cookie may not reach at all
# =============================================================================


@pytest.mark.parametrize(("method", "path"), _DATA_PLANE_PROBES)
def test_a_session_cookie_does_not_authenticate_the_data_plane(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    method: str,
    path: str,
) -> None:
    """The second half of otari-ai#1880, and the half that needed no key at all.

    ``verify_api_key_or_master_key`` used to return "this is the master key" for
    any session, and ``is_master_key`` is what makes the request resolve its
    workspace, its organization's provider credentials and its budget through the
    deployment *default*. So a signed-in member of an unrelated organization
    could spend the default organization's BYO credential on a completion, and
    file usage rows into a tenant they belong to nothing in, without ever minting
    a key. Gating the management plane alone would have left this open.
    """
    _provision(client, master_key_header)
    organization_id = _default_organization_id(db_session_factory)
    token = _session_for(db_session_factory, organization_id=organization_id, email="mallory@example.com")

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        assert _call(client, method, path) == 401
    finally:
        client.cookies.clear()


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/v1/chat/completions",
            {"model": "openai:gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        ),
        (
            "/v1/messages/count_tokens",
            {"model": "openai:gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        ),
    ],
)
def test_a_session_cookie_does_not_authenticate_a_completion(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    path: str,
    body: dict[str, object],
) -> None:
    """The two routes that authenticate inside the handler rather than by dependency.

    Given a body of their own, because both parse the request model before they
    reach the credential check and would otherwise answer 422 without ever
    getting there. This is the exact request the review reproduced: it used to
    clear auth, resolve the default workspace, and dial the provider on that
    organization's stored credential.
    """
    _provision(client, master_key_header)
    organization_id = _default_organization_id(db_session_factory)
    token = _session_for(db_session_factory, organization_id=organization_id, email="eve@example.com")

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        response = client.post(path, json=body)
    finally:
        client.cookies.clear()

    assert response.status_code == 401, response.text


def test_even_a_superuser_session_does_not_authenticate_the_data_plane(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Not an authority check: this plane takes a key or the master key, full stop.

    Worth pinning separately from the case above, because the obvious wrong fix
    is to reuse the operator gate here. That would keep the breach open for the
    one identity that can also read every organization's credentials.
    """
    _provision(client, master_key_header)
    organization_id = _default_organization_id(db_session_factory)
    token = _session_for(
        db_session_factory,
        organization_id=organization_id,
        email="root@example.com",
        is_superuser=True,
    )

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        assert _call(client, "POST", "/v1/embeddings") == 401
    finally:
        client.cookies.clear()


@pytest.mark.parametrize(("method", "path"), _CATALOG_PROBES)
def test_a_plain_member_session_still_reads_the_catalog(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    method: str,
    path: str,
) -> None:
    """These call no provider, write nothing and bill nothing, so a session reads them."""
    _provision(client, master_key_header)
    organization_id = _default_organization_id(db_session_factory)
    token = _session_for(db_session_factory, organization_id=organization_id, email="ada@example.com")

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        assert _call(client, method, path) == 200
    finally:
        client.cookies.clear()


def test_an_api_key_still_reaches_the_data_plane(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The control for the refusals above: nothing changed for a real credential."""
    _provision(client, master_key_header)
    created = client.post("/v1/keys", json={"key_name": "data-plane"}, headers=master_key_header)
    assert created.status_code == 200, created.text
    secret = created.json()["key"]

    listed = client.get("/v1/files", headers={"Otari-Key": secret})

    assert listed.status_code == 200, listed.text
