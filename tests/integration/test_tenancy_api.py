"""The organization and workspace endpoints, end to end.

Three things are under test here, in rough order of how badly they would hurt if
they broke: that a master-key-authenticated operator is resolved to a real
identity with a default organization and workspace on first use (otari-ai#1716
option A), that the response shapes the ported dashboard pages will be generated
from stay as they are, and that the authorization and tenant-scoping rules the
platform enforced survived the port.
"""

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import RuntimeSetting
from gateway.models.tenancy import Organization, OrganizationMember, User
from gateway.services.tenancy.provisioning_service import (
    BOOTSTRAP_IDENTITY_KEY,
    DEFAULT_ORGANIZATION_NAME,
    DEFAULT_WORKSPACE_NAME,
)


def _context(client: TestClient, headers: dict[str, str]) -> dict[str, Any]:
    response = client.get("/v1/organizations/me", headers=headers)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def _add_identity(
    session_factory: Callable[[], Session],
    *,
    organization_id: uuid.UUID,
    full_name: str,
    email: str,
    role: str = "member",
    status: str = "active",
) -> uuid.UUID:
    """Insert a second identity with an organization membership.

    Written directly rather than through the API because this slice has no
    identity-creation endpoint: adding members arrives with the invitation flow.
    """
    session = session_factory()
    try:
        user = User(email=email, full_name=full_name, active_organization_id=organization_id)
        session.add(user)
        session.flush()
        session.add(
            OrganizationMember(
                organization_id=organization_id,
                user_id=user.id,
                role=role,
                status=status,
            )
        )
        session.commit()
        return user.id
    finally:
        session.close()


# =============================================================================
# First boot
# =============================================================================


def test_first_request_provisions_the_default_organization(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The master key resolves to an owner identity in a default organization."""
    context = _context(client, master_key_header)

    assert context["organization"]["name"] == DEFAULT_ORGANIZATION_NAME
    assert context["organization"]["slug"] == "default"
    assert context["role"] == "owner"
    assert context["status"] == "active"
    # A standalone deployment is its own gateway, so its own provider keys are
    # always available to it.
    assert context["byo_provider_keys_allowed"] is True


def test_first_boot_also_provisions_a_default_workspace(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """An organization with no workspace has no usable surface, so one is created."""
    response = client.get("/v1/workspaces", headers=master_key_header)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["data"][0]["name"] == DEFAULT_WORKSPACE_NAME


def test_provisioning_is_idempotent(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Repeated requests resolve the same identity instead of provisioning again."""
    first = _context(client, master_key_header)
    second = _context(client, master_key_header)

    assert first["organization_member_id"] == second["organization_member_id"]
    assert db_session.query(Organization).count() == 1
    assert db_session.query(User).count() == 1


def test_a_marker_that_no_longer_resolves_re_provisions(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """An unresolvable marker must self-heal, not wedge the deployment.

    The marker naming the operator identity is the only thing that makes
    provisioning idempotent, so a value that no longer resolves has to be
    replaced rather than inserted beside itself, which would collide on the
    primary key and leave every later request answering 500 forever.
    """
    _context(client, master_key_header)
    session = db_session_factory()
    try:
        marker = session.get(RuntimeSetting, BOOTSTRAP_IDENTITY_KEY)
        assert marker is not None
        marker.value = "not-a-uuid"
        session.commit()
    finally:
        session.close()

    recovered = client.get("/v1/organizations/me", headers=master_key_header)

    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["role"] == "owner"


def test_the_dashboard_session_cookie_also_authenticates(
    client: TestClient,
    test_config: Any,
    master_key_header: dict[str, str],
) -> None:
    """The tenancy routes accept what the rest of the management API accepts."""
    signed_in = client.post("/v1/auth/session", json={"master_key": test_config.master_key})
    assert signed_in.status_code == 200, signed_in.text

    response = client.get("/v1/organizations/me")

    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "path",
    ["/v1/organizations/me", "/v1/organizations/me/members", "/v1/workspaces"],
)
def test_tenancy_routes_require_a_credential(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 401


def test_an_api_key_is_not_an_operator(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """A request-plane credential must not reach the control plane's tenancy surface."""
    response = client.get("/v1/organizations/me", headers=api_key_header)

    assert response.status_code == 401


# =============================================================================
# Organizations
# =============================================================================


def test_rename_the_active_organization(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.patch("/v1/organizations/me", json={"name": "Acme"}, headers=master_key_header)

    assert response.status_code == 200, response.text
    assert response.json()["organization"]["name"] == "Acme"
    assert _context(client, master_key_header)["organization"]["name"] == "Acme"


def test_creating_an_organization_switches_into_it(client: TestClient, master_key_header: dict[str, str]) -> None:
    created = client.post("/v1/organizations/me", json={"name": "Acme Labs"}, headers=master_key_header)

    assert created.status_code == 201, created.text
    body = created.json()
    assert body["organization"]["name"] == "Acme Labs"
    assert body["organization"]["slug"] == "acme-labs"
    assert body["role"] == "owner"
    assert _context(client, master_key_header)["organization"]["id"] == body["organization"]["id"]


def test_a_new_organization_gets_its_own_default_workspace(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    client.post("/v1/organizations/me", json={"name": "Acme Labs"}, headers=master_key_header)

    workspaces = client.get("/v1/workspaces", headers=master_key_header).json()

    assert [workspace["name"] for workspace in workspaces["data"]] == [DEFAULT_WORKSPACE_NAME]


def test_slugs_are_made_unique(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Two organizations may share a name; their slugs address them, so those differ."""
    first = client.post("/v1/organizations/me", json={"name": "Acme"}, headers=master_key_header).json()
    second = client.post("/v1/organizations/me", json={"name": "Acme"}, headers=master_key_header).json()

    assert first["organization"]["slug"] == "acme"
    assert second["organization"]["slug"] == "acme-2"


def test_memberships_list_every_organization_by_name(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    client.post("/v1/organizations/me", json={"name": "Zebra"}, headers=master_key_header)
    client.post("/v1/organizations/me", json={"name": "Acme"}, headers=master_key_header)

    body = client.get("/v1/organizations/me/memberships", headers=master_key_header).json()

    assert body["count"] == 3
    assert [context["organization"]["name"] for context in body["data"]] == ["Acme", DEFAULT_ORGANIZATION_NAME, "Zebra"]


def test_switching_organizations(client: TestClient, master_key_header: dict[str, str]) -> None:
    default_organization_id = _context(client, master_key_header)["organization"]["id"]
    client.post("/v1/organizations/me", json={"name": "Acme"}, headers=master_key_header)

    switched = client.post(
        "/v1/organizations/me/switch",
        json={"organization_id": default_organization_id},
        headers=master_key_header,
    )

    assert switched.status_code == 200, switched.text
    assert switched.json()["organization"]["id"] == default_organization_id
    assert _context(client, master_key_header)["organization"]["id"] == default_organization_id


def test_switching_to_an_unknown_organization_is_not_found(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    response = client.post(
        "/v1/organizations/me/switch",
        json={"organization_id": str(uuid.uuid4())},
        headers=master_key_header,
    )

    assert response.status_code == 404


def test_an_organization_you_do_not_belong_to_is_indistinguishable_from_a_missing_one(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Switch is the one endpoint taking a tenant id from the request.

    Answering 403 for "exists but not yours" and 404 for "does not exist" would
    make the pair an existence oracle over other tenants' organizations.
    """
    _context(client, master_key_header)
    session = db_session_factory()
    try:
        stranger = Organization(name="Stranger", slug="stranger")
        session.add(stranger)
        session.commit()
        stranger_id = stranger.id
    finally:
        session.close()

    response = client.post(
        "/v1/organizations/me/switch",
        json={"organization_id": str(stranger_id)},
        headers=master_key_header,
    )

    assert response.status_code == 404


def test_deleting_an_organization_moves_the_caller_to_another(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    default_organization_id = _context(client, master_key_header)["organization"]["id"]
    created = client.post("/v1/organizations/me", json={"name": "Acme"}, headers=master_key_header).json()

    deleted = client.delete("/v1/organizations/me", headers=master_key_header)

    assert deleted.status_code == 200, deleted.text
    context = _context(client, master_key_header)
    assert context["organization"]["id"] == default_organization_id
    assert created["organization"]["id"] != context["organization"]["id"]


def test_the_last_organization_cannot_be_deleted(client: TestClient, master_key_header: dict[str, str]) -> None:
    """``user.active_organization_id`` is NOT NULL, so there has to be somewhere to go."""
    _context(client, master_key_header)

    response = client.delete("/v1/organizations/me", headers=master_key_header)

    assert response.status_code == 400
    assert "no other organization" in response.json()["detail"]


# =============================================================================
# Organization membership
# =============================================================================


def test_the_roster_joins_identities(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    organization_id = uuid.UUID(_context(client, master_key_header)["organization"]["id"])
    _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )

    body = client.get("/v1/organizations/me/members", headers=master_key_header).json()

    assert body["count"] == 2
    rows = {row["full_name"]: row for row in body["data"]}
    assert rows["Ada Lovelace"]["email"] == "ada@example.com"
    assert rows["Ada Lovelace"]["role"] == "member"
    # The operator identity is a label, not a sign-in address.
    assert rows["Operator"]["email"] is None


def test_adding_a_member_creates_a_claimable_identity(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """An address nobody holds yet becomes an identity carrying it.

    The platform would email an invitation here and answer "invited". This
    edition has neither an invitation to send nor a way to accept one, so it
    answers on the other arm of the same result union.
    """
    added = client.post(
        "/v1/organizations/me/members",
        json={"email": "Ada@Example.com", "role": "admin"},
        headers=master_key_header,
    )

    assert added.status_code == 201, added.text
    body = added.json()
    assert body["status"] == "active"
    assert body["email"] == "ada@example.com"
    assert body["role"] == "admin"
    assert body["invitation_id"] is None

    roster = client.get("/v1/organizations/me/members", headers=master_key_header).json()
    assert roster["count"] == 2
    assert {row["email"] for row in roster["data"]} == {None, "ada@example.com"}


def test_adding_an_existing_identity_reuses_it(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """An address that already has an identity joins as that identity."""
    organization_id = uuid.UUID(_context(client, master_key_header)["organization"]["id"])
    existing = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )
    # Remove them, so the address exists but the membership does not.
    roster = client.get("/v1/organizations/me/members", headers=master_key_header).json()
    member_id = next(row["organization_member_id"] for row in roster["data"] if row["full_name"] == "Ada Lovelace")
    client.delete(f"/v1/organizations/me/members/{member_id}", headers=master_key_header)

    re_added = client.post(
        "/v1/organizations/me/members",
        json={"email": "ada@example.com", "role": "viewer"},
        headers=master_key_header,
    )

    assert re_added.status_code == 201, re_added.text
    body = re_added.json()
    assert body["user_id"] == str(existing)
    assert body["full_name"] == "Ada Lovelace"
    assert body["role"] == "viewer"
    # The suspended membership is revived rather than duplicated, so the history
    # attached to it survives.
    assert body["organization_member_id"] == member_id


def test_adding_an_active_member_twice_conflicts(client: TestClient, master_key_header: dict[str, str]) -> None:
    client.post(
        "/v1/organizations/me/members",
        json={"email": "ada@example.com"},
        headers=master_key_header,
    )

    again = client.post(
        "/v1/organizations/me/members",
        json={"email": "ada@example.com"},
        headers=master_key_header,
    )

    assert again.status_code == 409


def test_workspace_assignments_are_applied_with_the_member(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """No acceptance step exists to park them until, so they are granted now."""
    workspace = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header).json()

    added = client.post(
        "/v1/organizations/me/members",
        json={
            "email": "ada@example.com",
            "workspace_assignments": [{"workspace_id": workspace["id"], "role": "admin"}],
        },
        headers=master_key_header,
    )

    assert added.status_code == 201, added.text
    members = client.get(f"/v1/workspaces/{workspace['id']}/members", headers=master_key_header).json()
    assigned = next(row for row in members["data"] if row["user_id"] == added.json()["user_id"])
    assert assigned["role"] == "admin"


def test_an_assignment_naming_another_organizations_workspace_adds_nobody(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The whole add fails, rather than silently dropping one grant."""
    workspace = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header).json()
    client.post("/v1/organizations/me", json={"name": "Acme"}, headers=master_key_header)

    added = client.post(
        "/v1/organizations/me/members",
        json={
            "email": "ada@example.com",
            "workspace_assignments": [{"workspace_id": workspace["id"], "role": "member"}],
        },
        headers=master_key_header,
    )

    assert added.status_code == 404
    roster = client.get("/v1/organizations/me/members", headers=master_key_header).json()
    assert roster["count"] == 1


@pytest.mark.parametrize("email", ["not-an-address", "ada@example", "ada @example.com", ""])
def test_an_address_that_could_not_be_a_handle_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    email: str,
) -> None:
    response = client.post(
        "/v1/organizations/me/members",
        json={"email": email},
        headers=master_key_header,
    )

    assert response.status_code == 400
    assert "not a valid email address" in response.json()["detail"]


def test_a_member_cannot_be_parked_in_a_status_nothing_can_produce(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """``invited`` is a real stored status with no producer here, so it is not settable.

    Accepting it would let an admin move a member into a state that this edition
    has no way to leave, since accepting an invitation is what clears it.
    """
    added = client.post(
        "/v1/organizations/me/members",
        json={"email": "ada@example.com"},
        headers=master_key_header,
    ).json()

    response = client.patch(
        f"/v1/organizations/me/members/{added['organization_member_id']}",
        json={"status": "invited"},
        headers=master_key_header,
    )

    assert response.status_code == 422


def test_a_members_role_can_be_changed(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    organization_id = uuid.UUID(_context(client, master_key_header)["organization"]["id"])
    _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )
    roster = client.get("/v1/organizations/me/members", headers=master_key_header).json()
    member_id = next(row["organization_member_id"] for row in roster["data"] if row["full_name"] == "Ada Lovelace")

    response = client.patch(
        f"/v1/organizations/me/members/{member_id}",
        json={"role": "admin"},
        headers=master_key_header,
    )

    assert response.status_code == 200, response.text
    assert response.json()["role"] == "admin"


def test_removing_a_member_suspends_them_and_drops_them_from_the_roster(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    organization_id = uuid.UUID(_context(client, master_key_header)["organization"]["id"])
    _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )
    roster = client.get("/v1/organizations/me/members", headers=master_key_header).json()
    member_id = next(row["organization_member_id"] for row in roster["data"] if row["full_name"] == "Ada Lovelace")

    removed = client.delete(f"/v1/organizations/me/members/{member_id}", headers=master_key_header)

    assert removed.status_code == 200, removed.text
    after = client.get("/v1/organizations/me/members", headers=master_key_header).json()
    assert [row["full_name"] for row in after["data"]] == ["Operator"]


def test_the_last_owner_cannot_be_demoted(client: TestClient, master_key_header: dict[str, str]) -> None:
    """An organization with no owner has nobody who can manage or delete it."""
    context = _context(client, master_key_header)

    response = client.patch(
        f"/v1/organizations/me/members/{context['organization_member_id']}",
        json={"role": "member"},
        headers=master_key_header,
    )

    assert response.status_code == 400
    assert "at least one active owner" in response.json()["detail"]


def test_the_last_owner_cannot_be_removed(client: TestClient, master_key_header: dict[str, str]) -> None:
    context = _context(client, master_key_header)

    response = client.delete(
        f"/v1/organizations/me/members/{context['organization_member_id']}",
        headers=master_key_header,
    )

    assert response.status_code == 400


def test_another_organizations_member_is_not_found(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """The cross-tenant boundary on the member routes."""
    default_organization_id = uuid.UUID(_context(client, master_key_header)["organization"]["id"])
    _add_identity(
        db_session_factory,
        organization_id=default_organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )
    roster = client.get("/v1/organizations/me/members", headers=master_key_header).json()
    member_id = next(row["organization_member_id"] for row in roster["data"] if row["full_name"] == "Ada Lovelace")
    client.post("/v1/organizations/me", json={"name": "Acme"}, headers=master_key_header)

    response = client.patch(
        f"/v1/organizations/me/members/{member_id}",
        json={"role": "admin"},
        headers=master_key_header,
    )

    assert response.status_code == 404


# =============================================================================
# Workspaces
# =============================================================================


def test_workspace_create_read_update_delete(client: TestClient, master_key_header: dict[str, str]) -> None:
    created = client.post(
        "/v1/workspaces",
        json={"name": "Research", "description": "Model evaluation"},
        headers=master_key_header,
    )
    assert created.status_code == 201, created.text
    workspace = created.json()
    assert workspace["name"] == "Research"
    assert workspace["description"] == "Model evaluation"

    fetched = client.get(f"/v1/workspaces/{workspace['id']}", headers=master_key_header)
    assert fetched.status_code == 200
    assert fetched.json()["id"] == workspace["id"]

    renamed = client.patch(
        f"/v1/workspaces/{workspace['id']}",
        json={"name": "Evaluation"},
        headers=master_key_header,
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["name"] == "Evaluation"
    assert renamed.json()["description"] == "Model evaluation"

    deleted = client.delete(f"/v1/workspaces/{workspace['id']}", headers=master_key_header)
    assert deleted.status_code == 200, deleted.text
    assert client.get(f"/v1/workspaces/{workspace['id']}", headers=master_key_header).status_code == 404


def test_the_creator_becomes_the_workspaces_owner(client: TestClient, master_key_header: dict[str, str]) -> None:
    workspace = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header).json()

    members = client.get(f"/v1/workspaces/{workspace['id']}/members", headers=master_key_header).json()

    assert members["count"] == 1
    assert members["data"][0]["role"] == "owner"


def test_duplicate_workspace_names_conflict(client: TestClient, master_key_header: dict[str, str]) -> None:
    client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header)

    response = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header)

    assert response.status_code == 409


def test_renaming_onto_an_existing_workspace_name_conflicts(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header)
    other = client.post("/v1/workspaces", json={"name": "Evaluation"}, headers=master_key_header).json()

    response = client.patch(
        f"/v1/workspaces/{other['id']}",
        json={"name": "Research"},
        headers=master_key_header,
    )

    assert response.status_code == 409


def test_a_workspace_name_is_required(client: TestClient, master_key_header: dict[str, str]) -> None:
    """``name`` is NOT NULL with no minimum length, and a table model skips validation.

    The generated client types the update's ``name`` as ``string | null``, so a
    form clearing the field sends an explicit null; before the service checked,
    that reached the column as an integrity error rather than a 400.
    """
    workspace = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header).json()

    blank_create = client.post("/v1/workspaces", json={"name": "   "}, headers=master_key_header)
    null_update = client.patch(
        f"/v1/workspaces/{workspace['id']}",
        json={"name": None},
        headers=master_key_header,
    )

    assert blank_create.status_code == 400
    assert null_update.status_code == 400
    assert client.get(f"/v1/workspaces/{workspace['id']}", headers=master_key_header).json()["name"] == "Research"


def test_workspace_names_are_trimmed(client: TestClient, master_key_header: dict[str, str]) -> None:
    created = client.post("/v1/workspaces", json={"name": "  Research  "}, headers=master_key_header)

    assert created.status_code == 201, created.text
    assert created.json()["name"] == "Research"


def test_an_unknown_workspace_is_not_found(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.get(f"/v1/workspaces/{uuid.uuid4()}", headers=master_key_header)

    assert response.status_code == 404


def test_a_workspace_in_another_organization_is_not_found(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Cross-tenant reads answer 404, not 403: existence itself is scoped."""
    workspace = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header).json()
    client.post("/v1/organizations/me", json={"name": "Acme"}, headers=master_key_header)

    response = client.get(f"/v1/workspaces/{workspace['id']}", headers=master_key_header)

    assert response.status_code == 404


def test_workspaces_are_listed_per_organization(client: TestClient, master_key_header: dict[str, str]) -> None:
    client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header)
    client.post("/v1/organizations/me", json={"name": "Acme"}, headers=master_key_header)

    body = client.get("/v1/workspaces", headers=master_key_header).json()

    assert [workspace["name"] for workspace in body["data"]] == [DEFAULT_WORKSPACE_NAME]


# =============================================================================
# Workspace membership
# =============================================================================


def test_adding_and_removing_a_workspace_member(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    organization_id = uuid.UUID(_context(client, master_key_header)["organization"]["id"])
    user_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )
    workspace = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header).json()

    added = client.post(
        f"/v1/workspaces/{workspace['id']}/members/{user_id}",
        params={"role": "admin"},
        headers=master_key_header,
    )
    assert added.status_code == 201, added.text
    assert added.json()["role"] == "admin"

    members = client.get(f"/v1/workspaces/{workspace['id']}/members", headers=master_key_header).json()
    assert members["count"] == 2

    removed = client.delete(
        f"/v1/workspaces/{workspace['id']}/members/{user_id}",
        headers=master_key_header,
    )
    assert removed.status_code == 200, removed.text
    # Idempotent: removing a member who is already gone still succeeds.
    assert client.delete(
        f"/v1/workspaces/{workspace['id']}/members/{user_id}",
        headers=master_key_header,
    ).status_code == 200


def test_adding_the_same_member_twice_conflicts(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    organization_id = uuid.UUID(_context(client, master_key_header)["organization"]["id"])
    user_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )
    workspace = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header).json()
    client.post(f"/v1/workspaces/{workspace['id']}/members/{user_id}", headers=master_key_header)

    response = client.post(f"/v1/workspaces/{workspace['id']}/members/{user_id}", headers=master_key_header)

    assert response.status_code == 409


def test_a_workspace_is_not_a_back_door_into_the_organization(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Only an existing organization member can be added to a workspace."""
    workspace = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header).json()

    response = client.post(
        f"/v1/workspaces/{workspace['id']}/members/{uuid.uuid4()}",
        headers=master_key_header,
    )

    assert response.status_code == 400


def test_a_suspended_organization_member_cannot_join_a_workspace(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    organization_id = uuid.UUID(_context(client, master_key_header)["organization"]["id"])
    user_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
        status="suspended",
    )
    workspace = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header).json()

    response = client.post(f"/v1/workspaces/{workspace['id']}/members/{user_id}", headers=master_key_header)

    assert response.status_code == 400


def test_a_workspace_members_role_can_be_changed(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    organization_id = uuid.UUID(_context(client, master_key_header)["organization"]["id"])
    user_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )
    workspace = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header).json()
    client.post(f"/v1/workspaces/{workspace['id']}/members/{user_id}", headers=master_key_header)

    response = client.patch(
        f"/v1/workspaces/{workspace['id']}/members/{user_id}",
        params={"role": "viewer"},
        headers=master_key_header,
    )

    assert response.status_code == 200, response.text
    assert response.json()["role"] == "viewer"


def test_changing_a_role_that_does_not_exist_is_not_found(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    organization_id = uuid.UUID(_context(client, master_key_header)["organization"]["id"])
    user_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )
    workspace = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header).json()

    response = client.patch(
        f"/v1/workspaces/{workspace['id']}/members/{user_id}",
        params={"role": "viewer"},
        headers=master_key_header,
    )

    assert response.status_code == 404


def test_an_unknown_role_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Table models skip construction validation, so the service checks the role."""
    organization_id = uuid.UUID(_context(client, master_key_header)["organization"]["id"])
    user_id = _add_identity(
        db_session_factory,
        organization_id=organization_id,
        full_name="Ada Lovelace",
        email="ada@example.com",
    )
    workspace = client.post("/v1/workspaces", json={"name": "Research"}, headers=master_key_header).json()

    response = client.post(
        f"/v1/workspaces/{workspace['id']}/members/{user_id}",
        params={"role": "superuser"},
        headers=master_key_header,
    )

    assert response.status_code == 400
    assert "Invalid role" in response.json()["detail"]
