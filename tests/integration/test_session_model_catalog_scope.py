"""``GET /api/v1/models`` shows a tenant only the providers their organization reaches.

The roles matrix wants a member's model list narrowed to the providers they have
access to. The narrowing reuses the allow-list machinery an API key already goes
through (``services/model_access``), so the assertions here are about *which*
allow-list a caller is answered by rather than about a second matcher:

* a header master key is the deployment credential and is unrestricted;
* a session that operates the deployment is unrestricted;
* any other session is answered by its membership, which is every
  ``config.providers`` instance (deployment-wide, so every tenant reaches them)
  plus the organization's own BYO providers, plus the hosted providers the
  port serves where a workspace has no key.

The test deployment configures no ``providers:`` block, so every entry a caller
is shown here comes from a BYO key. A deployment with config-file providers gives
every tenant those on top.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.provider_keys import (
    OrgProviderKey,
    OrgProviderKeyModel,
    WorkspaceProviderKeyOverride,
    WorkspaceProviderModelRestriction,
)
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.secret_box import encrypt_secret, generate_secret_key

from .hosted_port_helpers import HostedModelProvider, bind_model_provider

# Priced but undiscovered models: phase 2 of the listing publishes them, so the
# catalog is deterministic without dialing a provider.
_OPENAI_MODEL = "openai:gpt-4o-mini"
_OPENAI_OTHER = "openai:gpt-4o"
_ANTHROPIC_MODEL = "anthropic:claude-3-5-haiku-latest"
_MISTRAL_MODEL = "mistral:mistral-small-latest"
_ALL_MODELS = (_OPENAI_MODEL, _OPENAI_OTHER, _ANTHROPIC_MODEL, _MISTRAL_MODEL)


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real key, so a stored credential decrypts as it would in production.

    Without it every BYO row is unusable and the catalog withholds its provider,
    which is correct behavior (see the undecryptable-key test below) and would
    make every other case here vacuous.
    """
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


@dataclass
class _World:
    alpha: uuid.UUID
    beta: uuid.UUID
    workspaces: dict[str, uuid.UUID] = field(default_factory=dict)
    keys: dict[str, uuid.UUID] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)


def _identity(
    session: Session,
    *,
    email: str,
    organization_id: uuid.UUID,
    role: str = "member",
    is_superuser: bool = False,
    workspace_ids: tuple[uuid.UUID, ...] = (),
) -> str:
    user = User(
        email=email,
        full_name=email.split("@")[0].title(),
        active_organization_id=organization_id,
        is_superuser=is_superuser,
    )
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


def _unmembered_identity(session: Session, *, email: str, organization_id: uuid.UUID) -> str:
    """A signed-in identity pointed at a real organization it holds no membership in.

    ``users.active_organization_id`` is a foreign key, so a pointer at nothing is
    not a state the database can hold; a pointer with no membership behind it is,
    and it is the one the scope resolver refuses.
    """
    user = User(email=email, full_name="Orphan", active_organization_id=organization_id)
    session.add(user)
    session.commit()
    session.refresh(user)
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


def _byo_key(
    session: Session,
    *,
    organization_id: uuid.UUID,
    provider: str,
    encrypted_api_key: str | None = None,
) -> uuid.UUID:
    key = OrgProviderKey(
        organization_id=organization_id,
        provider=provider,
        name=f"{provider}-primary",
        encrypted_api_key=encrypted_api_key or encrypt_secret("sk-test-value"),
        last4="1234",
        is_org_default=True,
    )
    session.add(key)
    session.commit()
    session.refresh(key)
    return key.id


@pytest.fixture
def world(client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]) -> _World:
    """Two tenants with different BYO providers, and a priced catalog spanning three."""
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == status.HTTP_200_OK
    for model_key in _ALL_MODELS:
        priced = client.post(
            f"{API_ROOT}/pricing",
            json={"model_key": model_key, "input_price_per_million": 1.0, "output_price_per_million": 2.0},
            headers=master_key_header,
        )
        assert priced.status_code == status.HTTP_200_OK, priced.text

    session = db_session_factory()
    try:
        alpha = Organization(name="Alpha", slug="alpha")
        beta = Organization(name="Beta", slug="beta")
        session.add_all([alpha, beta])
        session.commit()
        session.refresh(alpha)
        session.refresh(beta)

        alpha_one = Workspace(name="Alpha one", organization_id=alpha.id)
        alpha_two = Workspace(name="Alpha two", organization_id=alpha.id)
        beta_one = Workspace(name="Beta one", organization_id=beta.id)
        session.add_all([alpha_one, alpha_two, beta_one])
        session.commit()
        for workspace in (alpha_one, alpha_two, beta_one):
            session.refresh(workspace)

        built = _World(alpha=alpha.id, beta=beta.id)
        built.workspaces = {"alpha_one": alpha_one.id, "alpha_two": alpha_two.id, "beta_one": beta_one.id}
        built.keys = {
            "alpha_openai": _byo_key(session, organization_id=alpha.id, provider="openai"),
            "beta_anthropic": _byo_key(session, organization_id=beta.id, provider="anthropic"),
        }
        built.sessions = {
            "alpha_owner": _identity(session, email="owner@alpha.test", organization_id=alpha.id, role="owner"),
            "alpha_member": _identity(
                session,
                email="member@alpha.test",
                organization_id=alpha.id,
                workspace_ids=(alpha_one.id,),
            ),
            "alpha_newcomer": _identity(session, email="new@alpha.test", organization_id=alpha.id),
            "beta_member": _identity(
                session,
                email="member@beta.test",
                organization_id=beta.id,
                workspace_ids=(beta_one.id,),
            ),
            "orphan": _unmembered_identity(session, email="orphan@nowhere.test", organization_id=beta.id),
            "superuser": _identity(
                session,
                email="root@alpha.test",
                organization_id=alpha.id,
                role="owner",
                is_superuser=True,
            ),
        }
        return built
    finally:
        session.close()


def _listing_as(client: TestClient, world: _World, who: str) -> dict[str, dict[str, Any]]:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        response = client.get(f"{API_ROOT}/models")
        assert response.status_code == status.HTTP_200_OK, response.text
        return {model["id"]: model for model in response.json()["data"]}
    finally:
        client.cookies.clear()


def _catalog_as(client: TestClient, world: _World, who: str) -> set[str]:
    return set(_listing_as(client, world, who))


def test_the_master_key_still_sees_every_priced_model(
    client: TestClient, master_key_header: dict[str, str], world: _World
) -> None:
    """The control: the deployment credential is not narrowed by anyone's membership."""
    response = client.get(f"{API_ROOT}/models", headers=master_key_header)
    assert response.status_code == status.HTTP_200_OK, response.text
    assert {model["id"] for model in response.json()["data"]} == set(_ALL_MODELS)


def test_a_member_sees_only_their_organizations_providers(client: TestClient, world: _World) -> None:
    listed = _catalog_as(client, world, "alpha_member")
    assert listed == {_OPENAI_MODEL, _OPENAI_OTHER}
    assert _ANTHROPIC_MODEL not in listed
    assert _MISTRAL_MODEL not in listed


def test_two_tenants_are_shown_disjoint_catalogs(client: TestClient, world: _World) -> None:
    """Stated on its own, because it is the claim the narrowing rests on."""
    assert _catalog_as(client, world, "alpha_member").isdisjoint(_catalog_as(client, world, "beta_member"))


def test_an_admin_is_answered_from_the_organizations_providers_not_one_workspaces(
    client: TestClient, world: _World
) -> None:
    """An owner belongs to no workspace here, and still reads the organization's providers.

    An implementation that walked the caller's workspace memberships would show
    them nothing, which is the failure this pins.
    """
    assert _catalog_as(client, world, "alpha_owner") == {_OPENAI_MODEL, _OPENAI_OTHER}


def test_a_member_of_no_workspace_sees_no_byo_models_rather_than_a_refusal(client: TestClient, world: _World) -> None:
    """Nothing was refused; no workspace of theirs holds a key yet."""
    assert _catalog_as(client, world, "alpha_newcomer") == set()


def test_a_deployment_operator_session_is_not_narrowed(client: TestClient, world: _World) -> None:
    """A superuser operates the deployment, so the catalog is the deployment's."""
    assert _catalog_as(client, world, "superuser") == set(_ALL_MODELS)


def test_a_workspace_model_restriction_narrows_a_members_catalog(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    """The allow-list a workspace already enforces at dispatch decides the listing too."""
    session = db_session_factory()
    try:
        session.add(
            WorkspaceProviderModelRestriction(
                workspace_id=world.workspaces["alpha_one"],
                organization_id=world.alpha,
                org_provider_key_id=world.keys["alpha_openai"],
                model="gpt-4o-mini",
            )
        )
        session.commit()
    finally:
        session.close()

    assert _catalog_as(client, world, "alpha_member") == {_OPENAI_MODEL}
    # The admin still reads the organization's whole provider, because lifting one
    # workspace's restriction is theirs to do.
    assert _catalog_as(client, world, "alpha_owner") == {_OPENAI_MODEL, _OPENAI_OTHER}


def test_an_offered_model_narrows_the_catalog_and_appears_in_it(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    """Two things at once, and they are the point of the offered-models table.

    Narrowing: the organization has adopted one of its provider's models, so the
    other stops being listed even though the key still reaches the provider.

    Listing: ``gpt-5-adopted`` is priced by nothing and discovered by nothing, so
    before this table it could not appear in the catalog at all. It appears
    because the organization offered it.
    """
    session = db_session_factory()
    try:
        session.add(
            OrgProviderKeyModel(
                organization_id=world.alpha,
                org_provider_key_id=world.keys["alpha_openai"],
                model="gpt-5-adopted",
                enabled=True,
            )
        )
        session.commit()
    finally:
        session.close()

    assert _catalog_as(client, world, "alpha_member") == {"openai:gpt-5-adopted"}
    assert _catalog_as(client, world, "alpha_owner") == {"openai:gpt-5-adopted"}


def test_a_model_switched_off_leaves_the_catalog(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    """The catalog never advertises a model that would be refused at inference,
    and the serving switch is what refuses it."""
    session = db_session_factory()
    try:
        session.add_all(
            [
                OrgProviderKeyModel(
                    organization_id=world.alpha,
                    org_provider_key_id=world.keys["alpha_openai"],
                    model="gpt-4o-mini",
                    enabled=True,
                ),
                OrgProviderKeyModel(
                    organization_id=world.alpha,
                    org_provider_key_id=world.keys["alpha_openai"],
                    model="gpt-4o",
                    enabled=False,
                ),
            ]
        )
        session.commit()
    finally:
        session.close()

    assert _catalog_as(client, world, "alpha_member") == {_OPENAI_MODEL}


def test_the_two_narrowings_intersect_rather_than_widen(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    """A workspace restriction may only narrow what the organization offers, and
    the organization offering a model does not lift the workspace's own list."""
    session = db_session_factory()
    try:
        session.add_all(
            [
                OrgProviderKeyModel(
                    organization_id=world.alpha,
                    org_provider_key_id=world.keys["alpha_openai"],
                    model="gpt-4o-mini",
                    enabled=True,
                ),
                OrgProviderKeyModel(
                    organization_id=world.alpha,
                    org_provider_key_id=world.keys["alpha_openai"],
                    model="gpt-4o",
                    enabled=True,
                ),
                WorkspaceProviderModelRestriction(
                    workspace_id=world.workspaces["alpha_one"],
                    organization_id=world.alpha,
                    org_provider_key_id=world.keys["alpha_openai"],
                    model="gpt-4o-mini",
                ),
            ]
        )
        session.commit()
    finally:
        session.close()

    # The member is in the restricted workspace, so they get the intersection.
    assert _catalog_as(client, world, "alpha_member") == {_OPENAI_MODEL}
    # The admin reads the organization whole, so the workspace's own list does
    # not bind them, but what the organization withdrew still does.
    assert _catalog_as(client, world, "alpha_owner") == {_OPENAI_MODEL, _OPENAI_OTHER}


def test_another_organization_never_sees_an_offered_model(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    session = db_session_factory()
    try:
        session.add(
            OrgProviderKeyModel(
                organization_id=world.alpha,
                org_provider_key_id=world.keys["alpha_openai"],
                model="gpt-5-adopted",
                enabled=True,
            )
        )
        session.commit()
    finally:
        session.close()

    assert "openai:gpt-5-adopted" not in _catalog_as(client, world, "beta_member")


def test_a_single_model_read_agrees_with_the_listing(client: TestClient, world: _World) -> None:
    """A model withheld from the listing is 404 by id, never a different answer."""
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    try:
        assert client.get(f"{API_ROOT}/models/{_OPENAI_MODEL}").status_code == status.HTTP_200_OK
        assert client.get(f"{API_ROOT}/models/{_ANTHROPIC_MODEL}").status_code == status.HTTP_404_NOT_FOUND
    finally:
        client.cookies.clear()


def test_an_identity_with_no_live_membership_is_answered_rather_than_refused(client: TestClient, world: _World) -> None:
    """A catalog read is not one of the routes whose whole question is "which organization".

    The pointer is not the authority, so this caller reaches no organization's
    keys, which is an empty list here (the test deployment configures no
    ``providers:`` block) rather than a 403 over a page that used to render. In
    particular it must not be shown Beta's models just because it points there.
    """
    assert _catalog_as(client, world, "orphan") == set()


def test_a_foreign_workspaces_alias_names_are_not_listed(
    client: TestClient, master_key_header: dict[str, str], world: _World
) -> None:
    """An alias name is not filtered by the allow-list, so the layer has to be scoped.

    ``services/alias_service`` reads the deployment's default workspace when no
    workspace is named, which is what the master-key write below means and what a
    session used to be answered from. The alias points at a target Alpha *can*
    reach, so the model allow-list permits it and only the workspace scoping keeps
    the name out: the case a filter over targets cannot catch (otari-ai#1969).
    """
    created = client.post(
        f"{API_ROOT}/aliases",
        json={"name": "acme-confidential-summarizer", "target": _OPENAI_MODEL},
        headers=master_key_header,
    )
    assert created.status_code == status.HTTP_200_OK, created.text

    listed = _catalog_as(client, world, "alpha_member")
    assert "acme-confidential-summarizer" not in listed
    # The providers themselves are unaffected: this withholds another workspace's
    # names, not a tenant's models. The target stays listed for the member, and
    # correctly so: an alias withholds its target to keep the indirection intact,
    # and there is no indirection here for a caller who is shown no alias.
    assert listed == {_OPENAI_MODEL, _OPENAI_OTHER}

    # The control: an operator is answered from the workspace the alias lives in,
    # so the name is still there for the caller it belongs to.
    assert "acme-confidential-summarizer" in _catalog_as(client, world, "superuser")


def test_a_provider_whose_only_key_will_not_decrypt_is_withheld(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    """The catalog must not advertise what dispatch cannot serve.

    ``refresh_org_provider_cache`` skips a row whose secret will not decrypt, so
    a request through that provider has no credential. Listing its models would
    break the rule this filter exists to keep, that the catalog never shows a
    model the caller cannot actually use.
    """
    session = db_session_factory()
    try:
        _byo_key(
            session,
            organization_id=world.beta,
            provider="mistral",
            encrypted_api_key="not-a-fernet-token",
        )
    finally:
        session.close()

    listed = _catalog_as(client, world, "beta_member")
    assert _ANTHROPIC_MODEL in listed, "beta's decryptable key still counts"
    assert _MISTRAL_MODEL not in listed, "the undecryptable key's provider is withheld"


def test_a_hosted_provider_is_listed_for_a_member_of_an_organization_holding_no_key_for_it(
    client: TestClient, world: _World
) -> None:
    """Alpha holds a BYO key for openai only, and the deployment serves mistral."""
    port = HostedModelProvider("mistral")
    bind_model_provider(client, port)

    listed = _listing_as(client, world, "alpha_member")
    assert set(listed) == {_OPENAI_MODEL, _OPENAI_OTHER, _MISTRAL_MODEL}
    assert listed[_MISTRAL_MODEL]["deployment_managed"] is True, "the deployment pays the mistral bill"
    assert listed[_OPENAI_MODEL]["deployment_managed"] is False, "alpha's own key pays the openai bill"
    assert port.asked_for == [world.alpha], "the port is asked for the caller's organization, once"


def test_a_hosted_provider_the_organization_also_holds_a_key_for_is_the_organizations_to_price(
    client: TestClient, world: _World
) -> None:
    """Alpha holds an openai key and may price the model, while Beta holds none and may not."""
    bind_model_provider(client, HostedModelProvider("openai"))

    alpha = _listing_as(client, world, "alpha_member")
    assert alpha[_OPENAI_MODEL]["deployment_managed"] is False

    beta = _listing_as(client, world, "beta_member")
    assert set(beta) == {_OPENAI_MODEL, _OPENAI_OTHER, _ANTHROPIC_MODEL}
    assert beta[_OPENAI_MODEL]["deployment_managed"] is True
    assert beta[_ANTHROPIC_MODEL]["deployment_managed"] is False


@pytest.mark.parametrize(
    ("model", "disabled_in_alpha_two", "deployment_managed"),
    [
        pytest.param(_MISTRAL_MODEL, False, True, id="no-byo-key"),
        pytest.param(_OPENAI_MODEL, False, False, id="byo-key-in-every-workspace"),
        pytest.param(_OPENAI_MODEL, True, True, id="one-workspace-disables-the-byo-key"),
    ],
)
def test_the_deployment_managed_flag_agrees_with_the_rate_override_gate(
    client: TestClient,
    world: _World,
    db_session_factory: Callable[[], Session],
    model: str,
    disabled_in_alpha_two: bool,
    deployment_managed: bool,
) -> None:
    """The flag must predict whether the organization may set its own rate."""
    bind_model_provider(client, HostedModelProvider("openai", "mistral"))
    if disabled_in_alpha_two:
        session = db_session_factory()
        try:
            session.add(
                WorkspaceProviderKeyOverride(
                    workspace_id=world.workspaces["alpha_two"],
                    organization_id=world.alpha,
                    org_provider_key_id=world.keys["alpha_openai"],
                    is_default=False,
                    disabled=True,
                )
            )
            session.commit()
        finally:
            session.close()

    listed = _listing_as(client, world, "alpha_owner")
    assert listed[model]["deployment_managed"] is deployment_managed

    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_owner"])
    try:
        written = client.post(
            f"{API_ROOT}/organizations/me/pricing",
            json={"model_key": model, "input_price_per_million": 2.5, "output_price_per_million": 5.0},
        )
    finally:
        client.cookies.clear()
    expected = status.HTTP_403_FORBIDDEN if deployment_managed else status.HTTP_201_CREATED
    assert written.status_code == expected, written.text


@pytest.mark.parametrize(
    ("who", "restricted_in_alpha_one", "disabled_in_alpha_one", "hosted", "expected"),
    [
        pytest.param(
            "alpha_member", True, False, ("openai",), {_OPENAI_MODEL}, id="the-port-does-not-widen-a-restricted-key"
        ),
        pytest.param(
            "alpha_member",
            True,
            True,
            ("openai",),
            {_OPENAI_MODEL, _OPENAI_OTHER},
            id="a-disabled-key-leaves-the-provider-to-the-port",
        ),
        pytest.param("alpha_member", False, True, (), set(), id="a-disabled-key-and-no-port-list-nothing"),
        pytest.param("alpha_newcomer", False, False, ("mistral",), set(), id="a-member-of-no-workspace-gets-nothing"),
        pytest.param(
            "alpha_owner",
            True,
            False,
            ("mistral",),
            {_OPENAI_MODEL, _OPENAI_OTHER, _MISTRAL_MODEL},
            id="an-owner-sees-the-whole-organization",
        ),
    ],
)
def test_a_hosted_provider_is_listed_only_where_dispatch_would_ask_the_port(
    client: TestClient,
    world: _World,
    db_session_factory: Callable[[], Session],
    who: str,
    restricted_in_alpha_one: bool,
    disabled_in_alpha_one: bool,
    hosted: tuple[str, ...],
    expected: set[str],
) -> None:
    """Dispatch asks the port only when a workspace has no active key with a credential for the provider."""
    bind_model_provider(client, HostedModelProvider(*hosted))
    session = db_session_factory()
    try:
        if restricted_in_alpha_one:
            session.add(
                WorkspaceProviderModelRestriction(
                    workspace_id=world.workspaces["alpha_one"],
                    organization_id=world.alpha,
                    org_provider_key_id=world.keys["alpha_openai"],
                    model="gpt-4o-mini",
                )
            )
        if disabled_in_alpha_one:
            session.add(
                WorkspaceProviderKeyOverride(
                    workspace_id=world.workspaces["alpha_one"],
                    organization_id=world.alpha,
                    org_provider_key_id=world.keys["alpha_openai"],
                    is_default=False,
                    disabled=True,
                )
            )
        session.commit()
    finally:
        session.close()

    assert _catalog_as(client, world, who) == expected


def test_the_single_model_read_agrees_with_the_listing_about_a_hosted_model(client: TestClient, world: _World) -> None:
    bind_model_provider(client, HostedModelProvider("mistral"))
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    try:
        response = client.get(f"{API_ROOT}/models/{_MISTRAL_MODEL}")
        assert response.status_code == status.HTTP_200_OK, response.text
        assert response.json()["deployment_managed"] is True
        assert client.get(f"{API_ROOT}/models/{_ANTHROPIC_MODEL}").status_code == status.HTTP_404_NOT_FOUND
    finally:
        client.cookies.clear()


def test_the_grouped_catalog_lists_a_hosted_model_for_a_member(client: TestClient, world: _World) -> None:
    bind_model_provider(client, HostedModelProvider("mistral"))
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    try:
        response = client.get(f"{API_ROOT}/catalog/models")
        assert response.status_code == status.HTTP_200_OK, response.text
        selectors = {selector for model in response.json()["models"] for selector in model["selectors"]}
        assert _MISTRAL_MODEL in selectors
        assert _ANTHROPIC_MODEL not in selectors
    finally:
        client.cookies.clear()


def test_the_grouped_catalog_labels_a_hosted_offering_hosted(client: TestClient, world: _World) -> None:
    """An offering labeled ``organization`` is one the organization may price."""
    bind_model_provider(client, HostedModelProvider("mistral"))
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    credentials: dict[str, str] = {}
    try:
        listing = client.get(f"{API_ROOT}/catalog/models")
        assert listing.status_code == status.HTTP_200_OK, listing.text
        for model in listing.json()["models"]:
            detail = client.get(f"{API_ROOT}/catalog/models/{model['id']}")
            assert detail.status_code == status.HTTP_200_OK, detail.text
            for offering in detail.json()["offerings"]:
                credentials[offering["selector"]] = offering["credential"]
    finally:
        client.cookies.clear()

    assert credentials[_MISTRAL_MODEL] == "hosted"
    assert credentials[_OPENAI_MODEL] == "organization"


def test_an_operator_session_flags_no_hosted_model(client: TestClient, world: _World) -> None:
    """An operator sees every model and may price any of them."""
    bind_model_provider(client, HostedModelProvider("mistral"))
    listed = _listing_as(client, world, "superuser")
    assert set(listed) == set(_ALL_MODELS)
    assert listed[_MISTRAL_MODEL]["deployment_managed"] is False


def test_a_hosted_port_failure_fails_the_read(client: TestClient, world: _World) -> None:
    """A catalog that hid the failure would list fewer models and flag them wrongly."""
    bind_model_provider(client, HostedModelProvider("mistral", error=RuntimeError("fleet store unreachable")))
    with pytest.raises(RuntimeError, match="fleet store unreachable"):
        _catalog_as(client, world, "alpha_member")


def test_an_identity_with_no_live_membership_is_not_shown_hosted_models(client: TestClient, world: _World) -> None:
    """The port is asked about an organization, and this caller has none."""
    port = HostedModelProvider("mistral")
    bind_model_provider(client, port)
    assert _catalog_as(client, world, "orphan") == set()
    assert port.asked_for == []


def test_an_offered_model_resolves_by_its_catalog_spellings_for_its_organization_alone(
    client: TestClient,
    master_key_header: dict[str, str],
    world: _World,
    db_session_factory: Callable[[], Session],
) -> None:
    """A BYO offering is reached by the catalog id and by the pinned spelling.

    Nebius spells DeepSeek's model ``deepseek-ai/DeepSeek-V4.1-Flash``, and
    nothing on the deployment serves it. Once alpha offers it on its own nebius
    key, alpha's callers reach it by every catalog spelling and resolution lands
    on the raw id the key's allow-list and pricing are keyed on; beta's callers
    and the deployment's own view are left alone, so a tenant's key never decides
    where another tenant's selector goes.
    """
    from typing import cast

    from fastapi import FastAPI

    from gateway.services import catalog_selectors as selectors
    from gateway.services.provider_kwargs import resolve_provider_selector

    raw = "nebius:deepseek-ai/DeepSeek-V4.1-Flash"
    session = db_session_factory()
    try:
        nebius = _byo_key(session, organization_id=world.alpha, provider="nebius")
        session.add(
            OrgProviderKeyModel(
                organization_id=world.alpha,
                org_provider_key_id=nebius,
                model="deepseek-ai/DeepSeek-V4.1-Flash",
                enabled=True,
            )
        )
        session.commit()
    finally:
        session.close()

    config = cast(FastAPI, client.app).state.config
    try:
        rebuilt = client.post(f"{API_ROOT}/catalog/selectors/refresh", headers=master_key_header)
        assert rebuilt.status_code == status.HTTP_200_OK, rebuilt.text
        alpha_workspace = world.workspaces["alpha_one"]
        for spelling in ("deepseek/deepseek-v4.1-flash", "nebius:deepseek/deepseek-v4.1-flash"):
            resolved = resolve_provider_selector(config, spelling, workspace_id=alpha_workspace)
            assert (resolved.instance, resolved.model, resolved.alias) == (
                "nebius",
                "deepseek-ai/DeepSeek-V4.1-Flash",
                spelling,
            ), spelling
            assert selectors.resolve_catalog_selector(spelling, workspace_id=world.workspaces["beta_one"]) is None
            assert selectors.resolve_catalog_selector(spelling) is None
        # The raw selector is never rewritten, for anyone.
        assert selectors.resolve_catalog_selector(raw, workspace_id=alpha_workspace) is None

        # The catalog tells alpha's admin the spellings in force, and the id resolves for them.
        client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_owner"])
        try:
            detail = client.get(f"{API_ROOT}/catalog/models/deepseek/deepseek-v4.1-flash")
            assert detail.status_code == status.HTTP_200_OK, detail.text
            body = detail.json()
            assert body["selector"] == "deepseek/deepseek-v4.1-flash"
            assert body["resolves_to"] == raw
            offering = next(row for row in body["offerings"] if row["selector"] == raw)
            assert offering["short_selector"] == "nebius:deepseek/deepseek-v4.1-flash"
        finally:
            client.cookies.clear()
    finally:
        selectors.reset_selector_index()
