"""The Playground on a hosted control plane, which serves no inference itself.

Standalone runs the completion in process (``tests/integration/test_playground``).
A control plane cannot: a completion served there would skip the usage report
that debits the wallet, which is otari#822. So the page is served and that one
request is forwarded to the data-plane gateway, under a key that stands for the
caller.

What that key is, is the whole of the risk, and most of this file. It decides
whose budget binds and whose usage is recorded on the far side, so it carries the
caller's own user, workspace and allow-list and nobody else's; and it is a
credential this deployment stores rather than only verifies, so no key surface
hands it back, rotates it, or lists it beside the keys somebody actually manages.
"""

import uuid
from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta

import httpx
import httpx2
import pytest
from cryptography.fernet import Fernet
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.usage_source import SERVED_HERE_SLUG
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.models.usage import UsageLog
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.secret_box import decrypt_secret, encrypt_secret, generate_secret_key

from .conftest import build_test_client

_DATA_PLANE = "https://gateway.example.test"
_COMPLETIONS = f"{API_ROOT}/playground/chat/completions"

# Bound before any test patches the name, so the factory below builds a real
# client rather than calling itself.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


class _Stream(httpx.AsyncByteStream):
    """A response body the mock transport hands over unread."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        yield self._payload


def _config(postgres_url: str, data_plane_url: str | None) -> GatewayConfig:
    return GatewayConfig(
        mode="hosted",
        database_url=postgres_url,
        data_plane_url=data_plane_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        bootstrap_api_key=False,
    )


@pytest.fixture
def secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deployment can encrypt at rest, as a hosted one always can."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


@pytest.fixture
def hosted_client(postgres_url: str, clean_database: None, secret_key: None) -> Generator[TestClient]:
    yield from build_test_client(_config(postgres_url, _DATA_PLANE))


@pytest.fixture
def hosted_client_without_a_data_plane(
    postgres_url: str, clean_database: None, secret_key: None
) -> Generator[TestClient]:
    yield from build_test_client(_config(postgres_url, None))


@pytest.fixture
def caller(db_session_factory: Callable[[], Session]) -> tuple[uuid.UUID, uuid.UUID, str]:
    """One member of one workspace, and the cookie they sign in with.

    Returns their identity id, their workspace id, and the session token.
    """
    session = db_session_factory()
    try:
        organization = Organization(name="Alpha", slug="alpha")
        session.add(organization)
        session.commit()
        session.refresh(organization)

        workspace = Workspace(name="Alpha one", organization_id=organization.id)
        session.add(workspace)
        session.commit()
        session.refresh(workspace)

        user = User(email="member@alpha.test", full_name="Member", active_organization_id=organization.id)
        session.add(user)
        session.commit()
        session.refresh(user)

        session.add(
            OrganizationMember(organization_id=organization.id, user_id=user.id, role="member", status="active")
        )
        session.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role="member", status="active"))
        token = "otari-sess-member"
        session.add(
            DashboardSession(
                token_hash=hash_session_token(token),
                user_id=user.id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=12),
            )
        )
        session.commit()
        return user.id, workspace.id, token
    finally:
        session.close()


def _answer_from_the_data_plane(
    monkeypatch: pytest.MonkeyPatch,
    seen: dict[str, object],
    *,
    status_code: int = 200,
    payload: bytes = b'{"id": "chat-1"}',
) -> None:
    """Stand a data-plane gateway up in place of the network."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            status_code,
            stream=_Stream(payload),
            headers={"content-type": "application/json"},
        )

    def build(**kwargs: object) -> httpx.AsyncClient:
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", build)


def _send(client: TestClient, token: str, workspace_id: uuid.UUID) -> httpx2.Response:
    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        return client.post(
            _COMPLETIONS,
            params={"workspace_id": str(workspace_id)},
            json={"model": "openai:gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
    finally:
        client.cookies.clear()


def _internal_keys(db_session_factory: Callable[[], Session]) -> list[APIKey]:
    session = db_session_factory()
    try:
        return list(session.query(APIKey).filter(APIKey.internal_secret.isnot(None)).all())
    finally:
        session.close()


def test_the_completion_is_forwarded_to_the_data_plane(
    hosted_client: TestClient,
    caller: tuple[uuid.UUID, uuid.UUID, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The page gets its answer, and the answer came from the gateway."""
    _, workspace_id, token = caller
    seen: dict[str, object] = {}
    _answer_from_the_data_plane(monkeypatch, seen)

    response = _send(hosted_client, token, workspace_id)

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"id": "chat-1"}
    assert seen["url"] == f"{_DATA_PLANE}{API_ROOT}/chat/completions"


def test_the_forwarded_key_stands_for_the_caller_and_their_workspace(
    hosted_client: TestClient,
    caller: tuple[uuid.UUID, uuid.UUID, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whose budget the far side binds is decided entirely by this key.

    So it names the caller's own attribution user and the workspace they proved
    membership of, and it is not budget-exempt: the one page that runs
    completions must not be the one page that ignores budgets.
    """
    _, workspace_id, token = caller
    seen: dict[str, object] = {}
    _answer_from_the_data_plane(monkeypatch, seen)

    _send(hosted_client, token, workspace_id)

    keys = _internal_keys(db_session_factory)
    assert len(keys) == 1
    minted = keys[0]
    assert minted.workspace_id == workspace_id
    assert minted.user_id is not None
    assert minted.exclude_from_budget is False
    # The bearer the gateway saw is the plaintext this row stores, and the row
    # stores it encrypted: presenting a credential is the one thing the hash
    # cannot do, so it is held the way provider credentials are.
    assert minted.internal_secret is not None
    assert seen["authorization"] == f"Bearer {decrypt_secret(minted.internal_secret)}"


def test_the_key_is_minted_once_and_reused(
    hosted_client: TestClient,
    caller: tuple[uuid.UUID, uuid.UUID, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chat page sends many messages and must not mint a key per message."""
    _, workspace_id, token = caller
    seen: dict[str, object] = {}
    _answer_from_the_data_plane(monkeypatch, seen)

    _send(hosted_client, token, workspace_id)
    first = seen["authorization"]
    _send(hosted_client, token, workspace_id)

    assert len(_internal_keys(db_session_factory)) == 1
    assert seen["authorization"] == first


def test_a_duplicate_key_from_a_race_is_read_rather_than_refused(
    hosted_client: TestClient,
    caller: tuple[uuid.UUID, uuid.UUID, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two first messages at once can both insert, and neither breaks the page.

    Nothing in the schema forbids a second row, so the lookup takes the oldest
    match rather than insisting on exactly one: insisting would turn a momentary
    race into a 500 on every later request, forever.
    """
    _, workspace_id, token = caller
    seen: dict[str, object] = {}
    _answer_from_the_data_plane(monkeypatch, seen)
    _send(hosted_client, token, workspace_id)
    first = _internal_keys(db_session_factory)[0]

    # A credential of its own, and explicitly the newer row. Copying the first
    # row's secret would let the assertion below pass whichever row the lookup
    # picked, which is the one thing this test exists to tell apart.
    session = db_session_factory()
    try:
        session.add(
            APIKey(
                id=str(uuid.uuid4()),
                workspace_id=first.workspace_id,
                key_hash="a-second-row-from-a-race",
                user_id=first.user_id,
                created_at=first.created_at + timedelta(minutes=1),
                internal_secret=encrypt_secret("gw-the-loser-of-the-race"),
            )
        )
        session.commit()
    finally:
        session.close()

    response = _send(hosted_client, token, workspace_id)

    assert response.status_code == status.HTTP_200_OK
    assert seen["authorization"] == f"Bearer {decrypt_secret(first.internal_secret or '')}"
    assert seen["authorization"] != "Bearer gw-the-loser-of-the-race"


def test_an_unreadable_stored_key_is_replaced_rather_than_refused(
    hosted_client: TestClient,
    caller: tuple[uuid.UUID, uuid.UUID, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secret this deployment cannot read is one only this code can mend.

    Rotating ``OTARI_SECRET_KEY`` without carrying the old value leaves every
    ciphertext undecryptable. For a provider credential that has to be a refusal,
    because the plaintext was the customer's. This one is ours and nobody else
    holds it, so the row is rewritten and the page keeps working.
    """
    _, workspace_id, token = caller
    seen: dict[str, object] = {}
    _answer_from_the_data_plane(monkeypatch, seen)
    _send(hosted_client, token, workspace_id)
    original = _internal_keys(db_session_factory)[0]
    original_id, original_hash = original.id, original.key_hash

    session = db_session_factory()
    try:
        row = session.get(APIKey, original_id)
        assert row is not None
        # Ciphertext from somebody else's key, which is what a rotation leaves.
        row.internal_secret = Fernet(generate_secret_key()).encrypt(b"gw-unreadable").decode()
        session.commit()
    finally:
        session.close()

    response = _send(hosted_client, token, workspace_id)

    assert response.status_code == status.HTTP_200_OK
    keys = _internal_keys(db_session_factory)
    # The same row, mended, rather than a second one beside it.
    assert [key.id for key in keys] == [original_id]
    assert keys[0].key_hash != original_hash
    assert seen["authorization"] == f"Bearer {decrypt_secret(keys[0].internal_secret or '')}"


def test_the_internal_key_is_absent_from_the_key_surfaces(
    hosted_client: TestClient,
    caller: tuple[uuid.UUID, uuid.UUID, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It is machinery, not a credential anybody manages.

    Listed, it is a key nobody created and cannot use. Reachable by id, it is a
    key a rotation would break silently: the stored plaintext would no longer
    match the hash, and the Playground would start failing with nothing on screen
    to explain it. So the same predicate hides it from both.
    """
    _, workspace_id, token = caller
    _answer_from_the_data_plane(monkeypatch, {})
    _send(hosted_client, token, workspace_id)
    minted = _internal_keys(db_session_factory)[0]

    hosted_client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        listed = hosted_client.get(f"{API_ROOT}/organizations/me/keys")
        rotated = hosted_client.post(f"{API_ROOT}/organizations/me/keys/{minted.id}/rotate")
    finally:
        hosted_client.cookies.clear()

    assert listed.status_code == status.HTTP_200_OK
    assert [key["id"] for key in listed.json()] == []
    # Rotation is the operation that would break it silently: a new hash with the
    # stored plaintext left behind is a Playground that stops working and says
    # nothing. The key is out of reach of that surface, so it answers 404.
    assert rotated.status_code == status.HTTP_404_NOT_FOUND, rotated.text


def test_a_deployment_with_no_data_plane_refuses_before_minting_anything(
    hosted_client_without_a_data_plane: TestClient,
    caller: tuple[uuid.UUID, uuid.UUID, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Nowhere to forward to is a deployment problem, said as one.

    The surface is withheld from the bootstrap in this state, so the page is not
    offered; a caller driving the API directly still gets an answer that names
    the missing configuration rather than a connection error.
    """
    _, workspace_id, token = caller

    response = _send(hosted_client_without_a_data_plane, token, workspace_id)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert _internal_keys(db_session_factory) == []


def test_an_unreachable_data_plane_answers_502_without_naming_it(
    hosted_client: TestClient,
    caller: tuple[uuid.UUID, uuid.UUID, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An upstream outage is an upstream failure, and its address is not public.

    The transport error names this deployment's own hosts and ports, which is
    topology a tenant is not owed and the error-detail boundary does not carry.
    """
    _, workspace_id, token = caller

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 111] Connection refused")

    def build(**kwargs: object) -> httpx.AsyncClient:
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", build)

    response = _send(hosted_client, token, workspace_id)

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    detail = response.json()["detail"]
    assert "gateway.example.test" not in detail
    assert "Connection refused" not in detail


def test_the_gateways_own_refusal_reaches_the_page(
    hosted_client: TestClient,
    caller: tuple[uuid.UUID, uuid.UUID, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget refusal is the gateway's answer, forwarded rather than restated."""
    _, workspace_id, token = caller
    _answer_from_the_data_plane(
        monkeypatch,
        {},
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        payload=b'{"detail": "Budget exceeded"}',
    )

    response = _send(hosted_client, token, workspace_id)

    assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED
    assert response.json()["detail"] == "Budget exceeded"


def test_a_workspace_the_caller_is_not_in_is_refused_before_any_forward(
    hosted_client: TestClient,
    caller: tuple[uuid.UUID, uuid.UUID, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forwarding does not relax the membership check it forwards on behalf of.

    The key is what tells the far side whose budget to spend, so a workspace the
    caller does not belong to must never reach the point of minting one.
    """
    _, _workspace_id, token = caller
    seen: dict[str, object] = {}
    _answer_from_the_data_plane(monkeypatch, seen)

    response = _send(hosted_client, token, uuid.uuid4())

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert seen == {}
    assert _internal_keys(db_session_factory) == []


# What the control plane's own usage rows are labelled with. Every report a
# gateway sends lands under one fixed label (the overlay's ``USAGE_LOG_ENDPOINT``
# in ``mozilla-ai/otari-ai``), because the surface that made the call is not on
# the wire. That is the point of the two tests below: the label cannot tell a
# Playground message from an SDK call, and the credential can.
_REPORTED_BY_A_GATEWAY = "attached-gateway"


def _record_gateway_usage(
    db_session_factory: Callable[[], Session],
    *,
    workspace_id: uuid.UUID,
    user_id: str | None,
    api_key_id: str | None,
) -> None:
    """Write the usage row a hosted deployment's gateway report produces."""
    session = db_session_factory()
    try:
        session.add(
            UsageLog(
                workspace_id=workspace_id,
                api_key_id=api_key_id,
                user_id=user_id,
                timestamp=datetime.now(UTC),
                model="openai:gpt-4o",
                provider="openai",
                endpoint=_REPORTED_BY_A_GATEWAY,
                source=SERVED_HERE_SLUG,
                status="success",
            )
        )
        session.commit()
    finally:
        session.close()


def _activation(client: TestClient, token: str, workspace_id: uuid.UUID) -> dict[str, object]:
    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        response = client.get(f"{API_ROOT}/workspaces/{workspace_id}/activation")
    finally:
        client.cookies.clear()
    assert response.status_code == status.HTTP_200_OK, response.text
    body: dict[str, object] = response.json()
    return body


def test_a_forwarded_playground_message_does_not_close_the_activation_guide(
    hosted_client: TestClient,
    caller: tuple[uuid.UUID, uuid.UUID, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guide marks somebody integrating Otari, which our own page is not.

    Standalone catches this on the endpoint label, because it wrote the row
    itself. Hosted cannot: the row comes from the gateway's usage report, under
    the one label every report from that gateway carries. The dispatch key is
    what carries the distinction instead, and it is minted by this deployment for
    this purpose alone, so it is not a claim the caller could make.
    """
    _, workspace_id, token = caller
    _answer_from_the_data_plane(monkeypatch, {})
    _send(hosted_client, token, workspace_id)
    dispatch_key = _internal_keys(db_session_factory)[0]

    assert _activation(hosted_client, token, workspace_id)["status"] == "waiting"

    _record_gateway_usage(
        db_session_factory,
        workspace_id=workspace_id,
        user_id=dispatch_key.user_id,
        api_key_id=dispatch_key.id,
    )

    after = _activation(hosted_client, token, workspace_id)
    assert after["status"] == "waiting"
    assert after["activation_attempt"] is None
    # Nor the "your last attempt" line, which would otherwise report the product
    # talking to itself as the caller's most recent try.
    assert after["latest_attempt"] is None


def test_a_real_gateway_request_still_closes_it(
    hosted_client: TestClient,
    caller: tuple[uuid.UUID, uuid.UUID, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other side of the same predicate, so the exclusion cannot swallow the feature.

    An identical row under a key somebody made is the milestone the guide exists
    for, and it still closes it.
    """
    _, workspace_id, token = caller
    _answer_from_the_data_plane(monkeypatch, {})
    _send(hosted_client, token, workspace_id)
    dispatch_key = _internal_keys(db_session_factory)[0]

    session = db_session_factory()
    try:
        own_key = APIKey(
            id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            key_hash="a-key-somebody-made",
            key_name="My SDK key",
            user_id=dispatch_key.user_id,
        )
        session.add(own_key)
        session.commit()
        own_key_id = own_key.id
    finally:
        session.close()

    _record_gateway_usage(
        db_session_factory,
        workspace_id=workspace_id,
        user_id=dispatch_key.user_id,
        api_key_id=own_key_id,
    )

    after = _activation(hosted_client, token, workspace_id)
    assert after["status"] != "waiting"
    assert after["activation_attempt"] is not None
