"""The guardrail-definition routes over HTTP: the gate, the statuses, and the wiring.

The rules and the storage are covered at the service layer in
`test_organization_guardrail_definition_service.py`. What is here is the composition in
`api/deps.py`, which is new for this surface: the service is built on the
request's Unit of Work and takes the session only for the role gate.

Every case acts as the bootstrap operator, who is the owner of a standalone
deployment's one organization and therefore the identity a dashboard form
actually arrives as. A plain member is not reachable through the API, which is
why that case lives beside the service.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT
from gateway.services.secret_box import generate_secret_key


@pytest.fixture
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Not autouse, because one case below is about the key being absent."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


def test_the_surface_needs_a_credential(client: TestClient) -> None:
    """Anonymous and a bad bearer alike, before any role question is asked."""
    for headers in ({}, {"Authorization": "Bearer not-a-key"}):
        assert client.get(f"{API_ROOT}/organizations/me/guardrail-definitions", headers=headers).status_code == 401


def test_crud_over_http(client: TestClient, master_key_header: dict[str, str], _secret_key: None) -> None:
    """The wiring in `api/deps.py`: one Unit of Work over the request's session.

    The bootstrap operator is the owner of a standalone deployment's one
    organization, so this is the identity a dashboard form actually arrives as.
    """
    created = client.post(
        f"{API_ROOT}/organizations/me/guardrail-definitions",
        json={
            "name": "prod-lakera",
            "guardrail_name": "lakera_guard",
            "create_kwargs": {"api_key": "lakera-key", "endpoint": "https://api.lakera.ai"},
        },
        headers=master_key_header,
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["create_kwargs"] == {"endpoint": "https://api.lakera.ai"}
    assert body["create_secrets"] == {"api_key": "***"}
    assert "lakera-key" not in created.text

    listed = client.get(f"{API_ROOT}/organizations/me/guardrail-definitions", headers=master_key_header)
    assert listed.status_code == 200
    assert listed.json()["count"] == 1

    patched = client.patch(
        f"{API_ROOT}/organizations/me/guardrail-definitions/{body['id']}",
        json={"enabled": False},
        headers=master_key_header,
    )
    assert patched.status_code == 200
    assert patched.json()["enabled"] is False
    assert patched.json()["create_secrets"] == {"api_key": "***"}, "an edit elsewhere keeps the credential"

    deleted = client.delete(
        f"{API_ROOT}/organizations/me/guardrail-definitions/{body['id']}", headers=master_key_header
    )
    assert deleted.status_code == 200
    remaining = client.get(f"{API_ROOT}/organizations/me/guardrail-definitions", headers=master_key_header)
    assert remaining.json()["count"] == 0


def test_a_refused_definition_answers_400_with_the_reason(
    client: TestClient, master_key_header: dict[str, str], _secret_key: None
) -> None:
    """The domain errors carry their own statuses, so the route catches nothing.

    Each message names the parameter or the guardrail, because a form has to say
    which field to fix.
    """
    unknown = client.post(
        f"{API_ROOT}/organizations/me/guardrail-definitions",
        json={"name": "nope", "guardrail_name": "not_a_guardrail"},
        headers=master_key_header,
    )
    assert unknown.status_code == 400
    assert "built-in guardrail catalog" in unknown.json()["detail"]

    unmetered = client.post(
        f"{API_ROOT}/organizations/me/guardrail-definitions",
        json={"name": "judge", "guardrail_name": "any_llm"},
        headers=master_key_header,
    )
    assert unmetered.status_code == 400
    assert "metering" in unmetered.json()["detail"]

    live_object = client.post(
        f"{API_ROOT}/organizations/me/guardrail-definitions",
        json={
            "name": "bedrock",
            "guardrail_name": "bedrock_guardrails",
            "create_kwargs": {"guardrail_identifier": "gr-1", "boto3_session": {"profile": "default"}},
        },
        headers=master_key_header,
    )
    assert live_object.status_code == 400
    assert "aws_access_key_id" in live_object.json()["detail"]


def test_storing_a_credential_without_a_secret_key_blames_the_deployment(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 500, not the 400 a bad body would get, and the body says nothing more.

    The caller sent a well-formed definition and cannot configure
    ``OTARI_SECRET_KEY``. Blaming them would also keep the condition out of the
    5xx alerting that reaches the people who can fix it. A definition with no
    secret in it still saves, because nothing needed encrypting.
    """
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)

    with_secret = client.post(
        f"{API_ROOT}/organizations/me/guardrail-definitions",
        json={
            "name": "prod-lakera",
            "guardrail_name": "lakera_guard",
            "create_kwargs": {"api_key": "lakera-key"},
        },
        headers=master_key_header,
    )
    assert with_secret.status_code == 500
    assert "lakera-key" not in with_secret.text

    without_secret = client.post(
        f"{API_ROOT}/organizations/me/guardrail-definitions",
        json={
            "name": "env-lakera",
            "guardrail_name": "lakera_guard",
            "create_kwargs": {"endpoint": "https://api.lakera.ai"},
        },
        headers=master_key_header,
    )
    assert without_secret.status_code == 201, without_secret.text
    assert without_secret.json()["create_secrets"] == {}
