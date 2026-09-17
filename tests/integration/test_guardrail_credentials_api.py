"""Integration tests for the /api/v1/guardrail-credentials CRUD endpoints.

A guardrail profile used to be a key in a YAML file inside the guardrails
container, so the picker at GET /tool-settings/guardrails/catalog had nowhere to
write to. These cover the route in: credentials are write-only, a definition is
held to what the catalog says its guardrail accepts, and an editor that echoes
the mask back keeps the key it was never shown.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.secret_box import generate_secret_key

_LAKERA_KEY = "lak-live-notreal-9876"
_ENDPOINT = "https://api.lakera.ai/v2/guard"


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
    )


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


def _create(client: TestClient, headers: dict[str, str], **body: Any) -> Any:
    payload: dict[str, Any] = {
        "name": "prompt-injection",
        "guardrail_name": "lakera_guard",
        "create_kwargs": {"api_key": _LAKERA_KEY, "endpoint": _ENDPOINT},
        **body,
    }
    return client.post(f"{API_ROOT}/guardrail-credentials", json=payload, headers=headers)


def test_requires_master_key(client: TestClient) -> None:
    assert client.get(f"{API_ROOT}/guardrail-credentials").status_code == 401
    assert client.post(f"{API_ROOT}/guardrail-credentials", json={}).status_code == 401
    assert client.get(f"{API_ROOT}/guardrail-credentials/x").status_code == 401
    assert client.patch(f"{API_ROOT}/guardrail-credentials/x", json={}).status_code == 401
    assert client.delete(f"{API_ROOT}/guardrail-credentials/x").status_code == 401
    assert client.post(f"{API_ROOT}/guardrail-credentials/reencrypt").status_code == 401


def test_create_lists_and_never_returns_the_credential(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The secret goes in, its name comes back, and the value never does."""
    resp = _create(client, master_key_header)
    assert resp.status_code == 201, resp.text
    body = resp.json()

    assert body["guardrail_name"] == "lakera_guard"
    assert body["create_kwargs"] == {"endpoint": _ENDPOINT}
    assert body["create_secrets"] == {"api_key": "***"}
    assert body["enabled"] is True
    assert body["decryptable"] is True
    assert _LAKERA_KEY not in resp.text

    listed = client.get(f"{API_ROOT}/guardrail-credentials", headers=master_key_header)
    assert listed.status_code == 200
    assert [row["name"] for row in listed.json()] == ["prompt-injection"]
    assert _LAKERA_KEY not in listed.text

    one = client.get(f"{API_ROOT}/guardrail-credentials/prompt-injection", headers=master_key_header)
    assert one.status_code == 200
    assert one.json()["create_secrets"] == {"api_key": "***"}
    assert _LAKERA_KEY not in one.text


def test_the_credential_never_reaches_a_log_line(
    client: TestClient, master_key_header: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("DEBUG"):
        assert _create(client, master_key_header).status_code == 201

    assert _LAKERA_KEY not in caplog.text


def test_echoing_the_mask_back_keeps_the_stored_credential(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """How the dashboard form saves: it loads a row, edits one field, submits all of it.

    It was never shown the key, so it sends the mask for it. Taking that
    literally would overwrite a live credential with three asterisks.
    """
    assert _create(client, master_key_header).status_code == 201

    resp = client.patch(
        f"{API_ROOT}/guardrail-credentials/prompt-injection",
        json={"create_kwargs": {"api_key": "***", "endpoint": "https://guard.example.invalid"}},
        headers=master_key_header,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["create_kwargs"] == {"endpoint": "https://guard.example.invalid"}
    assert resp.json()["create_secrets"] == {"api_key": "***"}

    # The stored key is still the original one, which only a re-encryption pass
    # can show from the outside: it reports one readable map rather than none.
    rotated = client.post(f"{API_ROOT}/guardrail-credentials/reencrypt", headers=master_key_header)
    assert rotated.json() == {"reencrypted": 1, "unreadable": 0}


def test_a_credential_left_out_is_cleared(client: TestClient, master_key_header: dict[str, str]) -> None:
    """``create_kwargs`` replaces the whole map, so omission means removal."""
    assert _create(client, master_key_header).status_code == 201

    resp = client.patch(
        f"{API_ROOT}/guardrail-credentials/prompt-injection",
        json={"create_kwargs": {"endpoint": _ENDPOINT}},
        headers=master_key_header,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["create_secrets"] == {}


def test_changing_the_guardrail_resplits_the_stored_arguments(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """``endpoint`` is plain for Lakera and plain for Alinia; ``api_key`` stays secret.

    The point is that the split is redone under the new class rather than
    carried over, so it can never be left classified by a guardrail the row no
    longer names.
    """
    assert _create(client, master_key_header).status_code == 201

    resp = client.patch(
        f"{API_ROOT}/guardrail-credentials/prompt-injection",
        json={"guardrail_name": "alinia", "create_kwargs": {"api_key": "ali-1", "endpoint": _ENDPOINT}},
        headers=master_key_header,
    )
    # Alinia additionally requires detection_config, which nothing supplies.
    assert resp.status_code == 400
    assert "detection_config" in resp.json()["detail"]

    resp = client.patch(
        f"{API_ROOT}/guardrail-credentials/prompt-injection",
        json={
            "guardrail_name": "alinia",
            "create_kwargs": {"api_key": "ali-1", "endpoint": _ENDPOINT, "detection_config": {"a": 1}},
        },
        headers=master_key_header,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["guardrail_name"] == "alinia"
    assert resp.json()["create_kwargs"] == {"endpoint": _ENDPOINT, "detection_config": {"a": 1}}
    assert resp.json()["create_secrets"] == {"api_key": "***"}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"guardrail_name": "not_a_guardrail"}, "not a guardrail this gateway can run"),
        ({"guardrail_name": "llama_guard"}, "not a guardrail this gateway can run"),
        ({"create_kwargs": {"api_key": "k", "nope": 1}}, "no create argument"),
        (
            {
                "guardrail_name": "bedrock_guardrails",
                "create_kwargs": {"guardrail_identifier": "gr-1", "boto3_session": {}},
            },
            "live object that cannot be stored",
        ),
        (
            {"guardrail_name": "alinia", "create_kwargs": {"api_key": "k", "endpoint": _ENDPOINT}},
            "detection_config",
        ),
    ],
)
def test_a_definition_the_catalog_refuses_is_a_400(
    client: TestClient, master_key_header: dict[str, str], body: dict[str, Any], expected: str
) -> None:
    resp = _create(client, master_key_header, **body)

    assert resp.status_code == 400, resp.text
    assert expected in resp.json()["detail"]
    assert client.get(f"{API_ROOT}/guardrail-credentials", headers=master_key_header).json() == []


def test_a_name_that_is_not_one_path_segment_is_refused(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A stored '/' would be a row no route could address again.

    Neither ``/guardrail-credentials/team/prompt`` nor the ``%2F`` spelling
    reaches ``/{name}``, so the row would be readable, editable and deletable by
    nobody. Refusing at write time is the only place that can be prevented.
    """
    assert _create(client, master_key_header, name="team/prompt").status_code == 422
    assert client.get(f"{API_ROOT}/guardrail-credentials", headers=master_key_header).json() == []


def test_a_duplicate_name_is_a_409(client: TestClient, master_key_header: dict[str, str]) -> None:
    assert _create(client, master_key_header).status_code == 201

    resp = _create(client, master_key_header)
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


def test_an_unknown_name_is_a_404(client: TestClient, master_key_header: dict[str, str]) -> None:
    assert client.get(f"{API_ROOT}/guardrail-credentials/nope", headers=master_key_header).status_code == 404
    assert (
        client.patch(f"{API_ROOT}/guardrail-credentials/nope", json={}, headers=master_key_header).status_code == 404
    )
    assert client.delete(f"{API_ROOT}/guardrail-credentials/nope", headers=master_key_header).status_code == 404


def test_a_stale_expected_updated_at_is_a_412(client: TestClient, master_key_header: dict[str, str]) -> None:
    assert _create(client, master_key_header).status_code == 201

    resp = client.patch(
        f"{API_ROOT}/guardrail-credentials/prompt-injection",
        json={"enabled": False, "expected_updated_at": "2020-01-01T00:00:00+00:00"},
        headers=master_key_header,
    )
    assert resp.status_code == 412


def test_a_matching_expected_updated_at_succeeds(client: TestClient, master_key_header: dict[str, str]) -> None:
    created = _create(client, master_key_header).json()

    resp = client.patch(
        f"{API_ROOT}/guardrail-credentials/prompt-injection",
        json={"enabled": False, "expected_updated_at": created["updated_at"]},
        headers=master_key_header,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["enabled"] is False
    # Disabling keeps the credential, which is the whole point of the flag.
    assert resp.json()["create_secrets"] == {"api_key": "***"}


def test_delete_removes_it(client: TestClient, master_key_header: dict[str, str]) -> None:
    assert _create(client, master_key_header).status_code == 201

    path = f"{API_ROOT}/guardrail-credentials/prompt-injection"
    assert client.delete(path, headers=master_key_header).status_code == 204
    assert client.get(path, headers=master_key_header).status_code == 404


def test_an_unreadable_credential_is_flagged_rather_than_hidden(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key the deployment no longer has must not take the whole listing down.

    The operator is the one person who can fix it, so the row is listed with no
    credential names and ``decryptable: false``, and a re-encryption pass counts
    it as unreadable rather than silently rewriting it.
    """
    assert _create(client, master_key_header).status_code == 201

    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    listed = client.get(f"{API_ROOT}/guardrail-credentials", headers=master_key_header)
    assert listed.status_code == 200
    assert listed.json()[0]["decryptable"] is False
    assert listed.json()[0]["create_secrets"] == {}
    assert listed.json()[0]["create_kwargs"] == {"endpoint": _ENDPOINT}

    assert client.post(f"{API_ROOT}/guardrail-credentials/reencrypt", headers=master_key_header).json() == {
        "reencrypted": 0,
        "unreadable": 1,
    }


def test_a_row_whose_key_is_gone_can_still_be_disabled(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling touches neither the arguments nor the guardrail, so it reads no secret.

    ``enabled`` exists to stop a guardrail without losing the configuration it
    took to set up, which a lost key must not take away.
    """
    assert _create(client, master_key_header).status_code == 201

    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    patched = client.patch(
        f"{API_ROOT}/guardrail-credentials/prompt-injection", json={"enabled": False}, headers=master_key_header
    )
    assert patched.status_code == 200
    assert patched.json()["enabled"] is False
    # The unreadable map was carried over rather than rewritten under the new key.
    assert patched.json()["decryptable"] is False


def test_resending_the_credentials_repairs_a_row_whose_key_is_gone(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented recovery: a replacement carrying no mask needs no old key."""
    assert _create(client, master_key_header).status_code == 201

    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    patched = client.patch(
        f"{API_ROOT}/guardrail-credentials/prompt-injection",
        json={"create_kwargs": {"api_key": "lak-live-notreal-0001", "endpoint": _ENDPOINT}},
        headers=master_key_header,
    )
    assert patched.status_code == 200
    assert patched.json()["decryptable"] is True
    assert patched.json()["create_secrets"] == {"api_key": "***"}


def test_a_mask_with_no_readable_value_behind_it_is_refused(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``***`` stands for the stored value, so it needs the key that wrote it."""
    assert _create(client, master_key_header).status_code == 201

    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    refused = client.patch(
        f"{API_ROOT}/guardrail-credentials/prompt-injection",
        json={"create_kwargs": {"api_key": "***", "endpoint": _ENDPOINT}},
        headers=master_key_header,
    )
    assert refused.status_code == 400


def test_a_rotation_reencrypts_under_the_new_primary_key(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``OTARI_SECRET_KEY="new,old"`` is the documented rotation; the pass moves the row."""
    assert _create(client, master_key_header).status_code == 201

    old = generate_secret_key()
    monkeypatch.setenv("OTARI_SECRET_KEY", old)
    assert _create(client, master_key_header, name="second").status_code == 201

    new = generate_secret_key()
    monkeypatch.setenv("OTARI_SECRET_KEY", f"{new},{old}")

    # One row was written under a key that is no longer in the set at all.
    assert client.post(f"{API_ROOT}/guardrail-credentials/reencrypt", headers=master_key_header).json() == {
        "reencrypted": 1,
        "unreadable": 1,
    }

    monkeypatch.setenv("OTARI_SECRET_KEY", new)
    listed = {row["name"]: row for row in client.get(
        f"{API_ROOT}/guardrail-credentials", headers=master_key_header
    ).json()}
    assert listed["second"]["decryptable"] is True
    assert listed["prompt-injection"]["decryptable"] is False


def test_a_refused_commit_is_a_500_and_not_a_crash(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every write runs inside one rollback block, so every write answers the same way.

    The detail says only "Database error", as the sibling stores' does: what the
    database objected to is not something a response may carry.
    """
    assert _create(client, master_key_header).status_code == 201

    async def _refuse(*_args: object, **_kwargs: object) -> None:
        raise SQLAlchemyError("refused")

    monkeypatch.setattr("sqlalchemy.ext.asyncio.AsyncSession.commit", _refuse)

    created = _create(client, master_key_header, name="second")
    patched = client.patch(
        f"{API_ROOT}/guardrail-credentials/prompt-injection", json={"enabled": False}, headers=master_key_header
    )
    deleted = client.delete(f"{API_ROOT}/guardrail-credentials/prompt-injection", headers=master_key_header)
    reencrypted = client.post(f"{API_ROOT}/guardrail-credentials/reencrypt", headers=master_key_header)

    for response in (created, patched, deleted, reencrypted):
        assert response.status_code == 500
        assert response.json()["detail"] == "Database error"


def test_a_refused_flush_is_a_500_too(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repository call flushes, so the insert can fail before a commit is reached.

    The rollback has to cover that window as well, or the route answers from a
    session still holding the failed insert.
    """

    async def _refuse(*_args: object, **_kwargs: object) -> None:
        raise SQLAlchemyError("refused")

    monkeypatch.setattr("sqlalchemy.ext.asyncio.AsyncSession.flush", _refuse)

    created = _create(client, master_key_header)
    assert created.status_code == 500
    assert created.json()["detail"] == "Database error"


def test_a_guardrail_with_no_credential_stores_fine(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """``any_llm`` takes no constructor arguments at all, so the map is empty."""
    resp = _create(
        client,
        master_key_header,
        name="judge",
        guardrail_name="any_llm",
        create_kwargs={},
        validate_kwargs={"policy": "no medical advice"},
    )

    assert resp.status_code == 201, resp.text
    assert resp.json()["create_secrets"] == {}
    assert resp.json()["validate_kwargs"] == {"policy": "no medical advice"}
