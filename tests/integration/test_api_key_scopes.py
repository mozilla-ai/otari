"""An API key reaches the endpoint groups its scopes name, and nothing else.

Written against what a leaked key could do: each test holds a key to fewer
scopes than its owner's access and asserts the refusal, then that the master key
and the open catalog reads are unaffected. The route declarations themselves are
covered by ``tests/unit/test_key_scope_declarations.py``.
"""

from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT

from .conftest import ALL_SERVICE_KEY_SCOPES

_KEYS = f"{API_ROOT}/keys"


def _create(client: TestClient, master_key_header: dict[str, str], **body: Any) -> dict[str, Any]:
    created = client.post(_KEYS, json={"key_name": "scoped", **body}, headers=master_key_header)
    assert created.status_code == status.HTTP_200_OK, created.text
    key: dict[str, Any] = created.json()
    return key


def _header(key: dict[str, Any]) -> dict[str, str]:
    return {API_KEY_HEADER: f"Bearer {key['key']}"}


def _refusal(response: Any) -> str:
    """The message of a 403, in the error shape of whichever API answered it."""
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.text
    body = response.json()
    message: str = body["detail"] if "detail" in body else body["error"]["message"]
    return message


def test_a_key_created_without_scopes_calls_models_and_nothing_else(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key = _create(client, master_key_header)
    headers = _header(key)

    assert key["scopes"] == ["inference"]
    assert "'usage:read'" in _refusal(client.get(f"{API_ROOT}/key/budget", headers=headers))
    # The MCP routes answer every refusal with one fixed message per code, so this one names no scope.
    mcp = client.get(f"{API_ROOT}/mcp/servers", headers=headers)
    assert (mcp.status_code, mcp.json()["code"]) == (status.HTTP_403_FORBIDDEN, "forbidden"), mcp.text
    assert "'telemetry:write'" in _refusal(client.post("/otlp/v1/logs", json={}, headers=headers))
    assert "'usage:write'" in _refusal(
        client.post(f"{API_ROOT}/usage/external-events", json={"events": []}, headers=headers)
    )


def test_a_key_reaches_only_the_groups_it_holds(client: TestClient, master_key_header: dict[str, str]) -> None:
    headers = _header(_create(client, master_key_header, scopes=["usage:read"]))
    chat = {"model": "openai:gpt-4o-mini", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}

    assert client.get(f"{API_ROOT}/key/budget", headers=headers).status_code == status.HTTP_200_OK
    assert "'inference'" in _refusal(client.post(f"{API_ROOT}/chat/completions", json=chat, headers=headers))
    assert "'inference'" in _refusal(client.post(f"{API_ROOT}/messages", json=chat, headers=headers))
    assert "'inference'" in _refusal(client.post(f"{API_ROOT}/embeddings", json={}, headers=headers))
    assert "'web'" in _refusal(client.post(f"{API_ROOT}/search", json={"query": "q"}, headers=headers))
    assert "'code:execute'" in _refusal(client.post(f"{API_ROOT}/code/execute", json={}, headers=headers))
    assert "'agents'" in _refusal(client.post(f"{API_ROOT}/hooks/check", json={}, headers=headers))


def test_the_catalog_reads_are_open_to_every_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    headers = _header(_create(client, master_key_header, scopes=["usage:read"]))

    for path in ("/models", "/tools", "/pricing", "/providers/catalog"):
        assert client.get(f"{API_ROOT}{path}", headers=headers).status_code == status.HTTP_200_OK, path


def test_the_master_key_holds_every_scope(client: TestClient, master_key_header: dict[str, str]) -> None:
    looked_up = client.get(f"{API_ROOT}/usage/requests/no-such-request", headers=master_key_header)

    assert looked_up.status_code == status.HTTP_404_NOT_FOUND, looked_up.text


def test_scopes_are_stored_once_each_in_one_order(client: TestClient, master_key_header: dict[str, str]) -> None:
    key = _create(client, master_key_header, scopes=["usage:read", "inference", "usage:read"])

    assert key["scopes"] == ["inference", "usage:read"]
    listed = client.get(f"{_KEYS}/{key['id']}", headers=master_key_header).json()
    assert listed["scopes"] == ["inference", "usage:read"]


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"scopes": []}, status.HTTP_422_UNPROCESSABLE_CONTENT),
        ({"scopes": ["keys:write"]}, status.HTTP_422_UNPROCESSABLE_CONTENT),
        ({"scopes": ["inference", "end_users:read"]}, status.HTTP_400_BAD_REQUEST),
    ],
    ids=["none", "unknown", "end-users-on-a-plain-key"],
)
def test_a_key_cannot_be_created_with_scopes_it_cannot_hold(
    client: TestClient, master_key_header: dict[str, str], body: dict[str, Any], code: int
) -> None:
    refused = client.post(_KEYS, json={"key_name": "bad", **body}, headers=master_key_header)

    assert refused.status_code == code, refused.text


def test_narrowing_a_key_takes_effect_on_its_next_request(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key = _create(client, master_key_header, scopes=["inference", "usage:read"])
    headers = _header(key)
    assert client.get(f"{API_ROOT}/key/budget", headers=headers).status_code == status.HTTP_200_OK

    narrowed = client.patch(f"{_KEYS}/{key['id']}", json={"scopes": ["inference"]}, headers=master_key_header)

    assert narrowed.status_code == status.HTTP_200_OK, narrowed.text
    assert narrowed.json()["scopes"] == ["inference"]
    assert "'usage:read'" in _refusal(client.get(f"{API_ROOT}/key/budget", headers=headers))

    widened = client.patch(f"{_KEYS}/{key['id']}", json={"scopes": ["usage:read"]}, headers=master_key_header)
    assert widened.status_code == status.HTTP_200_OK, widened.text
    assert client.get(f"{API_ROOT}/key/budget", headers=headers).status_code == status.HTTP_200_OK


@pytest.mark.parametrize("body", [{"key_name": "renamed"}, {"scopes": None}], ids=["absent", "null"])
def test_an_update_that_names_no_scopes_leaves_them(
    client: TestClient, master_key_header: dict[str, str], body: dict[str, Any]
) -> None:
    key = _create(client, master_key_header, scopes=["inference", "usage:read"])

    updated = client.patch(f"{_KEYS}/{key['id']}", json=body, headers=master_key_header)

    assert updated.status_code == status.HTTP_200_OK, updated.text
    assert updated.json()["scopes"] == ["inference", "usage:read"]


def test_an_update_cannot_give_a_plain_key_an_end_user_scope(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key = _create(client, master_key_header)

    refused = client.patch(f"{_KEYS}/{key['id']}", json={"scopes": ["end_users:read"]}, headers=master_key_header)

    assert refused.status_code == status.HTTP_400_BAD_REQUEST, refused.text
    assert "Only a service key" in refused.json()["detail"]


def test_a_service_key_deletes_end_users_only_with_the_delete_scope(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Deleting erases an end user's records, so a key that manages end users does not get it by default."""
    key = _create(
        client, master_key_header, user_id="svc", is_service_key=True, scopes=["end_users:read", "end_users:write"]
    )
    headers = _header(key)

    assert client.get(f"{API_ROOT}/end-users", headers=headers).status_code == status.HTTP_200_OK
    assert "'end_users:delete'" in _refusal(client.delete(f"{API_ROOT}/end-users/alice", headers=headers))
    assert "'end_users:delete'" in _refusal(
        client.delete(f"{API_ROOT}/end-users?created_before=2999-01-01T00:00:00Z", headers=headers)
    )

    granted = client.patch(f"{_KEYS}/{key['id']}", json={"scopes": ALL_SERVICE_KEY_SCOPES}, headers=master_key_header)
    assert granted.status_code == status.HTTP_200_OK, granted.text
    deleted = client.delete(f"{API_ROOT}/end-users/alice", headers=headers)
    assert deleted.status_code != status.HTTP_403_FORBIDDEN, deleted.text


def test_a_plain_key_hears_that_the_end_user_routes_need_a_service_key(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    headers = _header(_create(client, master_key_header))

    assert _refusal(client.get(f"{API_ROOT}/end-users", headers=headers)) == "This endpoint requires a service key."


def test_a_refusal_names_no_secret(client: TestClient, master_key_header: dict[str, str]) -> None:
    key = _create(client, master_key_header)

    refused = client.get(f"{API_ROOT}/key/budget", headers=_header(key))

    assert key["key"] not in refused.text


def test_one_update_makes_a_key_a_service_key_and_grants_it_the_end_user_scopes(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key = _create(client, master_key_header, user_id="svc")

    updated = client.patch(
        f"{_KEYS}/{key['id']}",
        json={"is_service_key": True, "scopes": ["inference", "end_users:read"]},
        headers=master_key_header,
    )

    assert updated.status_code == status.HTTP_200_OK, updated.text
    assert updated.json()["scopes"] == ["inference", "end_users:read"]
    assert client.get(f"{API_ROOT}/end-users", headers=_header(key)).status_code == status.HTTP_200_OK


def test_a_key_that_stops_being_a_service_key_loses_its_end_user_scopes(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    key = _create(client, master_key_header, user_id="svc", is_service_key=True, scopes=ALL_SERVICE_KEY_SCOPES)

    updated = client.patch(f"{_KEYS}/{key['id']}", json={"is_service_key": False}, headers=master_key_header)

    assert updated.status_code == status.HTTP_200_OK, updated.text
    assert not any(scope.startswith("end_users:") for scope in updated.json()["scopes"])


def test_rotating_a_key_keeps_its_scopes(client: TestClient, master_key_header: dict[str, str]) -> None:
    key = _create(client, master_key_header, scopes=["inference", "usage:read"])

    rotated = client.post(f"{_KEYS}/{key['id']}/rotate", headers=master_key_header)

    assert rotated.status_code == status.HTTP_200_OK, rotated.text
    assert rotated.json()["scopes"] == ["inference", "usage:read"]
    assert client.get(f"{API_ROOT}/key/budget", headers=_header(rotated.json())).status_code == status.HTTP_200_OK
