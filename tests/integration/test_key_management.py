from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, GatewayConfig

from .conftest import MODEL_NAME


def test_create_api_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating an API key."""
    response = client.post(
        "/v1/keys",
        json={"key_name": "test-key"},
        headers=master_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert "id" in data
    assert "key" in data
    assert data["key"].startswith("gw-")
    # The prefix is the leading slice of the plaintext, echoed for the show-once
    # reveal so the list can later fingerprint the key without the full secret.
    assert data["key_prefix"] == data["key"][:10]
    assert data["key_name"] == "test-key"
    assert data["is_active"] is True
    assert "created_at" in data


def test_create_api_key_with_expiration(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating an API key with expiration."""
    expires_at = (datetime.now(UTC) + timedelta(days=30)).isoformat()

    response = client.post(
        "/v1/keys",
        json={"key_name": "expiring-key", "expires_at": expires_at},
        headers=master_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["expires_at"] is not None


def test_create_api_key_with_metadata(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating an API key with metadata."""
    metadata = {"team": "engineering", "environment": "production"}

    response = client.post(
        "/v1/keys",
        json={"key_name": "metadata-key", "metadata": metadata},
        headers=master_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["metadata"] == metadata


def test_create_api_key_without_master_key_fails(client: TestClient) -> None:
    """Test that creating key without master key fails."""
    response = client.post(
        "/v1/keys",
        json={"key_name": "test-key"},
    )

    assert response.status_code in [401, 422]


def test_list_api_keys(client: TestClient, master_key_header: dict[str, str], api_key_obj: dict[str, Any]) -> None:
    """Test listing API keys."""
    response = client.get("/v1/keys", headers=master_key_header)

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) > 0
    assert any(key["id"] == api_key_obj["id"] for key in data)


def test_list_api_keys_pagination(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test API key listing with pagination."""
    for i in range(5):
        client.post(
            "/v1/keys",
            json={"key_name": f"key-{i}"},
            headers=master_key_header,
        )

    response = client.get("/v1/keys?skip=0&limit=3", headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 3


def test_get_api_key(client: TestClient, master_key_header: dict[str, str], api_key_obj: dict[str, Any]) -> None:
    """Test getting specific API key details."""
    response = client.get(f"/v1/keys/{api_key_obj['id']}", headers=master_key_header)

    assert response.status_code == 200
    data = response.json()

    assert data["id"] == api_key_obj["id"]
    assert data["key_name"] == api_key_obj["key_name"]
    # The fingerprint is listed; the full key never is.
    assert "key_prefix" in data
    assert "key" not in data


def test_get_nonexistent_api_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test getting non-existent API key returns 404."""
    response = client.get("/v1/keys/nonexistent-id", headers=master_key_header)

    assert response.status_code == 404


def test_update_api_key(client: TestClient, master_key_header: dict[str, str], api_key_obj: dict[str, Any]) -> None:
    """Test updating an API key."""
    response = client.patch(
        f"/v1/keys/{api_key_obj['id']}",
        json={"key_name": "updated-key", "is_active": False},
        headers=master_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["key_name"] == "updated-key"
    assert data["is_active"] is False


def test_update_api_key_metadata(
    client: TestClient, master_key_header: dict[str, str], api_key_obj: dict[str, Any]
) -> None:
    """Test updating API key metadata."""
    new_metadata = {"updated": True, "version": 2}

    response = client.patch(
        f"/v1/keys/{api_key_obj['id']}",
        json={"metadata": new_metadata},
        headers=master_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["metadata"] == new_metadata


def test_delete_api_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test deleting an API key."""
    create_response = client.post(
        "/v1/keys",
        json={"key_name": "delete-me"},
        headers=master_key_header,
    )
    key_id = create_response.json()["id"]

    delete_response = client.delete(f"/v1/keys/{key_id}", headers=master_key_header)
    assert delete_response.status_code == 204

    get_response = client.get(f"/v1/keys/{key_id}", headers=master_key_header)
    assert get_response.status_code == 404


def test_delete_nonexistent_api_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test deleting non-existent API key returns 404."""
    response = client.delete("/v1/keys/nonexistent-id", headers=master_key_header)

    assert response.status_code == 404


def test_rotate_api_key_returns_new_working_key_same_id(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Rotating a key returns a new secret for the same id, and the new key authenticates."""
    create_response = client.post(
        "/v1/keys",
        json={"key_name": "rotate-me", "metadata": {"team": "eng"}},
        headers=master_key_header,
    )
    assert create_response.status_code == 200
    original = create_response.json()

    rotate_response = client.post(
        f"/v1/keys/{original['id']}/rotate",
        headers=master_key_header,
    )
    assert rotate_response.status_code == 200
    rotated = rotate_response.json()

    assert rotated["id"] == original["id"]
    assert rotated["key"].startswith("gw-")
    assert rotated["key"] != original["key"]
    # Regenerate re-fingerprints: the prefix tracks the new secret, not the old one.
    assert rotated["key_prefix"] == rotated["key"][:10]
    assert rotated["key_prefix"] != original["key_prefix"]
    assert rotated["key_name"] == original["key_name"]
    assert rotated["user_id"] == original["user_id"]
    assert rotated["metadata"] == {"team": "eng"}
    assert rotated["is_active"] is True

    # The new secret authenticates.
    response = client.get(
        "/v1/models",
        headers={API_KEY_HEADER: f"Bearer {rotated['key']}"},
    )
    assert response.status_code == 200


def test_rotate_api_key_old_secret_stops_working(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """After rotation the previous secret no longer authenticates."""
    create_response = client.post(
        "/v1/keys",
        json={"key_name": "rotate-invalidate"},
        headers=master_key_header,
    )
    assert create_response.status_code == 200
    original = create_response.json()

    # Old secret works before rotation.
    before = client.get(
        "/v1/models",
        headers={API_KEY_HEADER: f"Bearer {original['key']}"},
    )
    assert before.status_code == 200

    rotate_response = client.post(
        f"/v1/keys/{original['id']}/rotate",
        headers=master_key_header,
    )
    assert rotate_response.status_code == 200

    # Old secret is rejected immediately, with no grace window.
    after = client.get(
        "/v1/models",
        headers={API_KEY_HEADER: f"Bearer {original['key']}"},
    )
    assert after.status_code == 401


def test_rotate_api_key_resets_last_used_at(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Rotation clears last_used_at since the new secret has never been used."""
    create_response = client.post(
        "/v1/keys",
        json={"key_name": "rotate-last-used"},
        headers=master_key_header,
    )
    api_key = create_response.json()

    # Exercise the key so last_used_at is populated.
    client.get("/v1/models", headers={API_KEY_HEADER: f"Bearer {api_key['key']}"})
    used_state = client.get(f"/v1/keys/{api_key['id']}", headers=master_key_header)
    assert used_state.json()["last_used_at"] is not None

    client.post(f"/v1/keys/{api_key['id']}/rotate", headers=master_key_header)

    rotated_state = client.get(f"/v1/keys/{api_key['id']}", headers=master_key_header)
    assert rotated_state.json()["last_used_at"] is None


def test_rotate_nonexistent_api_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Rotating a non-existent key returns 404."""
    response = client.post("/v1/keys/nonexistent-id/rotate", headers=master_key_header)

    assert response.status_code == 404


def test_rotate_api_key_without_master_key_fails(
    client: TestClient, api_key_obj: dict[str, Any]
) -> None:
    """Rotation without master key authentication is rejected."""
    response = client.post(f"/v1/keys/{api_key_obj['id']}/rotate")

    assert response.status_code in [401, 422]


def test_api_key_last_used_tracking(
    client: TestClient, master_key_header: dict[str, str], test_config: GatewayConfig
) -> None:
    """Test that last_used_at is updated when key is used."""
    create_response = client.post(
        "/v1/keys",
        json={"key_name": "usage-tracking"},
        headers=master_key_header,
    )
    api_key = create_response.json()

    get_response = client.get(f"/v1/keys/{api_key['id']}", headers=master_key_header)
    initial_last_used = get_response.json()["last_used_at"]

    import time

    time.sleep(0.1)

    client.post(
        "/v1/users",
        json={"user_id": "test-tracking-user"},
        headers=master_key_header,
    )

    _completion_response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hi"}],
            "user": "test-tracking-user",
        },
        headers={API_KEY_HEADER: f"Bearer {api_key['key']}"},
    )

    get_response = client.get(f"/v1/keys/{api_key['id']}", headers=master_key_header)
    updated_last_used = get_response.json()["last_used_at"]

    assert updated_last_used is not None
    assert updated_last_used != initial_last_used


def test_api_key_last_used_tracking_throttles_db_writes(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Requests within the throttle window should not rewrite last_used_at."""
    create_response = client.post(
        "/v1/keys",
        json={"key_name": "usage-throttle"},
        headers=master_key_header,
    )
    api_key = create_response.json()

    first_use = datetime(2026, 1, 1, tzinfo=UTC)
    second_use = first_use + timedelta(seconds=60)

    with patch("gateway.api.deps.datetime") as mock_datetime:
        mock_datetime.now.return_value = first_use
        response = client.get(
            "/v1/models",
            headers={API_KEY_HEADER: f"Bearer {api_key['key']}"},
        )
        assert response.status_code == 200

    first_state = client.get(f"/v1/keys/{api_key['id']}", headers=master_key_header)
    first_last_used = first_state.json()["last_used_at"]

    with patch("gateway.api.deps.datetime") as mock_datetime:
        mock_datetime.now.return_value = second_use
        response = client.get(
            "/v1/models",
            headers={API_KEY_HEADER: f"Bearer {api_key['key']}"},
        )
        assert response.status_code == 200

    second_state = client.get(f"/v1/keys/{api_key['id']}", headers=master_key_header)
    second_last_used = second_state.json()["last_used_at"]

    assert first_last_used is not None
    assert second_last_used == first_last_used


def test_inactive_api_key_rejected(
    client: TestClient, master_key_header: dict[str, str], test_config: GatewayConfig
) -> None:
    """Test that inactive API keys are rejected."""
    create_response = client.post(
        "/v1/keys",
        json={"key_name": "inactive-test"},
        headers=master_key_header,
    )
    api_key = create_response.json()

    client.post(
        "/v1/users",
        json={"user_id": "test-inactive-user"},
        headers=master_key_header,
    )

    client.patch(
        f"/v1/keys/{api_key['id']}",
        json={"is_active": False},
        headers=master_key_header,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hi"}],
            "user": "test-inactive-user",
        },
        headers={API_KEY_HEADER: f"Bearer {api_key['key']}"},
    )
    assert response.status_code == 401


def test_authorization_header_accepted(client: TestClient, test_config: GatewayConfig) -> None:
    """Test that Authorization header works as fallback for OpenAI client compatibility."""
    # Use Authorization header instead of Otari-Key
    auth_header = {"Authorization": f"Bearer {test_config.master_key}"}

    response = client.post(
        "/v1/keys",
        json={"key_name": "auth-header-test"},
        headers=auth_header,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["key_name"] == "auth-header-test"
