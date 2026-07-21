"""Integration tests for the per-user model access-control layer.

Covers the user default flowing to a key that has no list of its own (catalog +
inference), the narrow-only subset-on-write rule for keys, and the user
allowed_models write API (round-trip, validation, PATCH tri-state).
"""

from typing import Any

from fastapi.testclient import TestClient

DENIED = "gemini:gemini-2.5-flash"
ALLOWED = "openai:gpt-4o"


def _make_user(client: TestClient, headers: dict[str, str], user_id: str, allowed_models: Any) -> None:
    resp = client.post(
        "/v1/users",
        json={"user_id": user_id, "allowed_models": allowed_models},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text


def _key_for_user(
    client: TestClient, headers: dict[str, str], user_id: str, allowed_models: Any = None
) -> dict[str, str]:
    resp = client.post(
        "/v1/keys",
        json={"key_name": "k", "user_id": user_id, "allowed_models": allowed_models},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    return {"Otari-Key": f"Bearer {resp.json()['key']}"}


def _seed_pricing(client: TestClient, headers: dict[str, str], model_key: str) -> None:
    resp = client.post(
        "/v1/pricing",
        json={"model_key": model_key, "input_price_per_million": 1.0, "output_price_per_million": 2.0},
        headers=headers,
    )
    assert resp.status_code in (200, 201), resp.text


# --- user write API --------------------------------------------------------


def test_user_allowed_models_round_trips_and_validates(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    created = client.post(
        "/v1/users",
        json={"user_id": "u-rt", "allowed_models": ["openai:*", "openai:*"]},
        headers=master_key_header,
    )
    assert created.status_code == 200
    # Canonicalized + de-duplicated like the per-key layer.
    assert created.json()["allowed_models"] == ["openai:*"]

    bad = client.post(
        "/v1/users",
        json={"user_id": "u-bad", "allowed_models": ["gpt-4o"]},
        headers=master_key_header,
    )
    assert bad.status_code == 400
    assert "instance:model" in bad.json()["detail"]


def test_user_allowed_models_patch_tri_state(client: TestClient, master_key_header: dict[str, str]) -> None:
    _make_user(client, master_key_header, "u-tri", ["openai:*"])

    # Absent: unchanged.
    client.patch("/v1/users/u-tri", json={"alias": "x"}, headers=master_key_header)
    assert client.get("/v1/users/u-tri", headers=master_key_header).json()["allowed_models"] == ["openai:*"]

    # Explicit null: clear to unrestricted.
    client.patch("/v1/users/u-tri", json={"allowed_models": None}, headers=master_key_header)
    assert client.get("/v1/users/u-tri", headers=master_key_header).json()["allowed_models"] is None

    # Empty list: deny all.
    client.patch("/v1/users/u-tri", json={"allowed_models": []}, headers=master_key_header)
    assert client.get("/v1/users/u-tri", headers=master_key_header).json()["allowed_models"] == []


# --- inheritance: user default flows to a key with no list of its own ------


def test_key_inherits_user_default_in_catalog(client: TestClient, master_key_header: dict[str, str]) -> None:
    _seed_pricing(client, master_key_header, ALLOWED)
    _seed_pricing(client, master_key_header, DENIED)
    _make_user(client, master_key_header, "u-cat", ["openai:*"])
    # Key has no list of its own -> inherits the user's openai-only default.
    inheriting = _key_for_user(client, master_key_header, "u-cat", allowed_models=None)

    ids = {m["id"] for m in client.get("/v1/models", headers=inheriting).json()["data"]}
    assert ALLOWED in ids
    assert DENIED not in ids


def test_key_inherits_user_default_at_inference(client: TestClient, master_key_header: dict[str, str]) -> None:
    _make_user(client, master_key_header, "u-inf", ["openai:*"])
    inheriting = _key_for_user(client, master_key_header, "u-inf", allowed_models=None)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": DENIED, "messages": [{"role": "user", "content": "hi"}]},
        headers=inheriting,
    )
    assert resp.status_code == 403
    assert "not permitted" in str(resp.json())


def test_unrestricted_user_leaves_inheriting_key_open(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _seed_pricing(client, master_key_header, ALLOWED)
    _seed_pricing(client, master_key_header, DENIED)
    _make_user(client, master_key_header, "u-open", None)
    inheriting = _key_for_user(client, master_key_header, "u-open", allowed_models=None)

    ids = {m["id"] for m in client.get("/v1/models", headers=inheriting).json()["data"]}
    assert {ALLOWED, DENIED} <= ids


# --- narrow-only: subset-on-write for keys ---------------------------------


def test_key_narrower_than_user_default_is_accepted(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _make_user(client, master_key_header, "u-narrow", ["openai:*"])
    resp = client.post(
        "/v1/keys",
        json={"key_name": "k", "user_id": "u-narrow", "allowed_models": ["openai:gpt-4o"]},
        headers=master_key_header,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["allowed_models"] == ["openai:gpt-4o"]


def test_key_broader_than_user_default_is_rejected_on_create(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _make_user(client, master_key_header, "u-cap", ["openai:gpt-4o"])
    resp = client.post(
        "/v1/keys",
        json={"key_name": "k", "user_id": "u-cap", "allowed_models": ["openai:*"]},
        headers=master_key_header,
    )
    assert resp.status_code == 400
    assert "narrow" in resp.json()["detail"].lower()


def test_key_broadening_rejected_on_patch(client: TestClient, master_key_header: dict[str, str]) -> None:
    _make_user(client, master_key_header, "u-patch", ["openai:gpt-4o"])
    key_id = client.post(
        "/v1/keys",
        json={"key_name": "k", "user_id": "u-patch", "allowed_models": ["openai:gpt-4o"]},
        headers=master_key_header,
    ).json()["id"]

    resp = client.patch(
        f"/v1/keys/{key_id}",
        json={"allowed_models": ["openai:*"]},
        headers=master_key_header,
    )
    assert resp.status_code == 400
    assert "narrow" in resp.json()["detail"].lower()


def test_deny_all_user_rejects_a_granting_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    _make_user(client, master_key_header, "u-deny", [])
    resp = client.post(
        "/v1/keys",
        json={"key_name": "k", "user_id": "u-deny", "allowed_models": ["openai:gpt-4o"]},
        headers=master_key_header,
    )
    assert resp.status_code == 400
    assert "narrow" in resp.json()["detail"].lower()
