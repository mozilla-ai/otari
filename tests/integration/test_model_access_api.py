"""Integration tests for per-key model access control.

Covers the write API (allowed_models round-trip, validation, PATCH tri-state),
the catalog filter (list_models / get_model), and the inference gate (403 before
dispatch on chat and batches, master-key bypass).
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

DENIED = "gemini:gemini-2.5-flash"
ALLOWED = "openai:gpt-4o"


def _make_key(client: TestClient, headers: dict[str, str], allowed_models: Any) -> dict[str, str]:
    resp = client.post(
        "/v1/keys",
        json={"key_name": "scoped", "allowed_models": allowed_models},
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


# --- write API -------------------------------------------------------------


def test_create_and_get_key_round_trips_allowed_models(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/keys",
        json={"key_name": "k", "allowed_models": ["openai:*", "anthropic:claude-3*"]},
        headers=master_key_header,
    )
    assert resp.status_code == 200
    key_id = resp.json()["id"]
    assert resp.json()["allowed_models"] == ["openai:*", "anthropic:claude-3*"]

    got = client.get(f"/v1/keys/{key_id}", headers=master_key_header)
    assert got.json()["allowed_models"] == ["openai:*", "anthropic:claude-3*"]


def test_create_key_rejects_non_canonical_entry(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.post(
        "/v1/keys",
        json={"key_name": "bad", "allowed_models": ["gpt-4o"]},
        headers=master_key_header,
    )
    assert resp.status_code == 400
    assert "instance:model" in resp.json()["detail"]


def test_patch_tri_state(client: TestClient, master_key_header: dict[str, str]) -> None:
    key_id = client.post(
        "/v1/keys",
        json={"key_name": "t", "allowed_models": ["openai:*"]},
        headers=master_key_header,
    ).json()["id"]

    # Absent field: unchanged.
    client.patch(f"/v1/keys/{key_id}", json={"key_name": "renamed"}, headers=master_key_header)
    assert client.get(f"/v1/keys/{key_id}", headers=master_key_header).json()["allowed_models"] == ["openai:*"]

    # Explicit null: clear to unrestricted.
    client.patch(f"/v1/keys/{key_id}", json={"allowed_models": None}, headers=master_key_header)
    assert client.get(f"/v1/keys/{key_id}", headers=master_key_header).json()["allowed_models"] is None

    # Empty list: deny all.
    client.patch(f"/v1/keys/{key_id}", json={"allowed_models": []}, headers=master_key_header)
    assert client.get(f"/v1/keys/{key_id}", headers=master_key_header).json()["allowed_models"] == []


# --- catalog filter --------------------------------------------------------


def test_list_models_filtered_by_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    _seed_pricing(client, master_key_header, ALLOWED)
    _seed_pricing(client, master_key_header, DENIED)
    scoped = _make_key(client, master_key_header, ["openai:*"])

    ids = {m["id"] for m in client.get("/v1/models", headers=scoped).json()["data"]}
    assert ALLOWED in ids
    assert DENIED not in ids

    # Master key sees everything.
    master_ids = {m["id"] for m in client.get("/v1/models", headers=master_key_header).json()["data"]}
    assert {ALLOWED, DENIED} <= master_ids


def test_list_models_empty_for_deny_all(client: TestClient, master_key_header: dict[str, str]) -> None:
    _seed_pricing(client, master_key_header, ALLOWED)
    scoped = _make_key(client, master_key_header, [])
    assert client.get("/v1/models", headers=scoped).json()["data"] == []


def test_get_model_404_for_denied(client: TestClient, master_key_header: dict[str, str]) -> None:
    _seed_pricing(client, master_key_header, DENIED)
    scoped = _make_key(client, master_key_header, ["openai:*"])
    # 404 (not 403): a denied model is indistinguishable from a missing one.
    assert client.get(f"/v1/models/{DENIED}", headers=scoped).status_code == 404


# --- inference gate --------------------------------------------------------


def test_chat_403_for_denied_model(client: TestClient, master_key_header: dict[str, str]) -> None:
    scoped = _make_key(client, master_key_header, ["openai:*"])
    resp = client.post(
        "/v1/chat/completions",
        json={"model": DENIED, "messages": [{"role": "user", "content": "hi"}]},
        headers=scoped,
    )
    assert resp.status_code == 403
    assert "not permitted" in str(resp.json())


def test_chat_allowed_model_passes_the_gate(client: TestClient, master_key_header: dict[str, str]) -> None:
    from unittest.mock import patch

    scoped = _make_key(client, master_key_header, ["gemini:*"])
    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> None:
        captured.update(kwargs)
        raise RuntimeError("short-circuit after the gate")

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        client.post(
            "/v1/chat/completions",
            json={"model": DENIED, "messages": [{"role": "user", "content": "hi"}]},
            headers=scoped,
        )
    # The allowed model reached dispatch (the gate did not reject it).
    assert captured, "acompletion was never called; the gate wrongly blocked an allowed model"


def test_batches_403_for_denied_model(client: TestClient, master_key_header: dict[str, str]) -> None:
    scoped = _make_key(client, master_key_header, ["openai:*"])
    resp = client.post(
        "/v1/batches",
        json={
            "model": DENIED,
            "requests": [{"custom_id": "1", "body": {"messages": [{"role": "user", "content": "hi"}]}}],
        },
        headers=scoped,
    )
    assert resp.status_code == 403
    assert "not permitted" in str(resp.json())


@pytest.mark.parametrize("model", [ALLOWED, DENIED])
def test_master_key_never_restricted_on_catalog(
    client: TestClient, master_key_header: dict[str, str], model: str
) -> None:
    _seed_pricing(client, master_key_header, model)
    ids = {m["id"] for m in client.get("/v1/models", headers=master_key_header).json()["data"]}
    assert model in ids
