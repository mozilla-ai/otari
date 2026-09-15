"""Integration tests for POST /api/v1/hooks/check.

Covers auth, the pass/fail/blocked shapes, and the parser's strictness
(unknown gate type, duplicate keys) surfacing as a 422, never a silent pass.
"""

from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT

_VALID_POLICY = """\
schema_version: "1.0"
policy:
  id: test/repo-quality
gates:
  - id: no-scratch-files
    type: changed_path
    enforcement: required
    forbidden: ["scratch/**"]
    message: Do not commit scratch files.
"""


def test_requires_authentication(client: TestClient) -> None:
    response = client.post(f"{API_ROOT}/hooks/check", json={"policy_yaml": _VALID_POLICY})
    assert response.status_code in (401, 403)


def test_passes_with_no_forbidden_changes(client: TestClient, api_key_header: dict[str, str]) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _VALID_POLICY, "changed_paths": ["README.md"]},
        headers=api_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["policy_id"] == "test/repo-quality"
    assert body["blocked"] is False
    assert body["provenance"] == "client_reported"
    assert body["results"][0]["outcome"] == "pass"


def test_blocks_on_a_forbidden_change(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _VALID_POLICY, "changed_paths": ["scratch/notes.txt"]},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is True
    assert body["results"][0]["outcome"] == "fail"
    assert body["results"][0]["detail"] == "scratch/notes.txt"


def test_no_changed_paths_means_no_evidence_to_match(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _VALID_POLICY},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert response.json()["blocked"] is False


def test_unsupported_gate_type_is_rejected_not_skipped(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: command_match\n    enforcement: required\n    message: m\n"
    )
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy},
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_duplicate_yaml_keys_are_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    policy = 'schema_version: "1.0"\nschema_version: "1.0"\npolicy:\n  id: x\ngates: []\n'
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy},
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_extra_top_level_field_is_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _VALID_POLICY, "extra_field": "x"},
        headers=master_key_header,
    )
    assert response.status_code == 422
