"""Integration tests for POST /api/v1/hooks/check.

Covers auth, the pass/fail/blocked shapes, and the parser's strictness
(unknown gate type, duplicate keys) surfacing as a 422, never a silent pass.
"""

import time
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, PLATFORM_TOKEN_ENV_VAR, GatewayConfig

from .conftest import build_test_client

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


def test_oversized_aggregate_workload_is_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Each individual match is now linear, but a request can still pair a large
    forbidden-glob list with a large changed_paths list. The route's own work
    budget (not any per-match cost) must reject that combination with a 422
    rather than let it run.
    """
    forbidden = [f'"pattern-{i:03d}-{"x" * 40}"' for i in range(100)]
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: changed_path\n    enforcement: required\n"
        f"    forbidden: [{', '.join(forbidden)}]\n    message: m\n"
    )
    changed_paths = [f"src/{'y' * 40}-{i:05d}.txt" for i in range(10_000)]
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "changed_paths": changed_paths},
        headers=master_key_header,
    )
    assert response.status_code == 422
    assert "match operations" in response.json()["detail"]


def test_duplicated_globs_and_paths_resolve_quickly_instead_of_blocking(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Review's repro for the P1 this route used to have: 2,500 copies of one

    forbidden glob against 10,000 copies of one changed path pass the
    byte-weighted work budget (pattern_count * total_path_length +
    path_count * total_pattern_length is small when every string is one
    byte) yet, unmatched, cost 25,000,000 real match calls, which measured
    ~5s of synchronous blocking. Deduplicating at parse time and at the
    evidence boundary (domain.policy, PolicyCheckRequest.changed_path_evidence)
    collapses this to one pattern against one path.
    """
    quoted_b = '"b"'
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: changed_path\n    enforcement: required\n"
        f"    forbidden: [{', '.join([quoted_b] * 2500)}]\n    message: m\n"
    )
    changed_paths = ["a"] * 10_000
    start = time.time()
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "changed_paths": changed_paths},
        headers=master_key_header,
    )
    assert time.time() - start < 1.0
    assert response.status_code == 200, response.text
    assert response.json()["blocked"] is False


def test_many_distinct_short_globs_and_paths_trip_the_comparisons_bound(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A byte-weighted budget alone understates a request built from many

    short, distinct strings: 2,000 five-character forbidden globs against
    2,000 five-character changed paths estimate 40,000,000 work, under
    _MAX_MATCH_WORK, but mean 4,000,000 real match calls. Deduplication does
    not help here (every string is distinct), so this must be caught by a
    raw comparison-count bound instead.
    """
    forbidden = [f'"p{i:04d}"' for i in range(2000)]
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: changed_path\n    enforcement: required\n"
        f"    forbidden: [{', '.join(forbidden)}]\n    message: m\n"
    )
    changed_paths = [f"q{i:04d}" for i in range(2000)]
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "changed_paths": changed_paths},
        headers=master_key_header,
    )
    assert response.status_code == 422
    assert "comparisons" in response.json()["detail"]


class TestHybridMode:
    """The Hook Server answers on a hybrid gateway too.

    A gate evaluates only the policy and evidence the caller sent in the same
    request, so it needs no local tenancy, no provider and no database. Mounted
    on the standalone side of ``_register_core_routers``' hybrid early return,
    the endpoint 404s on a hybrid gateway and ``otari hook`` fails open against
    it forever, which is silent: fail-open is what the command promises for an
    unreachable gateway, so nothing tells the caller their gates stopped
    running.
    """

    @pytest.fixture(scope="class")
    def hybrid_client(self, postgres_url: str) -> Generator[TestClient]:
        # The platform token is resolved once and cached on the config, so it
        # need only be in the environment until that resolution happens, which
        # is what the explicit ``_resolve_platform_token()`` forces here (it is
        # otherwise lazy, and would fire later against an environment the
        # context has already restored). Left set for the whole fixture, it
        # would make the next standalone app built in this process refuse to
        # start.
        with pytest.MonkeyPatch.context() as env:
            env.setenv(PLATFORM_TOKEN_ENV_VAR, "test-platform-token")
            config = GatewayConfig(
                mode="hybrid",
                database_url=postgres_url,
                master_key="test-master-key",
                auto_migrate=False,
                require_pricing=False,
                model_discovery=False,
                bootstrap_api_key=False,
                platform={"base_url": "http://localhost:8100/api/v1"},
            )
            config._resolve_platform_token()
        yield from build_test_client(config)

    def test_is_mounted_and_evaluates(self, hybrid_client: TestClient) -> None:
        response = hybrid_client.post(
            f"{API_ROOT}/hooks/check",
            json={"policy_yaml": _VALID_POLICY, "changed_paths": ["scratch/notes.txt"]},
            headers={"Authorization": "Bearer any-platform-user-token"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["blocked"] is True
        assert body["results"][0]["outcome"] == "fail"

    def test_still_requires_a_token(self, hybrid_client: TestClient) -> None:
        """Hybrid cannot validate the token locally, but it does require one.

        The same thing the stateless MCP route does in this mode. Weaker than
        the standalone check on purpose: this endpoint reads no tenant data and
        bills nothing, and what one request can cost is bounded by the route's
        own work budgets rather than by who sent it.
        """
        response = hybrid_client.post(
            f"{API_ROOT}/hooks/check",
            json={"policy_yaml": _VALID_POLICY, "changed_paths": ["README.md"]},
        )
        assert response.status_code == 401
