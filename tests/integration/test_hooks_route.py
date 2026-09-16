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


def test_an_empty_changed_paths_list_is_not_applicable_not_a_pass(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """`[]` says evidence was collected and there is none: non-blocking, but
    reported as not_applicable rather than as a check that ran and passed.
    """
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _VALID_POLICY, "changed_paths": []},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False
    assert body["results"][0]["outcome"] == "not_applicable"


def test_omitted_changed_paths_blocks_rather_than_passing(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Omitting the field says this caller never collects path evidence at
    all, which must not read as a pass. It used to default to `[]` and
    certify every changed_path gate in the policy.
    """
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _VALID_POLICY},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["results"][0]["outcome"] == "unknown"
    assert body["blocked"] is True


def test_unsupported_gate_type_is_rejected_not_skipped(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: judge\n    enforcement: required\n    message: m\n"
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


_COMMAND_MATCH_POLICY = """\
schema_version: "1.0"
policy:
  id: test/no-force-push
gates:
  - id: no-force-push
    type: command_match
    enforcement: required
    forbidden: ["git push --force", "git push -f"]
    message: Force-pushing is not allowed.
"""

_COMMAND_MATCH_USE_PNPM_POLICY = """\
schema_version: "1.0"
policy:
  id: test/use-pnpm
gates:
  - id: use-pnpm
    type: command_match
    enforcement: required
    forbidden: ["npm"]
    message: Use pnpm, not npm.
"""


def test_command_match_passes_when_no_forbidden_command_run(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _COMMAND_MATCH_POLICY, "commands": ["git push"]},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert response.json()["blocked"] is False


def test_command_match_blocks_on_a_forbidden_command(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _COMMAND_MATCH_POLICY, "commands": ["git push --force"]},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is True
    assert body["results"][0]["outcome"] == "fail"
    assert body["results"][0]["detail"] == "git push --force"


def test_changed_path_and_command_match_gates_preserve_declaration_order(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: no-force-push\n    type: command_match\n    enforcement: required\n"
        '    forbidden: ["git push --force"]\n    message: no force push\n'
        "  - id: no-scratch-files\n    type: changed_path\n    enforcement: required\n"
        '    forbidden: ["scratch/**"]\n    message: no scratch files\n'
    )
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "commands": ["git push --force"], "changed_paths": ["scratch/x.txt"]},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [result["gate_id"] for result in body["results"]] == ["no-force-push", "no-scratch-files"]
    assert body["blocked"] is True


def test_command_match_oversized_workload_is_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Distinct, short forbidden phrases against many short, distinct commands:

    little token content (cheap by _MAX_COMMAND_MATCH_WORK) but a large
    number of phrase/command pairs, mirroring changed_path's own comparisons
    finding for the same reason: a byte/token-weighted budget alone
    understates many-short-items requests.
    """
    forbidden = [f'"p{i:04d}"' for i in range(500)]
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: command_match\n    enforcement: required\n"
        f"    forbidden: [{', '.join(forbidden)}]\n    message: m\n"
    )
    commands = [f"q{i:04d}" for i in range(2000)]
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "commands": commands},
        headers=master_key_header,
    )
    assert response.status_code == 422
    assert "comparisons" in response.json()["detail"]


def test_many_whitespace_only_commands_do_not_stall_tokenizing(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """shlex.split costs meaningfully more per character than a plain len()

    check, regardless of content, so a request built from many long,
    all-whitespace commands (which tokenize to zero tokens each, keeping
    _MAX_COMMAND_MATCH_WORK's estimate at zero no matter how many there are)
    can still cost real seconds just computing that estimate. This must be
    caught by a raw character-total budget before any command is tokenized,
    not discovered only after tokenizing all of them.
    """
    # Distinct (a trailing index) so evidence deduplication does not collapse
    # this back down to one command and hide the aggregate-length case.
    commands = [" " * 4000 + str(i) for i in range(600)]
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: command_match\n    enforcement: required\n"
        '    forbidden: ["npm"]\n    message: m\n'
    )
    start = time.time()
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "commands": commands},
        headers=master_key_header,
    )
    assert time.time() - start < 1.0
    assert response.status_code == 422
    assert "characters" in response.json()["detail"]


def test_a_policy_with_no_command_match_gate_never_tokenizes_commands(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Submitting `commands` evidence against a policy with no command_match

    gate must not pay any tokenizing cost at all: the result is moot
    regardless, so this must resolve quickly and successfully rather than
    being rejected by a budget meant for command_match gates that do not
    exist here.
    """
    # Distinct (a trailing index) so evidence deduplication does not collapse
    # this back down to one command and hide the aggregate-length case.
    commands = [" " * 4000 + str(i) for i in range(600)]
    start = time.time()
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _VALID_POLICY, "changed_paths": [], "commands": commands},
        headers=master_key_header,
    )
    assert time.time() - start < 1.0
    assert response.status_code == 200, response.text
    assert response.json()["blocked"] is False


def test_many_command_match_gates_do_not_retokenize_per_gate(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Review's repro: 100 command_match gates, each forbidding "npm", against

    250 distinct ~4,000-character mostly-whitespace commands passed every
    request-level budget (low token content, few phrases per gate) yet
    measured ~7s of synchronous blocking in check_policy, because
    evaluate_command_match was called once per gate and each call
    independently re-tokenized every command from scratch. Tokenizing once
    per request and sharing the result across every command_match gate's
    evaluation collapses this to well under a second.
    """
    gates_yaml = "".join(
        f'  - id: g{i}\n    type: command_match\n    enforcement: required\n'
        f'    forbidden: ["npm"]\n    message: m\n'
        for i in range(100)
    )
    policy = 'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n' + gates_yaml
    commands = [" " * 4000 + str(i) for i in range(250)]
    start = time.time()
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "commands": commands},
        headers=master_key_header,
    )
    assert time.time() - start < 1.0
    assert response.status_code == 200, response.text


def test_apostrophe_in_a_trailing_comment_does_not_evade_a_required_gate(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """shlex's default (comments=False) does not strip a trailing '#'

    comment, so an apostrophe inside one ("don't") used to raise a
    ValueError that fell back to one opaque, never-matching token: a
    required gate forbidding "npm" silently passed "npm install # don't
    use yarn".
    """
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: command_match\n    enforcement: required\n"
        '    forbidden: ["npm"]\n    message: m\n'
    )
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "commands": ["npm install # don't use yarn"]},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is True
    assert body["results"][0]["outcome"] == "fail"


def test_many_separator_only_commands_resolve_quickly(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Review's repro: one gate with 500 forbidden phrases against 100 commands

    built from 500 semicolons each (no real content) passed every request
    budget (the token count correctly counts 0 real tokens) yet measured
    ~1.1s, because evaluation still compared every one of ~50,000 resulting
    empty segments against every phrase. Dropping empty segments at the
    source, since a non-empty phrase can never match one, collapses this.
    """
    forbidden = [f'"p{i}"' for i in range(500)]
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: command_match\n    enforcement: required\n"
        f"    forbidden: [{', '.join(forbidden)}]\n    message: m\n"
    )
    commands = ["; " * 500 + " " * i for i in range(100)]
    start = time.time()
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "commands": commands},
        headers=master_key_header,
    )
    assert time.time() - start < 0.5
    assert response.status_code == 200, response.text
    assert response.json()["blocked"] is False


def test_multiline_command_with_a_leading_comment_still_blocks(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A comment on an earlier line must not swallow a real command on a

    later line: "# install dependencies\\nnpm install" is a comment, then a
    real, separate npm invocation.
    """
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: command_match\n    enforcement: required\n"
        '    forbidden: ["npm"]\n    message: m\n'
    )
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "commands": ["# install dependencies\nnpm install"]},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert response.json()["blocked"] is True


def test_escaped_quote_does_not_hide_a_later_command_as_a_bogus_comment(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    r"""'echo "a\" # b" && npm install' is one quoted argument (via the

    escaped quote) followed by a real, separate npm invocation; the '#'
    inside the quote must not be treated as a comment that discards it.
    """
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: command_match\n    enforcement: required\n"
        '    forbidden: ["npm"]\n    message: m\n'
    )
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "commands": ['echo "a\\" # b" && npm install']},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert response.json()["blocked"] is True


def test_ansi_c_quoted_escaped_apostrophe_does_not_evade_a_required_gate(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    r"""echo $'a\' # b' && npm install keeps '# b' inside its ANSI-C-quoted

    argument (backslash escapes are active inside $'...', so \' is a
    literal apostrophe there, not the closing quote); the real, separate
    "&& npm install" after it must still be visible to this gate.
    """
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _COMMAND_MATCH_USE_PNPM_POLICY, "commands": ["echo $'a\\' # b' && npm install"]},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert response.json()["blocked"] is True


def test_omitted_commands_blocks_a_required_command_match_gate(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Omitting `commands` from the request body entirely (as distinct from

    sending an explicit `[]`) means this caller never collected command
    evidence at all. A required command_match gate must read that as
    `unknown` and block, not silently `pass`.
    """
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _COMMAND_MATCH_POLICY},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is True
    assert body["results"][0]["outcome"] == "unknown"


def test_explicit_empty_commands_is_not_applicable_not_a_pass(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """An explicit `commands: []`, what a PreToolUse edit call or a Stop

    event submits, is evidence that was collected with nothing to report,
    not "checked, none forbidden". A required gate must resolve
    `not_applicable` (non-blocking, but honest that nothing was checked),
    never a `pass` that reads as a clean check.
    """
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _COMMAND_MATCH_POLICY, "commands": []},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False
    assert body["results"][0]["outcome"] == "not_applicable"
