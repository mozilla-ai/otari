"""Integration tests for POST /api/v1/hooks/check.

Covers auth, the pass/fail/blocked shapes, and the parser's strictness
(unknown gate type, duplicate keys) surfacing as a 422, never a silent pass.
"""

import time
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from gateway.agent_runtime.domain.evaluators import _command_segments, _contains_subsequence
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

_NO_NPM_POLICY = (
    'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
    "  - id: g\n    type: command_match\n    enforcement: required\n"
    '    forbidden: ["npm"]\n    message: m\n'
)


@pytest.fixture
def tokenized_commands(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Returns the commands the hooks check splits into segments, one entry per split."""
    calls: list[str] = []

    def recording_command_segments(command: str) -> list[list[str]]:
        calls.append(command)
        return _command_segments(command)

    monkeypatch.setattr("gateway.agent_runtime.domain.evaluators._command_segments", recording_command_segments)
    return calls


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


def test_unsupported_gate_type_is_rejected_not_skipped(client: TestClient, master_key_header: dict[str, str]) -> None:
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

    The budget below is deliberately far above what the deduplicated work
    costs. What the timer actually spans is a whole HTTP round trip, and most
    of what is left in it once the quadratic blowup is gone is the unavoidable
    cost of the payload itself: parsing and validating 10,000 paths and a
    2,500-entry policy. That floor scales with the request rather than with
    the bug, so a budget pressed close to it measures how loaded the runner
    is, not whether the blowup is back (it failed at 1.07s against a 1.0s
    budget on a four-worker CI runner). Three seconds still catches a return
    to ~5s, which is the regression this exists to hold.
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
    assert time.time() - start < 3.0
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


def test_command_match_work_estimate_charges_a_shared_phrase_per_gate(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Many gates sharing one identical, moderately long forbidden phrase.

    phrase_cache tokenizes identical phrase text once, since gate.forbidden
    entries are validated up front and never change per gate; the cost
    estimate used to sum tokens per *distinct* phrase text
    (`sum(len(tokens) for tokens in phrase_cache.values())`), so 50 gates
    sharing one 20-token phrase counted as if only one gate carried it, while
    evaluate_command_match still runs _contains_subsequence once per gate.
    Chosen so the buggy estimate (a single phrase's 20 tokens times the
    commands' 50,000 total tokens, 1,000,000) clears _MAX_COMMAND_MATCH_WORK,
    while the real, per-occurrence estimate (multiplied by all 50 gates,
    50,000,000) does not; command_comparisons (50 phrases * 500 commands =
    25,000) stays far under its own budget, isolating this to the token-work
    estimate rather than the comparison-count one.
    """
    shared_phrase = " ".join(["a"] * 20)
    gates_yaml = "".join(
        f'  - id: g{i}\n    type: command_match\n    enforcement: required\n'
        f'    forbidden: ["{shared_phrase}"]\n    message: m\n'
        for i in range(50)
    )
    policy = 'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n' + gates_yaml
    commands = [f"{'b ' * 99}c{i}" for i in range(500)]
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "commands": commands},
        headers=master_key_header,
    )
    assert response.status_code == 422
    assert "50,000,000" in response.json()["detail"]


def test_many_whitespace_only_commands_do_not_stall_tokenizing(
    client: TestClient, master_key_header: dict[str, str], tokenized_commands: list[str]
) -> None:
    """Commands over the total character budget are refused before any of them is tokenized.

    Whitespace-only commands have no tokens, so only a character count bounds their tokenizing cost.
    """
    # A trailing index keeps the commands distinct, so evidence deduplication cannot merge them.
    commands = [" " * 4000 + str(i) for i in range(600)]
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _NO_NPM_POLICY, "commands": commands},
        headers=master_key_header,
    )
    assert response.status_code == 422
    assert "characters" in response.json()["detail"]
    assert tokenized_commands == []

    # The empty list above means something only if the recorder fires on a real command.
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _NO_NPM_POLICY, "commands": commands[:1]},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert tokenized_commands == commands[:1]


def test_a_policy_with_no_command_match_gate_never_tokenizes_commands(
    client: TestClient, master_key_header: dict[str, str], tokenized_commands: list[str]
) -> None:
    """A policy with no command_match gate never tokenizes the submitted commands.

    The commands are over the character budget, so the 200 also shows that budget is not applied.
    """
    # A trailing index keeps the commands distinct, so evidence deduplication cannot merge them.
    commands = [" " * 4000 + str(i) for i in range(600)]
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _VALID_POLICY, "changed_paths": [], "commands": commands},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert response.json()["blocked"] is False
    assert tokenized_commands == []

    # The empty list above means something only if the recorder fires on a real command.
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _NO_NPM_POLICY, "commands": commands[:1]},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert tokenized_commands == commands[:1]


def test_many_command_match_gates_do_not_retokenize_per_gate(
    client: TestClient, master_key_header: dict[str, str], tokenized_commands: list[str]
) -> None:
    """Each command is tokenized once per request, however many command_match gates check it."""
    gates_yaml = "".join(
        f'  - id: g{i}\n    type: command_match\n    enforcement: required\n    forbidden: ["npm"]\n    message: m\n'
        for i in range(100)
    )
    policy = 'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n' + gates_yaml
    commands = [" " * 4000 + str(i) for i in range(250)]
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "commands": commands},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert sorted(tokenized_commands) == sorted(commands)


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


def test_separator_only_commands_are_never_compared_against_a_phrase(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command made only of separators leaves no segment to compare against a phrase.

    This counts comparisons because a timed request on a loaded CI runner measures the runner.
    """
    comparisons = 0

    def counting_contains_subsequence(segment: list[str], phrase: list[str]) -> bool:
        nonlocal comparisons
        comparisons += 1
        return _contains_subsequence(segment, phrase)

    monkeypatch.setattr("gateway.agent_runtime.domain.evaluators._contains_subsequence", counting_contains_subsequence)
    forbidden = [f'"p{i}"' for i in range(500)]
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: command_match\n    enforcement: required\n"
        f"    forbidden: [{', '.join(forbidden)}]\n    message: m\n"
    )
    commands = ["; " * 500 + " " * i for i in range(100)]
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "commands": commands},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert response.json()["blocked"] is False
    assert comparisons == 0

    # The zero above means something only if the counter fires on a real command.
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "commands": ["p0"]},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert comparisons > 0


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


_COMMAND_IF_CHANGED_POLICY = """\
schema_version: "1.0"
policy:
  id: test/openapi-needs-postman
gates:
  - id: openapi-changed-needs-postman
    type: command_if_changed
    enforcement: required
    when_changed: ["docs/public/openapi.json"]
    require: ["make postman"]
    message: Run make postman after changing openapi.json.
"""


def test_command_if_changed_passes_when_required_command_ran(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={
            "policy_yaml": _COMMAND_IF_CHANGED_POLICY,
            "changed_paths": ["docs/public/openapi.json"],
            "commands": ["make postman"],
            "command_scope": "session",
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False
    assert body["results"][0]["outcome"] == "pass"


def test_command_if_changed_blocks_when_required_command_did_not_run(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={
            "policy_yaml": _COMMAND_IF_CHANGED_POLICY,
            "changed_paths": ["docs/public/openapi.json"],
            "commands": ["git status"],
            "command_scope": "session",
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is True
    assert body["results"][0]["outcome"] == "fail"
    assert body["results"][0]["detail"] == "docs/public/openapi.json"


def test_command_if_changed_is_not_applicable_when_no_matching_path_changed(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={
            "policy_yaml": _COMMAND_IF_CHANGED_POLICY,
            "changed_paths": ["README.md"],
            "commands": [],
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False
    assert body["results"][0]["outcome"] == "not_applicable"


def test_command_if_changed_does_not_block_the_edit_that_triggers_it(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The exact evidence a PreToolUse edit-tool call submits for its own

    target: changed_paths naming the file about to be edited, and an
    explicit empty commands list (edit calls never collect command
    evidence). The edit has not happened yet, so the required command
    cannot possibly have already run; this must resolve not_applicable, not
    a fail that would permanently block ever editing a when_changed-matched
    path (the required command can never run before the change that needs
    it, since that change is the one this very call is about to make).
    """
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={
            "policy_yaml": _COMMAND_IF_CHANGED_POLICY,
            "changed_paths": ["docs/public/openapi.json"],
            "commands": [],
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False
    assert body["results"][0]["outcome"] == "not_applicable"


def test_command_if_changed_blocks_when_commands_is_omitted(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Omitting `commands` entirely means command evidence was never

    collected at all, distinct from collecting it and finding nothing; a
    required command_if_changed gate must read that as `unknown` and block.
    """
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _COMMAND_IF_CHANGED_POLICY, "changed_paths": ["docs/public/openapi.json"]},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is True
    assert body["results"][0]["outcome"] == "unknown"


def test_command_if_changed_oversized_workload_is_rejected(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """when_changed's globs share changed_path's own match-work budget: a

    command_if_changed gate with a large enough when_changed list against a
    large enough changed_paths list should trip the same existing limit
    (test_oversized_aggregate_workload_is_rejected's own shape), not a new,
    unbounded code path.
    """
    when_changed = [f'"pattern-{i:03d}-{"x" * 40}"' for i in range(100)]
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: command_if_changed\n    enforcement: required\n"
        f"    when_changed: [{', '.join(when_changed)}]\n"
        '    require: ["make postman"]\n    message: m\n'
    )
    changed_paths = [f"src/{'y' * 40}-{i:05d}.txt" for i in range(10_000)]
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "changed_paths": changed_paths},
        headers=master_key_header,
    )
    assert response.status_code == 422, response.text
    assert "match operations" in response.json()["detail"]


def test_command_if_changed_defers_call_scoped_evidence(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The default scope is one tool call, which cannot answer this gate.

    A PreToolUse call submits the path it is about to edit and its own
    command, before the edit has run: failing there would block every edit to
    a when_changed-matched path, the edit being the very thing blocked.
    """
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={
            "policy_yaml": _COMMAND_IF_CHANGED_POLICY,
            "changed_paths": ["docs/public/openapi.json"],
            "commands": ["git status"],
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False
    assert body["results"][0]["outcome"] == "not_applicable"


def test_command_if_changed_blocks_when_the_session_ran_no_commands(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Session-scoped and empty is a real answer: the required command is

    among the commands the session did not run. Before command_scope existed
    this was indistinguishable from a PreToolUse edit call's empty list, so
    it had to resolve not_applicable and a session could satisfy the gate by
    never invoking Bash.
    """
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={
            "policy_yaml": _COMMAND_IF_CHANGED_POLICY,
            "changed_paths": ["docs/public/openapi.json"],
            "commands": [],
            "command_scope": "session",
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is True
    assert body["results"][0]["outcome"] == "fail"


def test_command_match_does_not_judge_session_scoped_evidence(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A forbidden command is judged at the call about to run it.

    Session evidence only grows, so matching against it would fail every
    remaining check of the session over one command already run, with nothing
    left that could clear it.
    """


    def outcome_for(scope: str) -> str:
        response = client.post(
            f"{API_ROOT}/hooks/check",
            json={
                "policy_yaml": _COMMAND_MATCH_POLICY,
                "commands": ["git push --force"],
                "command_scope": scope,
            },
            headers=master_key_header,
        )
        assert response.status_code == 200, response.text
        return str(response.json()["results"][0]["outcome"])

    assert outcome_for("session") == "not_applicable"
    assert outcome_for("call") == "fail"


_JUDGE_POLICY = """\
schema_version: "1.0"
policy:
  id: test/judge
gates:
  - id: follows-error-handling-pattern
    type: judge
    enforcement: advisory
    rubric: Does this change follow the repository's error-handling conventions?
    message: This change does not follow the error-handling conventions.
"""


def test_judge_gate_relays_a_passing_verdict(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={
            "policy_yaml": _JUDGE_POLICY,
            "judge_results": [
                {"gate_id": "follows-error-handling-pattern", "outcome": "pass", "reasoning": "looks fine"}
            ],
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False
    assert body["results"][0]["outcome"] == "pass"
    assert body["results"][0]["detail"] == "looks fine"


def test_judge_gate_relays_a_failing_verdict_without_blocking(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A judge gate is always advisory (rejected as required at parse time), so a
    failing model verdict warns without ever setting `blocked`.
    """
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={
            "policy_yaml": _JUDGE_POLICY,
            "judge_results": [
                {"gate_id": "follows-error-handling-pattern", "outcome": "fail", "reasoning": "swallows exceptions"}
            ],
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False
    assert body["results"][0]["outcome"] == "fail"
    assert body["results"][0]["message"] == "This change does not follow the error-handling conventions."
    assert body["results"][0]["detail"] == "swallows exceptions"


def test_judge_gate_is_unknown_when_no_verdict_was_submitted(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": _JUDGE_POLICY},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False, "unknown never blocks: the gate is always advisory"
    assert body["results"][0]["outcome"] == "unknown"


def test_judge_gate_reports_a_callers_model_error_without_blocking(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={
            "policy_yaml": _JUDGE_POLICY,
            "judge_results": [
                {
                    "gate_id": "follows-error-handling-pattern",
                    "outcome": "error",
                    "reasoning": "the `claude` CLI was not found on PATH",
                }
            ],
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False
    assert body["results"][0]["outcome"] == "error"


_WHEN_CHANGED_JUDGE_POLICY = """\
schema_version: "1.0"
policy:
  id: test/judge-when-changed
gates:
  - id: follows-error-handling-pattern
    type: judge
    enforcement: advisory
    rubric: Does this change follow the repository's error-handling conventions?
    when_changed: ["src/**"]
    message: This change does not follow the error-handling conventions.
"""


def test_judge_gate_with_when_changed_is_not_applicable_when_nothing_matches(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={
            "policy_yaml": _WHEN_CHANGED_JUDGE_POLICY,
            "changed_paths": ["docs/README.md"],
            "judge_results": [
                {"gate_id": "follows-error-handling-pattern", "outcome": "fail", "reasoning": "should not matter"}
            ],
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False
    assert body["results"][0]["outcome"] == "not_applicable"


def test_judge_gate_with_when_changed_is_unknown_when_change_evidence_was_not_submitted(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={
            "policy_yaml": _WHEN_CHANGED_JUDGE_POLICY,
            "judge_results": [
                {"gate_id": "follows-error-handling-pattern", "outcome": "pass", "reasoning": "fine"}
            ],
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False
    assert body["results"][0]["outcome"] == "unknown"


def test_judge_gate_with_when_changed_still_resolves_the_verdict_once_a_matching_path_changed(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={
            "policy_yaml": _WHEN_CHANGED_JUDGE_POLICY,
            "changed_paths": ["src/module.py"],
            "judge_results": [
                {"gate_id": "follows-error-handling-pattern", "outcome": "fail", "reasoning": "swallows exceptions"}
            ],
        },
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["blocked"] is False
    assert body["results"][0]["outcome"] == "fail"
    assert body["results"][0]["detail"] == "swallows exceptions"


def test_judge_when_changed_oversized_workload_is_rejected(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A judge gate's `when_changed` globs share changed_path's own match-work

    budget, the same as command_if_changed's own when_changed globs
    (test_command_if_changed_oversized_workload_is_rejected's own shape):
    evaluate_judge calls the same matched_changed_paths those globs are
    checked against, so excluding them from the budget would let a policy
    with enough judge gates run that same unbounded match work anyway.
    """
    when_changed = [f'"pattern-{i:03d}-{"x" * 40}"' for i in range(100)]
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: judge\n    enforcement: advisory\n    rubric: r\n"
        f"    when_changed: [{', '.join(when_changed)}]\n    message: m\n"
    )
    changed_paths = [f"src/{'y' * 40}-{i:05d}.txt" for i in range(10_000)]
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy, "changed_paths": changed_paths},
        headers=master_key_header,
    )
    assert response.status_code == 422, response.text
    assert "match operations" in response.json()["detail"]


def test_judge_gate_rejects_required_enforcement(client: TestClient, master_key_header: dict[str, str]) -> None:
    policy = (
        'schema_version: "1.0"\npolicy:\n  id: x\ngates:\n'
        "  - id: g\n    type: judge\n    enforcement: required\n    rubric: r\n    message: m\n"
    )
    response = client.post(
        f"{API_ROOT}/hooks/check",
        json={"policy_yaml": policy},
        headers=master_key_header,
    )
    assert response.status_code == 422, response.text
    assert "judge" in response.json()["detail"]
