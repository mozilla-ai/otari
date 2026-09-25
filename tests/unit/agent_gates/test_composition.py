"""Composing several standalone guardrail files into one policy."""

import pytest

import otari_agent.domain.policy as policy_module
from otari_agent.domain.check import check_policy
from otari_agent.domain.policy import MAX_POLICY_FILES, PolicyError, PolicyFile, compose_policy, parse_policy
from otari_agent.domain.types import JudgeGate, VerifierGate, by_priority


def _file(name: str, *gate_ids: str, schema: str = "1.0", policy_id: str | None = None) -> PolicyFile:
    gates = "".join(
        f"  - id: {gate_id}\n"
        "    type: path\n"
        "    runs: [stop.working_tree]\n"
        "    enforcement: required\n"
        f"    forbidden: ['{gate_id}.txt']\n"
        f"    message: no {gate_id}\n"
        for gate_id in gate_ids
    )
    body = f'schema_version: "{schema}"\npolicy:\n  id: {policy_id or name}\ngates:\n{gates}'
    return PolicyFile(name=name, body=body)


def test_gates_compose_in_file_order() -> None:
    spec = compose_policy(
        [_file("a.yml", "a1", "a2"), _file("b.yml", "b1")],
        policy_id=".otari/guardrails",
    )
    assert [gate.id for gate in spec.gates] == ["a1", "a2", "b1"]
    assert spec.policy_id == ".otari/guardrails"
    assert spec.schema_version == "1.0"


def test_each_gate_carries_the_file_that_declared_it() -> None:
    spec = compose_policy([_file("a.yml", "a1"), _file("nested/b.yml", "b1")], policy_id="x")
    assert spec.gate_sources == {"a1": "a.yml", "b1": "nested/b.yml"}


def test_a_one_file_composition_carries_no_sources() -> None:
    """Nothing to disambiguate, so nothing to report: see PolicySpec.gate_sources."""
    spec = compose_policy([_file("a.yml", "a1")], policy_id="x")
    assert spec.gate_sources == {}


def test_a_gate_id_repeated_across_files_is_an_error_naming_both() -> None:
    with pytest.raises(PolicyError) as excinfo:
        compose_policy([_file("a.yml", "shared"), _file("b.yml", "shared")], policy_id="x")
    message = str(excinfo.value)
    assert "'shared'" in message
    assert "a.yml" in message and "b.yml" in message


def test_a_file_that_does_not_parse_fails_the_whole_composition() -> None:
    """Never a partial policy: the same contract parse_policy has within one file."""
    with pytest.raises(PolicyError, match="b.yml"):
        compose_policy([_file("a.yml", "a1"), PolicyFile(name="b.yml", body="not: a: policy:")], policy_id="x")


def test_an_unsupported_schema_version_is_refused_per_file() -> None:
    """The reachable half of the version rule today, since only one version exists."""
    with pytest.raises(PolicyError, match="b.yml"):
        compose_policy([_file("a.yml", "a1"), _file("b.yml", "b1", schema="2.0")], policy_id="x")


def test_files_must_agree_on_schema_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two versions this build understands must still not be mixed in one composed set.

    Only one version exists, so the second is introduced here rather than
    waiting for a release to make the rule reachable: it is the rule that
    keeps a newer file dropped into an older directory from being silently
    read as though it were the older shape.
    """
    monkeypatch.setattr(policy_module, "_SUPPORTED_SCHEMA_VERSIONS", {"1.0", "2.0"})
    with pytest.raises(PolicyError) as excinfo:
        compose_policy([_file("a.yml", "a1"), _file("b.yml", "b1", schema="2.0")], policy_id="x")
    assert "a.yml" in str(excinfo.value) and "b.yml" in str(excinfo.value)


def test_composing_nothing_is_an_error() -> None:
    with pytest.raises(PolicyError, match="composes no guardrail files"):
        compose_policy([], policy_id=".otari/guardrails")


def test_more_files_than_the_limit_is_an_error() -> None:
    files = [_file(f"f{i}.yml", f"g{i}") for i in range(MAX_POLICY_FILES + 1)]
    with pytest.raises(PolicyError, match=str(MAX_POLICY_FILES)):
        compose_policy(files, policy_id="x")


def test_exactly_the_limit_composes() -> None:
    files = [_file(f"f{i}.yml", f"g{i}") for i in range(MAX_POLICY_FILES)]
    assert len(compose_policy(files, policy_id="x").gates) == MAX_POLICY_FILES


def test_the_byte_limit_is_per_file_not_per_composed_set() -> None:
    """A file legal on its own stays legal beside others; that is what makes a snippet droppable."""
    big = 'schema_version: "1.0"\npolicy:\n  id: p\ngates:\n' + "".join(
        f"  - id: g{i}\n"
        "    type: path\n"
        "    runs: [stop.working_tree]\n"
        "    enforcement: advisory\n"
        f"    forbidden: ['{'x' * 200}{i}.txt']\n"
        f"    message: {'m' * 200}\n"
        for i in range(400)
    )
    one = PolicyFile(name="one.yml", body=big)
    assert len(big.encode("utf-8")) > 100_000
    spec = compose_policy([one, PolicyFile(name="two.yml", body=big.replace("id: g", "id: h"))], policy_id="x")
    assert len(spec.gates) == 800


def test_a_composed_result_names_the_file_its_gate_came_from() -> None:
    spec = compose_policy([_file("a.yml", "a1"), _file("b.yml", "b1")], policy_id="x")
    result = check_policy(spec, paths=["b1.txt"], commands=None, path_source="stop.working_tree")
    failing = [gate for gate in result.results if gate.outcome.value == "fail"]
    assert [(gate.gate_id, gate.source) for gate in failing] == [("b1", "b.yml")]


def test_a_single_file_policy_names_no_file() -> None:
    spec = parse_policy(_file("a.yml", "a1").body, source="a.yml")
    result = check_policy(spec, paths=["a1.txt"], commands=None, path_source="stop.working_tree")
    assert result.results[0].source is None


def _judge(gate_id: str, priority: int | None = None) -> str:
    line = f"    priority: {priority}\n" if priority is not None else ""
    return (
        f"  - id: {gate_id}\n"
        "    type: judge\n"
        "    runs: [stop.session]\n"
        "    enforcement: advisory\n"
        f"    rubric: check {gate_id}\n"
        f"{line}"
        f"    message: m\n"
    )


def test_priority_defaults_to_zero_and_keeps_declaration_order() -> None:
    spec = parse_policy('schema_version: "1.0"\npolicy:\n  id: p\ngates:\n' + _judge("a") + _judge("b"), source="p")
    judges = [gate for gate in spec.gates if isinstance(gate, JudgeGate)]
    assert [gate.priority for gate in judges] == [0, 0]
    assert [gate.id for gate in by_priority(judges)] == ["a", "b"]


def test_a_higher_priority_gate_outranks_an_earlier_one() -> None:
    """The point of the field: surviving a cap is something a gate says, not where its file sorts."""
    body = 'schema_version: "1.0"\npolicy:\n  id: p\ngates:\n' + _judge("a") + _judge("b", 5) + _judge("c")
    judges = [gate for gate in parse_policy(body, source="p").gates if isinstance(gate, JudgeGate)]
    assert [gate.id for gate in by_priority(judges)] == ["b", "a", "c"]


def test_priority_survives_composition_across_files() -> None:
    a = PolicyFile(name="10-first.yml", body='schema_version: "1.0"\npolicy:\n  id: a\ngates:\n' + _judge("early"))
    b = PolicyFile(name="90-last.yml", body='schema_version: "1.0"\npolicy:\n  id: b\ngates:\n' + _judge("late", 1))
    spec = compose_policy([a, b], policy_id="x")
    judges = [gate for gate in spec.gates if isinstance(gate, JudgeGate)]
    assert [gate.id for gate in by_priority(judges)] == ["late", "early"]


def test_priority_must_be_an_integer() -> None:
    body = 'schema_version: "1.0"\npolicy:\n  id: p\ngates:\n' + _judge("a").replace(
        "    message: m\n", "    priority: soon\n    message: m\n"
    )
    with pytest.raises(PolicyError, match="priority"):
        parse_policy(body, source="p")


def test_priority_true_is_a_mistake_not_one() -> None:
    body = 'schema_version: "1.0"\npolicy:\n  id: p\ngates:\n' + _judge("a").replace(
        "    message: m\n", "    priority: true\n    message: m\n"
    )
    with pytest.raises(PolicyError, match="priority"):
        parse_policy(body, source="p")


def test_a_verifier_gate_carries_priority_too() -> None:
    body = (
        'schema_version: "1.0"\npolicy:\n  id: p\ngates:\n'
        "  - id: v\n    type: verifier\n    runs: [stop.verifier]\n"
        "    enforcement: required\n    verifier: v.sh\n    priority: 3\n    message: m\n"
    )
    gate = parse_policy(body, source="p").gates[0]
    assert isinstance(gate, VerifierGate)
    assert gate.priority == 3


def test_a_gate_type_with_no_cap_rejects_priority() -> None:
    """Only the two capped types order; priority on a path gate would mean nothing."""
    body = (
        'schema_version: "1.0"\npolicy:\n  id: p\ngates:\n'
        "  - id: g\n    type: path\n    runs: [stop.working_tree]\n"
        "    enforcement: required\n    forbidden: ['x']\n    priority: 1\n    message: m\n"
    )
    with pytest.raises(PolicyError, match="priority"):
        parse_policy(body, source="p")
