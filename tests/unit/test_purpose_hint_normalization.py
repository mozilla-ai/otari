"""Purpose-hint normalization for the Reprise v0 fingerprint (otari-ai#1647).

``hint_hash`` is a fingerprint input, so anything that moves it regroups every
observation in the deployment. Three properties keep it stable against changes
the customer never made: declaration order does not count, the deployment's
header does not count, and a hint the caller never declared hashes as a sentinel
instead of as whatever text the deployment resolved for it. The text the model
actually saw stays visible as ``injected_block_hash``, which is expected to move
in exactly the cases the key must not.
"""

import pytest

from gateway.core.observation import UNDECLARED_HINT, hint_hash, text_hash
from gateway.services.mcp_loop import inject_purpose_hints
from gateway.services.tool_format import (
    build_hint_block,
    inject_purpose_hints_anthropic,
    inject_purpose_hints_responses,
    normalize_purpose_hints,
)

_DECLARED = ("github", "slack")


def _hint_hash(hints: list[tuple[str, str]], declared: tuple[str, ...] = _DECLARED) -> str:
    return hint_hash(normalize_purpose_hints(hints, caller_declared=set(declared)))


def test_declaration_order_does_not_change_the_hint_hash() -> None:
    forward = [("github", "issues and PRs"), ("slack", "team chat")]
    reversed_order = [("slack", "team chat"), ("github", "issues and PRs")]

    assert _hint_hash(forward) == _hint_hash(reversed_order)


def test_injected_block_keeps_the_callers_order() -> None:
    """Only the hash sorts: the model sees the servers in the declared order."""
    block = build_hint_block([("slack", "team chat"), ("github", "issues and PRs")])

    assert block.index("- slack:") < block.index("- github:")


def test_header_changes_the_block_but_not_the_hint_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OTARI_TOOLS_HEADER`` is a deployment knob, so it stays out of the key."""
    hints = [("github", "issues and PRs")]

    monkeypatch.setenv("OTARI_TOOLS_HEADER", "Tools at your disposal:")
    block_with_env_header = build_hint_block(hints)
    key_with_env_header = _hint_hash(hints)

    monkeypatch.delenv("OTARI_TOOLS_HEADER")
    block_with_default_header = build_hint_block(hints)
    key_with_default_header = _hint_hash(hints)

    assert key_with_env_header == key_with_default_header
    assert text_hash(block_with_env_header) != text_hash(block_with_default_header)


def test_undeclared_hint_normalizes_to_the_sentinel() -> None:
    normalized = normalize_purpose_hints(
        [("github", "issues and PRs"), ("otari_code_execution", "run python")],
        caller_declared={"github"},
    )

    assert normalized == [("github", "issues and PRs"), ("otari_code_execution", UNDECLARED_HINT)]


def test_editing_a_resolved_default_moves_the_block_but_not_the_hint_hash() -> None:
    """An operator editing the sandbox hint in the dashboard must not reset the piles."""
    before = [("otari_code_execution", "Run Python in a sandbox.")]
    after = [("otari_code_execution", "Execute code in an isolated sandbox.")]
    declared: set[str] = set()

    key_before = hint_hash(normalize_purpose_hints(before, caller_declared=declared))
    key_after = hint_hash(normalize_purpose_hints(after, caller_declared=declared))
    assert key_before == key_after

    # ...and the edit is still visible, because the payload records the block as sent.
    assert text_hash(build_hint_block(before)) != text_hash(build_hint_block(after))


def test_declared_hint_text_is_part_of_the_key() -> None:
    """The sentinel replaces only what the caller left out, not what it wrote."""
    assert _hint_hash([("github", "issues and PRs")]) != _hint_hash([("github", "code review")])


def test_all_three_formats_inject_the_same_block() -> None:
    """``injected_block_hash`` is format-neutral, so the three assemblies must agree."""
    hints = [("github", "issues and PRs"), ("slack", "team chat")]
    block = build_hint_block(hints)

    chat = inject_purpose_hints([{"role": "user", "content": "hi"}], hints)
    anthropic = inject_purpose_hints_anthropic({}, hints)
    responses = inject_purpose_hints_responses({}, hints)

    assert chat[0] == {"role": "system", "content": block}
    assert anthropic["system"] == block
    assert responses["instructions"] == block
