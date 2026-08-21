"""The two code-execution ceilings, pinned where the person raising one will see it.

`WorkspaceCodeExecutionPolicyUpdate` refuses a limit above the value that could
actually take effect, and both ceilings are derived rather than typed: the
loop's own cap (`mcp_loop.MAX_TOOL_ITERATIONS_CAP`) and the sandbox's default
execution budget (`sandbox_backend.DEFAULT_EXEC_TIMEOUT_S`). The dashboard's
policy card cannot import either, so it repeats both as literals and validates
against them before it calls the API.

That duplication is what this pins. Raising a backend constant without
`web/src/features/tools/WorkspaceCodeExecutionPolicyCard.tsx` following leaves
the form refusing a value the API would accept, with the failure showing up
nowhere. The dashboard suite compares its own literals against the committed
OpenAPI spec, which catches the same drift, but only for someone running vitest;
this catches it for someone editing `mcp_loop.py` and running pytest.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gateway.services.tenancy.workspace_code_execution_policy_service import (
    _MAX_EXEC_TIMEOUT_S,
    _MAX_ITERATIONS,
)

_CARD = (
    Path(__file__).resolve().parents[2] / "web" / "src" / "features" / "tools" / "WorkspaceCodeExecutionPolicyCard.tsx"
)


def _card_constant(name: str) -> int:
    match = re.search(rf"^export const {name} = (\d+)$", _CARD.read_text(), re.MULTILINE)
    assert match is not None, f"{name} is no longer declared in {_CARD.name}; update this test with it"
    return int(match.group(1))


@pytest.mark.parametrize(
    ("name", "server_value"),
    [("MAX_ITERATIONS", _MAX_ITERATIONS), ("MAX_EXEC_TIMEOUT_S", _MAX_EXEC_TIMEOUT_S)],
)
def test_the_card_repeats_the_ceiling_the_server_enforces(name: str, server_value: int) -> None:
    assert _card_constant(name) == server_value, (
        f"{name} in {_CARD.name} is {_card_constant(name)} but the server enforces {server_value}, "
        "so the form and the API disagree about which values are acceptable."
    )
