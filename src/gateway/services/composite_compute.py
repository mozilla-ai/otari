"""Resolve ``compute``-node bindings for a composite plan.

A ``compute`` node carries model-authored Python that derives a value from the
turn's visible tool results (for example, extracting the list of message ids from
an unstructured search result so a ``map`` can loop over it). That code is the one
place in the composite path where model-authored code runs, so it runs in the
**credential-free** sandbox (otari-exec) via the same ``SandboxBackend`` the
``code_execution`` tool uses: no network, no filesystem egress, no credentials.
A sandbox escape is therefore not a key leak (docs/compositor-script-driver.md).

This module is the only I/O in the compute path. The interpreter stays pure: the
hook calls ``resolve_bindings`` here, then folds the result into the plan with
``materialize_plan`` before ``next_action``. A binding that fails to resolve (bad
code, sandbox unreachable) is simply omitted, so its ``{"var": ...}`` reference
fails to evaluate downstream and the turn punts to the frontier. Compute never
makes a turn serve on a missing or wrong value.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from gateway.services.composite_interpreter import compute_nodes
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME

logger = logging.getLogger(__name__)

# Marks the JSON the wrapper prints, so we can pick the bound value out of stdout
# regardless of anything the model code itself printed.
_SENTINEL = "__OTARI_COMPUTE_OUTPUT__"

# Cap the total tool-result context embedded into one compute call, so a
# pathologically large history cannot blow up the sandbox payload. Over the cap
# the node is skipped (its binding stays unresolved and the turn punts).
MAX_RESULTS_BYTES = 262144


class SandboxRunner(Protocol):
    """The slice of ``SandboxBackend`` this module needs: run code, get stdout.

    Declared as a Protocol so tests can inject a local executor instead of a live
    otari-exec session, and production passes an entered ``SandboxBackend``.
    """

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


def visible_tool_results(messages: list[dict[str, Any]]) -> list[str]:
    """The tool-result payloads visible so far, in order, as strings.

    This is exactly what a compute node reads as ``results`` (mirrors the
    interpreter's own view of prior results), so the code sees the same bytes the
    model would have."""
    out: list[str] = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        for block in m.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                content = block.get("content")
                out.append(content if isinstance(content, str) else json.dumps(content, default=str))
    return out


def _wrap(code: str, results: list[str], prior: dict[str, Any]) -> str:
    """Wrap the node's code so ``results`` and ``bindings`` are in scope and the
    ``output`` it sets is emitted as a sentinel-tagged JSON line."""
    results_lit = json.dumps(json.dumps(results))
    prior_lit = json.dumps(json.dumps(prior, default=str))
    return (
        "import json as _otari_json\n"
        f"results = _otari_json.loads({results_lit})\n"
        f"bindings = _otari_json.loads({prior_lit})\n"
        "output = None\n"
        f"{code}\n"
        f"print({json.dumps(_SENTINEL)} + _otari_json.dumps(output, default=str))\n"
    )


def _parse_output(stdout: str) -> Any:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_SENTINEL):
            return json.loads(line[len(_SENTINEL):])
    raise ValueError("compute node produced no sentinel output")


async def resolve_bindings(
    plan: dict[str, Any], messages: list[dict[str, Any]], runner: SandboxRunner
) -> dict[str, Any]:
    """Run each top-level ``compute`` node in the sandbox and collect its binding.

    Nodes run in plan order and each sees the bindings resolved before it, so a
    later compute may build on an earlier one. Any node that fails is omitted (the
    plan then punts on its ``var``), never raised: correctness never depends on a
    compute resolving.
    """
    nodes = compute_nodes(plan)
    if not nodes:
        return {}
    results = visible_tool_results(messages)
    if sum(len(r) for r in results) > MAX_RESULTS_BYTES:
        logger.warning("compute context over %d bytes; skipping compute resolution (turn will punt)", MAX_RESULTS_BYTES)
        return {}
    bindings: dict[str, Any] = {}
    for node in nodes:
        bind = node.get("bind")
        code = node.get("code")
        if not isinstance(bind, str) or not isinstance(code, str):
            continue
        try:
            stdout = await runner.call_tool(CODE_EXECUTION_TOOL_NAME, {"code": _wrap(code, results, bindings)})
            bindings[bind] = _parse_output(stdout)
        except Exception:
            logger.warning(
                "compute node %r failed to resolve; leaving unbound so the turn punts", bind, exc_info=True
            )
    return bindings
