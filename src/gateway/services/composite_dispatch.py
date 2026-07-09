"""Composite dispatch decision: recognize-or-punt, serve-or-shadow.

The pure decision the messages.py hook makes per turn (sec 5.1, 5.4, 5.5),
factored out so it is unit-testable in isolation from the request hot path.
Given the fetched composite definitions, the visible messages, and the dispatch
key, it decides whether to serve a synthetic response, run the composite in
shadow, or leave the request untouched (punt to the provider).

Correctness never depends on recognition being complete: any doubt collapses to
NoDispatch, and the provider serves the turn exactly as today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gateway.services.composite_interpreter import (
    Action,
    EmitTerminal,
    EmitToolUse,
    Punt,
    next_action,
    verify_action,
)

_APPROVED = "approved"
_SHADOW = "shadow"


@dataclass(frozen=True)
class Serve:
    """Serve a synthetic response for this turn; skip the provider call."""

    action: EmitToolUse | EmitTerminal
    composite: dict[str, Any]


@dataclass(frozen=True)
class Shadow:
    """Run the provider as normal, but record what the composite would emit."""

    action: Action
    composite: dict[str, Any]


@dataclass(frozen=True)
class NoDispatch:
    """Leave the request untouched; the provider serves the turn."""

    reason: str


Decision = Serve | Shadow | NoDispatch


def _match(composites: list[dict[str, Any]], session_label: str) -> dict[str, Any] | None:
    return next((c for c in composites if c.get("automation_key") == session_label), None)


def decide(
    composites: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    *,
    session_label: str | None,
) -> Decision:
    """Decide how to handle the current turn.

    - No label / no matching composite -> NoDispatch (the fingerprint fallback,
      sec 5.3, is a later layer).
    - Shadow composite -> Shadow (always, even when the composite would punt, so
      punts are recorded as shadow data).
    - Approved composite -> Serve when the interpreter emits a verified action,
      else NoDispatch (recognize-or-punt).
    """
    if not session_label:
        return NoDispatch("no_session_label")

    composite = _match(composites, session_label)
    if composite is None:
        return NoDispatch("no_match")

    plan = composite.get("plan", {})
    action = next_action(plan, messages)

    # A failed mechanical verifier degrades an emit to a punt (sec 5.4): a bad
    # composite falls back to the frontier rather than serving a wrong action.
    if isinstance(action, EmitToolUse):
        verifier = composite.get("verifier_spec", {})
        if not verify_action(action, verifier, plan, messages):
            action = Punt("verifier_failed")

    status = composite.get("status")
    if status == _SHADOW:
        return Shadow(action=action, composite=composite)
    if status == _APPROVED:
        if isinstance(action, Punt):
            return NoDispatch(action.reason)
        return Serve(action=action, composite=composite)
    return NoDispatch("not_dispatchable_status")


def action_matches_model(action: Action, model_tool_use: dict[str, Any] | None) -> bool:
    """Per-turn shadow comparison: did the composite's action match the model's
    actual tool_use this turn? Name-level match (arg-map enrichment tightens this
    later). A composite punt never counts as a match."""
    if not isinstance(action, EmitToolUse):
        return False
    if not model_tool_use or model_tool_use.get("type") != "tool_use":
        return False
    return action.tool_name == model_tool_use.get("name")
