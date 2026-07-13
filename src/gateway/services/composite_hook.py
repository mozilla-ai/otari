"""Composite dispatch hook: serve a recognized turn from the gateway.

Gated OFF by default (``OTARI_COMPOSITES_ENABLED``); when off this is an
immediate no-op, so landing it is zero behavior change. When on, on a recognized
*approved* composite at a deterministic turn it returns a synthetic response and
releases the local LLM reservation, so the provider is never called for that
turn. Fail-open by construction: any error, or any non-Serve decision, returns
None and the request proceeds to the provider exactly as today.

Two things are intentionally deferred to the live-serving milestone (validated
with the gateway running against the platform), and are why this stays gated:

- Shadow mode (running ``next_action`` alongside the real provider and reporting
  per-turn match) needs the model's response to compare against; this hook acts
  only on Serve, never on Shadow.
- In hybrid mode ``resolve_request_context`` already placed a reservation on the
  platform, and ``release_reservation`` only refunds the *local* standalone
  reservation. Settling the platform-side hold when a composite serves must be
  coordinated with the platform (via the composite-usage report) before hybrid
  serving is enabled.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse

from gateway.core.env import otari_env
from gateway.services.composite_backend import CompositeBackend, build_composite_backend
from gateway.services.composite_dispatch import Serve, decide
from gateway.services.composite_interpreter import EmitTerminal, EmitToolUse
from gateway.services.composite_key import derive_automation_key
from gateway.services.composite_response import (
    terminal_response,
    tool_use_response,
    tool_use_stream_events,
)

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}
_backend: CompositeBackend | None = None


def composites_enabled() -> bool:
    return otari_env("COMPOSITES_ENABLED", "false").strip().lower() in _TRUE


def _get_backend(config: Any) -> CompositeBackend:
    global _backend
    if _backend is None:
        _backend = build_composite_backend(config)
    return _backend


def reset_backend() -> None:
    """Test/hot-reload hook: drop the cached backend singleton."""
    global _backend
    _backend = None


def _served_report(decision: Serve, session_label: str) -> dict[str, Any]:
    return {
        "composite_program_id": decision.composite.get("composite_program_id"),
        "composite_program_version_id": decision.composite.get("composite_program_version_id"),
        "tier": decision.composite.get("tier", "t0_deterministic"),
        "outcome": "served",
        "shadow": False,
        "session_label": session_label,
        "turns_served": 1,
        "turns_matched": 0,
    }


async def try_serve_composite(
    *,
    request: Any,
    ctx: Any,
    background_tasks: BackgroundTasks,
) -> dict[str, Any] | StreamingResponse | None:
    """Return a served response for a recognized approved composite, else None."""
    if not composites_enabled():
        return None
    automation_key = derive_automation_key(request)
    if not automation_key:
        return None

    try:
        backend = _get_backend(ctx.config)
        composites = await backend.fetch(user_token=ctx.user_token, automation_key=automation_key)
        raw_tools = getattr(request, "tools", None)
        tool_names = (
            [t["name"] for t in raw_tools if isinstance(t, dict) and t.get("name")]
            if isinstance(raw_tools, list)
            else None
        )
        decision = decide(
            composites, request.messages, session_label=automation_key, tool_names=tool_names
        )
        if not isinstance(decision, Serve):
            return None

        action = decision.action
        model = request.model

        # Report the served turn best-effort (never blocks the response).
        if ctx.user_token:
            from gateway.api.routes._platform import _report_platform_composite_usage

            background_tasks.add_task(
                _report_platform_composite_usage,
                ctx.config,
                ctx.user_token,
                _served_report(decision, automation_key),
            )

        # The provider will not be called for this turn; refund the reservation.
        from gateway.api.routes._pipeline import release_reservation

        await release_reservation(ctx)

        if isinstance(action, EmitToolUse):
            if getattr(request, "stream", False):
                events = tool_use_stream_events(
                    model=model, tool_name=action.tool_name, tool_input=action.tool_input
                )

                async def _gen() -> Any:
                    for event in events:
                        yield f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"

                return StreamingResponse(_gen(), media_type="text/event-stream")
            return tool_use_response(
                model=model, tool_name=action.tool_name, tool_input=action.tool_input
            ).model_dump(exclude_none=True)

        if isinstance(action, EmitTerminal) and not getattr(request, "stream", False):
            return terminal_response(model=model, text=action.text).model_dump(exclude_none=True)

        return None
    except Exception:
        logger.warning("composite hook failed; falling through to provider", exc_info=True)
        return None
