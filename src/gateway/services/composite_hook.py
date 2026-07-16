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

import httpx
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse

from gateway.core.env import otari_env
from gateway.services.composite_backend import CompositeBackend, build_composite_backend
from gateway.services.composite_dispatch import NoDispatch, Serve, ServeT1, decide
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


_DEFAULT_T1_MODEL = "claude-haiku-4-5-20251001"
# A below-frontier judgment emits a small output (a tool call or a short decision),
# so cap max_tokens well under the point where the provider requires streaming for
# "operations that may take longer than 10 minutes". Forwarding the tenant's large
# max_tokens verbatim on this non-streaming self-call made the provider reject it.
_T1_MAX_TOKENS = 4096


def composites_enabled() -> bool:
    return otari_env("COMPOSITES_ENABLED", "false").strip().lower() in _TRUE


def t1_enabled() -> bool:
    """T1 serving (a cheap model for a below-frontier judgment) is opt-in."""
    return otari_env("T1_ENABLED", "false").strip().lower() in _TRUE


def _t1_model(composite: dict[str, Any]) -> str:
    spec = composite.get("model_spec") or {}
    return str(spec.get("model") or otari_env("T1_MODEL", _DEFAULT_T1_MODEL))


def _valid_t1_tool(tool: Any) -> bool:
    """Whether a tool is safe to forward on the cheap-model judgment call.

    The normal request path normalizes tools (``prepare_gateway_tools`` +
    ``openai_to_anthropic_tools``) before dispatch; the T1 self-call sends them
    raw, so a schema-less short-form tool (e.g. a bare ``code_execution`` with no
    ``input_schema``) reaches the provider unnormalized and is rejected, failing
    the whole judgment. Keep only tools the provider accepts as-is: a custom or
    function tool that carries a schema, or a native server tool declared by
    ``type``. A below-frontier judgment does not need the dropped ones.
    """
    if not isinstance(tool, dict):
        return False
    if any(isinstance(tool.get(k), dict) for k in ("input_schema", "parameters", "function")):
        return True
    return isinstance(tool.get("type"), str)


def _sanitize_t1_tools(tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tools, list):
        return None
    kept = [t for t in tools if _valid_t1_tool(t)]
    return kept or None


_T1_KEEP_BLOCK_TYPES = {"text", "tool_use", "tool_result", "image"}


def _sanitize_t1_messages(messages: Any) -> list[dict[str, Any]]:
    """Strip content blocks the cheap-model self-call cannot round-trip.

    The captured conversation can carry blocks produced by the frontier's own
    tooling: thinking / redacted_thinking, server_tool_use, and web-search or
    code-execution result blocks. The gateway's self ``/v1/messages`` path does
    not serialize those, so re-sending them verbatim 502s the T1 call. A
    below-frontier judgment only needs the plain conversational flow, so keep
    text/tool_use/tool_result/image blocks and drop the rest; a message left with
    no content (e.g. an assistant turn that was only thinking) is dropped, which
    keeps tool_use/tool_result pairing intact.
    """
    if not isinstance(messages, list):
        return []
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        kept = [
            b for b in content if not isinstance(b, dict) or b.get("type") in _T1_KEEP_BLOCK_TYPES
        ]
        if kept:
            out.append({**m, "content": kept})
    return out


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
        logger.info(
            "composite dispatch key=%s fetched=%d req_tools=%d executed=%d decision=%s%s",
            automation_key,
            len(composites),
            len(tool_names or []),
            len([b for m in request.messages if m.get("role") == "assistant" for b in (m.get("content") or [])
                 if isinstance(b, dict) and b.get("type") == "tool_use"]),
            type(decision).__name__,
            f"({decision.reason})" if isinstance(decision, NoDispatch) else "",
        )
        if isinstance(decision, ServeT1):
            if not t1_enabled():
                return None  # T1 off: fall through so the frontier serves the judgment
            return await _serve_t1(
                request=request, ctx=ctx, decision=decision,
                automation_key=automation_key, background_tasks=background_tasks,
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


async def _serve_t1(
    *,
    request: Any,
    ctx: Any,
    decision: ServeT1,
    automation_key: str,
    background_tasks: BackgroundTasks,
) -> dict[str, Any] | StreamingResponse | None:
    """Serve a below-frontier judgment turn with a cheap model.

    Calls the gateway's own ``/v1/messages`` with the cheap model and the
    request's own context (reusing all credential/provider machinery), tagged
    ``X-Otari-T1`` so the composite hook is skipped on that inner call (no
    recursion), then serves the cheap model's decision. Any error falls through
    to the frontier.
    """
    cheap = _t1_model(decision.composite)
    self_url = otari_env("T1_SELF_URL", "http://localhost:8000").rstrip("/")
    payload: dict[str, Any] = {
        "model": cheap,
        "max_tokens": min(int(getattr(request, "max_tokens", 0) or 1024), _T1_MAX_TOKENS),
        "messages": _sanitize_t1_messages(request.messages),
    }
    system = getattr(request, "system", None)
    if system:
        payload["system"] = system
    tools = _sanitize_t1_tools(getattr(request, "tools", None))
    if tools:
        payload["tools"] = tools
    headers = {
        "Authorization": f"Bearer {ctx.user_token or ''}",
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "X-Otari-T1": "1",
    }
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(f"{self_url}/v1/messages", json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "T1 cheap-model call failed (status=%s); falling through to the frontier: %s",
            exc.response.status_code,
            exc.response.text[:300],
        )
        return None
    except Exception:
        logger.warning("T1 cheap-model call failed; falling through to the frontier", exc_info=True)
        return None

    content = body.get("content") if isinstance(body, dict) else None
    if not isinstance(content, list):
        return None
    usage = body.get("usage") or {}

    from gateway.api.routes._pipeline import release_reservation

    await release_reservation(ctx)

    if ctx.user_token:
        from gateway.api.routes._platform import _report_platform_composite_usage

        background_tasks.add_task(
            _report_platform_composite_usage,
            ctx.config,
            ctx.user_token,
            {
                "composite_program_id": decision.composite.get("composite_program_id"),
                "composite_program_version_id": decision.composite.get("composite_program_version_id"),
                "tier": "t1_model",
                "outcome": "served",
                "shadow": False,
                "session_label": automation_key,
                "turns_served": 1,
                "turns_matched": 0,
                "nested_input_tokens": int(usage.get("input_tokens", 0) or 0),
                "nested_output_tokens": int(usage.get("output_tokens", 0) or 0),
            },
        )

    model = request.model
    stream = bool(getattr(request, "stream", False))
    tool_use = next((b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"), None)
    if tool_use is not None:
        name = str(tool_use.get("name", ""))
        tool_input = tool_use.get("input") or {}
        if stream:
            events = tool_use_stream_events(model=model, tool_name=name, tool_input=tool_input)

            async def _gen() -> Any:
                for event in events:
                    yield f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"

            return StreamingResponse(_gen(), media_type="text/event-stream")
        return tool_use_response(model=model, tool_name=name, tool_input=tool_input).model_dump(exclude_none=True)

    text = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    if not stream:
        return terminal_response(model=model, text=text).model_dump(exclude_none=True)
    return None
