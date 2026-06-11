"""Tool-definition format conversion + per-format purpose-hint injection.

The duck-typed pool interface (``MCPClientPool`` / ``SandboxBackend`` /
``WebSearchBackend``) exposes ``openai_tools`` in the OpenAI Chat-Completions
shape::

    [{"type": "function", "function": {"name", "description", "parameters"}}]

The Anthropic Messages API and OpenAI Responses API expect different shapes.
Rather than push per-format properties down into the backends, we convert at
the boundary — each route handler converts the pool's OpenAI shape into the
shape its endpoint accepts. Backends stay format-agnostic.

For purpose-hint injection, the chat-completions helper
(:func:`gateway.services.mcp_loop.inject_purpose_hints`) merges hints into the
``messages`` system role. Anthropic and Responses both surface "system" as a
separate top-level field, so they get their own per-format helpers here.
"""

from __future__ import annotations

from typing import Any

from gateway.core.env import otari_env
from gateway.services.mcp_loop import PURPOSE_HINT_HEADER


def openai_to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-shape function tools into Anthropic Messages shape.

    Input::

        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]

    Output::

        [{"name": ..., "description": ..., "input_schema": {...}}]

    Entries that aren't OpenAI function tools (e.g. already-Anthropic shape, or
    a custom tool type the pool happens to expose) pass through unchanged so
    that mixed lists don't break.
    """
    out: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            converted: dict[str, Any] = {"name": fn.get("name", "")}
            if "description" in fn:
                converted["description"] = fn["description"]
            if "parameters" in fn:
                converted["input_schema"] = fn["parameters"]
            else:
                # Anthropic requires input_schema even when no parameters.
                converted["input_schema"] = {"type": "object", "properties": {}}
            out.append(converted)
        else:
            out.append(tool)
    return out


def _resolve_header(header: str | None) -> str:
    """Pick the effective purpose-hint preamble header.

    Priority: ``header`` arg → ``OTARI_TOOLS_HEADER`` env → built-in default.
    Mirrors :func:`gateway.services.mcp_loop.inject_purpose_hints`.
    """
    return header or otari_env("TOOLS_HEADER") or PURPOSE_HINT_HEADER


def _build_hint_block(hints: list[tuple[str, str]], header: str | None) -> str:
    lines = [_resolve_header(header)]
    for name, hint in hints:
        lines.append(f"- {name}: {hint}")
    return "\n".join(lines)


def inject_purpose_hints_anthropic(
    call_kwargs: dict[str, Any],
    hints: list[tuple[str, str]],
    *,
    header: str | None = None,
) -> dict[str, Any]:
    """Prepend per-tool purpose hints to the Anthropic ``system`` field.

    ``system`` may be a plain string or a list of content-block dicts (used for
    cache_control). The hint block is prepended in both cases — a string
    becomes ``"<hint>\\n\\n<existing>"``, a list gets a leading text block.

    Mutates ``call_kwargs`` in place and returns it for chaining. No-op when
    ``hints`` is empty so the caller can always wrap.
    """
    if not hints:
        return call_kwargs

    block = _build_hint_block(hints, header)
    existing = call_kwargs.get("system")

    if existing is None:
        call_kwargs["system"] = block
    elif isinstance(existing, str):
        call_kwargs["system"] = f"{block}\n\n{existing}" if existing else block
    elif isinstance(existing, list):
        call_kwargs["system"] = [{"type": "text", "text": block}, *existing]
    else:
        call_kwargs["system"] = block
    return call_kwargs


def openai_to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI Chat-Completions function tools to OpenAI Responses shape.

    Input (Chat Completions, nested)::

        [{"type": "function", "function": {"name", "description", "parameters"}}]

    Output (Responses, flat)::

        [{"type": "function", "name", "description", "parameters"}]

    Entries already in flat shape (no nested ``function`` dict) pass through
    unchanged, so a mixed list works.
    """
    out: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            converted: dict[str, Any] = {"type": "function", "name": fn.get("name", "")}
            if "description" in fn:
                converted["description"] = fn["description"]
            if "parameters" in fn:
                converted["parameters"] = fn["parameters"]
            out.append(converted)
        else:
            out.append(tool)
    return out


def inject_purpose_hints_responses(
    call_kwargs: dict[str, Any],
    hints: list[tuple[str, str]],
    *,
    header: str | None = None,
) -> dict[str, Any]:
    """Prepend per-tool purpose hints to the Responses API ``instructions`` field.

    Mutates ``call_kwargs`` in place. No-op when ``hints`` is empty.
    The Responses ``instructions`` field is a plain string (no content-block
    list form), so the handling is simpler than the Anthropic case.
    """
    if not hints:
        return call_kwargs

    block = _build_hint_block(hints, header)
    existing = call_kwargs.get("instructions")

    if not existing:
        call_kwargs["instructions"] = block
    elif isinstance(existing, str):
        call_kwargs["instructions"] = f"{block}\n\n{existing}"
    else:
        # Defensive: Responses spec types it as `str | None`, but if a future
        # version of the SDK widens it, fall back to a string replacement
        # rather than silently dropping the hints.
        call_kwargs["instructions"] = block
    return call_kwargs
