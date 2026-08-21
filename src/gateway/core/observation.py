"""Format-neutral vocabulary for the Reprise loop-round observation.

Reprise groups repeated automation by a fingerprint computed over a handful of
request properties (otari-ai#1482). The same logical request reaches the gateway
through three wire formats, and a property read "however this format spells it"
hashes differently in each, so one nightly automation driven through two SDKs
would never group and the report would understate its own support. Every value
feeding the fingerprint is therefore normalized to a format-neutral form before
it is hashed; this module owns those forms and the hashes over them.

The accessors that produce the values live where the format is known:
``FormatAdapter`` (``gateway.api.routes._pipeline``) for the request fields only
the route layer still holds unmodified, ``ToolLoopStrategy`` /
``StreamToolLoopStrategy`` (``gateway.services._tool_loop``) for what the loop
sees, and :func:`gateway.services.tool_format.normalize_purpose_hints` for the
purpose-hint block, which is format-neutral already.

Assembling the fingerprint out of these is otari-ai#1648 and emitting the record
is otari-ai#1484, so nothing here is called from a request path yet.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, NamedTuple

UNDECLARED_HINT = "<default>"
"""Stands in for a purpose hint the caller did not declare in the request body.

An undeclared hint resolves to whatever the deployment supplies (dashboard
override, env, YAML, or a built-in default), so hashing the resolved text would
let an operator editing a text field, or a gateway release rewording a built-in,
reset every pile in the deployment with nobody having touched the automation.
The sentinel records "the caller said nothing here", which is the property that
stays true across such an edit. What the model actually saw is still recorded,
unhashed, as ``injected_block_hash`` over the assembled block.
"""


class NormalizedTool(NamedTuple):
    """One tool in the shape all three wire formats agree on.

    The formats wrap the same JSON Schema object differently (nested
    ``function.parameters`` for chat completions, ``input_schema`` for Anthropic
    Messages, flat ``parameters`` for Responses), so each loop strategy unwraps
    its own shape into this triple.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


def normalized_tool(definition: dict[str, Any], schema_key: str) -> NormalizedTool:
    """Read one format's tool definition into a :class:`NormalizedTool`.

    ``schema_key`` names where the format keeps the JSON Schema (``parameters``
    for chat completions and Responses, ``input_schema`` for Anthropic Messages).

    The name falls back to ``type`` when the entry has none. A provider-native
    server tool the gateway never converts still has to contribute an identity,
    which is what :func:`tool_set_hash` counts, and only some of them spell it
    ``name``: Anthropic's carry one (``{"type": "web_search_20250305", "name":
    "web_search"}``) while the Responses ones carry none at all (``{"type":
    "web_search"}``, ``{"type": "file_search", ...}``). Reading ``name`` alone
    collapses every Responses server tool onto the empty string, so a workspace
    swapping one for another would fingerprint as an unchanged tool set. What such
    an entry configures (an ``mcp`` tool's ``server_label``, a ``file_search``
    tool's vector stores) stays out of both hashes, consistent with a key made of
    names.
    """
    schema = definition.get(schema_key)
    return NormalizedTool(
        name=str(definition.get("name") or definition.get("type") or ""),
        description=str(definition.get("description") or ""),
        input_schema=schema if isinstance(schema, dict) else {},
    )


def message_text(content: Any) -> str:
    """Flatten a message-content value to the text the model was shown.

    Covers the two legal shapes every format has for a prompt: a plain string,
    and a list of content blocks (dicts or SDK objects) whose text is joined with
    a blank line, so ``"foo"`` and ``[{"type": "text", "text": "foo"}]`` agree.
    A block carrying no text (an image, an unknown block type) contributes
    nothing, and ``None`` flattens to ``""``, so absent and empty normalize alike.

    ``cache_control`` markers are ignored for free, since only ``text`` is read:
    they move money rather than meaning, and a caller adding one mid-run must not
    split its own pile.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list | tuple):
        return "\n\n".join(text for block in content if (text := _block_text(block)))
    return ""


def _block_text(block: Any) -> str:
    """The text carried by one content block, or ``""`` when it carries none."""
    if isinstance(block, str):
        return block
    text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
    return text if isinstance(text, str) else ""


def text_hash(text: str) -> str:
    """sha256 of one normalized text value.

    Produces ``client_system_hash`` and ``first_user_message_hash`` from the
    per-format accessors, and ``injected_block_hash`` from the assembled
    purpose-hint block exactly as sent. Only the first is a fingerprint input;
    the other two are payload diagnostics.
    """
    return _sha256(text)


def hint_hash(hints: Sequence[tuple[str, str]]) -> str:
    """sha256 over normalized ``(name, hint)`` pairs. A fingerprint input.

    Expects the output of
    :func:`gateway.services.tool_format.normalize_purpose_hints`: sorted, and with
    every hint the caller did not declare replaced by :data:`UNDECLARED_HINT`.
    """
    return _sha256(_canonical_json([list(pair) for pair in hints]))


def tool_set_hash(tools: Sequence[NormalizedTool]) -> str:
    """sha256 over the sorted tool names. A fingerprint input.

    Names only. Descriptions and schemas drift from a third party, since an MCP
    server's ``list_tools()`` is refetched on every request (the pool is rebuilt
    per request), so a server adding one optional schema property, backward
    compatibly and breaking nothing, would otherwise split every pile in the
    workspace from that moment on. Those go to :func:`tool_definitions_hash`.
    """
    return _sha256(_canonical_json(sorted(tool.name for tool in tools)))


def tool_definitions_hash(tools: Sequence[NormalizedTool]) -> str:
    """sha256 over sorted ``(name, description, schema)`` triples. Payload only.

    The schema is canonicalized (sorted keys, no incidental whitespace) so that
    re-serializing an unchanged schema does not read as a different tool.
    """
    triples = sorted([tool.name, tool.description, _canonical_json(tool.input_schema)] for tool in tools)
    return _sha256(_canonical_json(triples))


def _canonical_json(value: Any) -> str:
    """JSON with sorted keys and no incidental whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
