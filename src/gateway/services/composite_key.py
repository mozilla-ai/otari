"""Derive the dispatch key for an automation turn from what the tenant already
sends, so the tenant needs zero code change (no session_label, no new field).

Two shapes of key, both valid everywhere the compositor keys off an automation
(capture, nomination, dispatch, reporting):

- ``automation:<id>`` — an explicit label the tenant chose to send (``session_label``
  starting with ``automation:``). Robust and human-readable; used when present.
- ``fp:<16 hex>`` — a content fingerprint of the stable request prefix (the system
  prompt plus the sorted tool names). An automation fires the same instructions
  and tool set on every run, so this is stable per automation; chat traffic, whose
  prefix varies per conversation, fingerprints differently every time and is
  filtered out downstream by volume (min-runs). No tenant cooperation required.

The fingerprint uses only the system prompt and tool names, never message
content, so the key itself carries nothing tenant-private beyond what a hash
reveals, matching the capture content boundary (sec 2.7).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

_AUTOMATION_PREFIX = "automation:"
_SEP = "\x00"


def _tool_names(tools: Any) -> list[str]:
    if not isinstance(tools, list):
        return []
    return sorted(str(t["name"]) for t in tools if isinstance(t, dict) and t.get("name"))


def _system_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    # Anthropic allows system as a list of content blocks; normalize stably.
    return json.dumps(system, sort_keys=True, default=str)


def derive_automation_key(request: Any) -> str | None:
    """The stable per-automation dispatch key, or None when the request is not
    automation-shaped (nothing to capture or serve).

    An explicit ``automation:`` session label wins. Otherwise a request must
    carry tools to be a tool-choreography candidate; without tools there is no
    composite to build, so it returns None.
    """
    label = getattr(request, "session_label", None)
    if isinstance(label, str) and label.startswith(_AUTOMATION_PREFIX):
        return label

    tool_names = _tool_names(getattr(request, "tools", None))
    if not tool_names:
        return None

    system_text = _system_text(getattr(request, "system", None))
    digest = hashlib.sha256(_SEP.join([system_text, *tool_names]).encode()).hexdigest()[:16]
    return f"fp:{digest}"
