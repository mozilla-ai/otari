"""Capture an automation's spec from the traffic the gateway proxies.

The compositor's detector must work from what the gateway already sees on the
request path (the system prompt, the tool definitions, the tool_use
choreography), never from the tenant's code or database. This module writes
those captures to a gateway-local store so the synthesizer (which runs
gateway-side, where content lives) can reason about the automation's intent.
Only the derived declarative plan ever leaves the gateway (sec 2.7); the raw
captures stay here.

Gated by ``OTARI_COMPOSITE_CAPTURE`` (default off). Bounded per automation so a
hot automation cannot grow the store without limit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from gateway.core.env import otari_env
from gateway.services.composite_key import derive_automation_key

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}
_MAX_CAPTURES_PER_AUTOMATION = 20


def capture_enabled() -> bool:
    return otari_env("COMPOSITE_CAPTURE", "false").strip().lower() in _TRUE


def _capture_dir() -> str:
    return otari_env("COMPOSITE_CAPTURE_DIR", "/app/captures")


def _tool_summary(tools: Any) -> list[dict[str, Any]]:
    """Keep tool name + description + input-schema shape; drop nothing sensitive
    that isn't already in the prompt, but this keeps captures compact."""
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for t in tools:
        if not isinstance(t, dict):
            continue
        out.append(
            {
                "name": t.get("name"),
                "description": t.get("description"),
                "input_schema": t.get("input_schema") or t.get("parameters"),
            }
        )
    return out


def capture_automation_request(request: Any) -> None:
    """Append this automation turn's spec to the gateway-local capture store."""
    if not capture_enabled():
        return
    automation_key = derive_automation_key(request)
    if automation_key is None:
        return
    try:
        cap_dir = _capture_dir()
        os.makedirs(cap_dir, exist_ok=True)
        key = hashlib.sha256(automation_key.encode()).hexdigest()[:16]
        path = os.path.join(cap_dir, f"{key}.jsonl")
        if os.path.exists(path) and sum(1 for _ in open(path)) >= _MAX_CAPTURES_PER_AUTOMATION:
            return
        record = {
            "automation_key": automation_key,
            "session_label": getattr(request, "session_label", None),
            "model": getattr(request, "model", None),
            "system": getattr(request, "system", None),
            "tools": _tool_summary(getattr(request, "tools", None)),
            "messages": getattr(request, "messages", None),
        }
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.warning("automation capture failed; ignoring", exc_info=True)
