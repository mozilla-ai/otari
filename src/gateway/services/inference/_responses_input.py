"""Take the output items the gateway minted itself back off an inbound Responses ``input``.

The documented way to continue a Responses conversation is to append the
previous ``response.output`` to the next ``input``, and the gateway has no
``previous_response_id`` support to do that server-side. An echoed turn would
otherwise ship a ``web_search_call`` to a provider that never declared a
web-search tool, or a reasoning id no upstream has stored. Each item is
recognized only by its id's gateway prefix, because OpenAI's own items are
legitimately echoed to OpenAI and must survive.
"""

from __future__ import annotations

from typing import Any

from gateway.log_config import logger
from gateway.services.inference._responses_bridge import REASONING_ITEM_ID_PREFIX
from gateway.services.tools import CODE_INTERPRETER_CALL_ID_PREFIX, WEB_SEARCH_CALL_ID_PREFIX


def _is_gateway_minted(item: Any, item_type: str, id_prefix: str) -> bool:
    if not isinstance(item, dict) or item.get("type") != item_type:
        return False
    return str(item.get("id") or "").startswith(id_prefix)


def _code_interpreter_call_as_message(item: dict[str, Any]) -> dict[str, Any]:
    """Fold a gateway-minted ``code_interpreter_call`` into an assistant message item.

    Unlike a search, whose results are already in the transcript, an execution's
    logs exist nowhere else, so dropping the item would make the model forget
    what its code printed on the previous turn. The Messages route folds its pair
    the same way (``messages._code_execution_pair_as_text``).
    """
    code = str(item.get("code") or "")
    parts = [f"[code executed]\n```\n{code}\n```"] if code else ["[code executed]"]
    parts.extend(
        f"logs:\n{output['logs']}"
        for output in item.get("outputs") or []
        if isinstance(output, dict) and output.get("type") == "logs" and output.get("logs")
    )
    if item.get("status") == "failed":
        parts.append("status: failed")
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "\n".join(parts)}]}


def strip_gateway_minted_items(input_data: Any) -> Any:
    """Take gateway-minted items back off an inbound ``input``.

    Only touches a list input, and only the items the gateway itself emits, told
    apart by their id prefix so a provider's own survives. A gateway-run search is
    dropped, which loses nothing the model needs because its results are already in
    the transcript. A gateway-run interpreter call is folded into a message instead
    (:func:`_code_interpreter_call_as_message`). The reasoning item the chat bridge
    makes, and any ``item_reference`` to it, is dropped: no upstream stored it, so a
    native provider would answer 404 for the id.
    """
    if not isinstance(input_data, list):
        return input_data
    kept: list[Any] = []
    touched = 0
    for item in input_data:
        if _is_gateway_minted(item, "code_interpreter_call", CODE_INTERPRETER_CALL_ID_PREFIX):
            kept.append(_code_interpreter_call_as_message(item))
            touched += 1
        elif _is_gateway_minted(item, "web_search_call", WEB_SEARCH_CALL_ID_PREFIX) or (
            _is_gateway_minted(item, "reasoning", REASONING_ITEM_ID_PREFIX)
            or _is_gateway_minted(item, "item_reference", REASONING_ITEM_ID_PREFIX)
        ):
            touched += 1
        else:
            kept.append(item)
    if touched:
        logger.debug("Rewrote %d gateway-minted output item(s) on the inbound input", touched)
    return kept
