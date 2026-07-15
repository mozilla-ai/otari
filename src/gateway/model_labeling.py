"""Relabeling of the ``model`` field on provider results and stream chunks."""

from typing import Any

__all__ = ["relabel_model"]


def relabel_model(obj: Any, display_model: str) -> Any:
    """Rewrite the ``model`` field on a result or stream chunk in place.

    Used to echo a configured alias back to the caller instead of the real
    upstream model. Handles the top-level ``model`` field (OpenAI chat chunks and
    every non-streaming result object) plus the nested locations streaming start
    events carry it in: Anthropic ``message_start`` (``.message.model``) and the
    Responses events (``.response.model``). Chunks with no model field anywhere
    are left untouched, so this is safe to call on every chunk of any format.
    """
    for holder in (obj, getattr(obj, "message", None), getattr(obj, "response", None)):
        if holder is not None and hasattr(holder, "model"):
            try:
                holder.model = display_model
            except (AttributeError, ValueError):  # frozen / validated field, leave as-is
                pass
    return obj
