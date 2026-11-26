"""Which priced model a harness's own name for a model stands for.

A harness names models its own way: Claude Code by Anthropic's id, a
``<provider>:<model>`` id or one of three aliases, pi by its provider prefix, an Otari id and sometimes a thinking
level. Pricing reads ``<provider>:<model>``. Pure: no lookup happens here, so
a key this returns may still have no rate.
"""

import re
from collections.abc import Iterable

CLAUDE_CODE = "claude-code"
_CLAUDE_CODE_ALIASES = ("haiku", "sonnet", "opus")
# Claude Code's marker for a context window variant, as in ``claude-opus-5[1m]``.
_VARIANT_SUFFIX = re.compile(r"\[[^\]]*\]$")
_PI_PREFIX = "otari/"
_PI_THINKING_LEVELS = frozenset({"off", "minimal", "low", "medium", "high", "xhigh"})
_OTARI_MODEL_ID = re.compile(r"^[^:\s]+:\S+$")


def model_key(harness: str, model: str) -> str | None:
    """The ``<provider>:<model>`` a harness's model name is priced as, or None where it names none."""
    if harness == CLAUDE_CODE:
        name = _VARIANT_SUFFIX.sub("", model.strip())
        if not name or name.lower() in _CLAUDE_CODE_ALIASES:
            return None
        return name if _OTARI_MODEL_ID.match(name) else f"anthropic:{name}"
    name = model.strip().removeprefix(_PI_PREFIX)
    head, separator, level = name.rpartition(":")
    if separator and level in _PI_THINKING_LEVELS and ":" in head:
        name = head
    return name if _OTARI_MODEL_ID.match(name) else None


def baseline_key(harness: str, baseline_model: str, *, parent_model: str, ran_on: Iterable[str]) -> str | None:
    """The priced model a decision's baseline stands for, or None where nothing says.

    A Claude Code alias names a family, not a model. It is read as the parent's
    model where the parent is of that family, then as a model of that family
    the subagent ran on.
    """
    alias = baseline_model.strip().lower()
    if harness != CLAUDE_CODE or alias not in _CLAUDE_CODE_ALIASES:
        return model_key(harness, baseline_model)
    for named in (parent_model, *ran_on):
        if alias in named.lower():
            key = model_key(harness, named)
            if key is not None:
                return key
    return None
