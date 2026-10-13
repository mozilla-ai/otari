"""A provider's undated name for a dated build it offers.

Anthropic lists ``claude-haiku-4-5-20251001`` and documents ``claude-haiku-4-5``
as its alias, and OpenAI does the same with ``gpt-4o-2024-08-06``. A key whose
offered models are the dated builds would otherwise refuse the documented name.
The rule is provider-agnostic and conservative: the undated name stands for a
dated build only where exactly one offered id is that name plus a real
calendar date, and never where the undated name is itself offered.
"""

import re
from collections.abc import Iterable
from datetime import date

_DATED = re.compile(r"^(?P<name>.+)-(?P<stamp>\d{8}|\d{4}-\d{2}-\d{2})$")


def undated_name(model_id: str) -> str | None:
    """``model_id`` without its trailing ``-YYYYMMDD`` or ``-YYYY-MM-DD`` release date, or None."""
    match = _DATED.match(model_id)
    if match is None:
        return None
    stamp = match["stamp"].replace("-", "")
    try:
        date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))
    except ValueError:
        return None
    return match["name"]


def dated_builds(model_id: str, offered: Iterable[str]) -> tuple[str, ...]:
    """The ``offered`` ids that are ``model_id`` plus a release date, A-Z."""
    return tuple(sorted({candidate for candidate in offered if undated_name(candidate) == model_id}))


def undated_spellings(selectors: Iterable[str]) -> dict[str, str]:
    """``instance:<name>`` to the one ``instance:<name>-<date>`` among ``selectors`` it stands for.

    A name with two dated builds on one instance, or one the instance offers
    undated as well, gets no entry.
    """
    offered = set(selectors)
    builds: dict[str, list[str]] = {}
    for selector in offered:
        instance, separator, model_id = selector.partition(":")
        name = undated_name(model_id) if separator else None
        if name is not None:
            builds.setdefault(f"{instance}:{name}", []).append(selector)
    return {
        spelling: candidates[0]
        for spelling, candidates in builds.items()
        if len(candidates) == 1 and spelling not in offered
    }
