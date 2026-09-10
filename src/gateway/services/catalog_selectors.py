"""The short spellings a request may use for a model, and what they resolve to.

A selector is ``instance:model``, with ``model`` the provider's own id, and a
provider's id can be as long as ``accounts/fireworks/models/gpt-oss-120b``.
The catalog groups such ids by the model they name, and this index lets a
caller send what the catalog shows:

- a **short selector**, ``instance:<cleaned id>`` (``fireworks:gpt-oss-120b``),
  which resolves to the provider's full id on that instance; and
- a **model selector**, the catalog's slug alone (``gpt-oss-120b``), which
  resolves to the cheapest offering of that model the deployment serves.

The index is process-wide and rebuilt from the deployment's own catalog view
(the configured instances, priced from the deployment's list and the
defaults) at startup and on a schedule; the catalog route feeds it as well
whenever it builds that view. It is consulted after aliases and static
policies and before the ordinary split, and only for a selector that names
no offering already, so a real selector is never rewritten.

Resolution is synchronous because :func:`provider_kwargs.resolve_provider_selector`
is, which is why the index is a snapshot rather than a lookup.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SelectorIndex:
    """One snapshot of the short spellings in force."""

    full: frozenset[str] = frozenset()
    """Every selector the deployment serves, verbatim."""

    short: dict[str, str] = field(default_factory=dict)
    """``instance:<cleaned id>``, lowercased, to the full selector, where the short id is unambiguous there."""

    models: dict[str, str] = field(default_factory=dict)
    """Catalog slug to the offering a bare slug resolves to."""

    model_selectors: dict[str, str] = field(default_factory=dict)
    """Full selector to its short spelling, for the catalog to show."""


_lock = threading.Lock()
_index = SelectorIndex()


def set_selector_index(index: SelectorIndex) -> None:
    global _index
    with _lock:
        _index = index


def current_selector_index() -> SelectorIndex:
    return _index


def reset_selector_index() -> None:
    set_selector_index(SelectorIndex())


def resolve_catalog_selector(model_selector: str) -> str | None:
    """The full selector a short or model selector stands for, or None.

    None for a selector that already names an offering (nothing to rewrite),
    for one the index does not know, and while the index is empty.
    """
    index = _index
    if model_selector in index.full:
        return None
    # Short spellings are case-insensitive: they are the catalog's, not the
    # provider's, and a provider's own casing is the thing they leave behind.
    spelling = model_selector.lower()
    short = index.short.get(spelling)
    if short is not None:
        return short
    return index.models.get(spelling)


def short_selector_for(full_selector: str) -> str | None:
    """The short spelling the catalog shows for an offering, when it has one."""
    return _index.model_selectors.get(full_selector)


def model_selector_for_slug(slug: str) -> str | None:
    """The offering a bare slug resolves to, when the index knows the slug."""
    return _index.models.get(slug)


def build_selector_index(
    offerings: list[tuple[str, str, str, float | None]],
    identities: dict[str, tuple[str, tuple[str, ...]]],
) -> SelectorIndex:
    """Fold ``(selector, instance, cleaned id, input rate)`` rows and the grouped identities into an index.

    A short spelling is kept only where one offering on the instance cleans to
    it, so two variants of a model on one provider (a base and a quantized
    build cleaning to the same id) get no short form rather than a wrong one.
    A slug resolves to its cheapest priced offering, ties to the first, and to
    the first offering when none is priced.
    """
    full = frozenset(selector for selector, _, _, _ in offerings)
    by_short: dict[str, list[str]] = {}
    for selector, instance, cleaned, _ in offerings:
        by_short.setdefault(f"{instance}:{cleaned}".lower(), []).append(selector)
    short = {spelling: targets[0] for spelling, targets in by_short.items() if len(targets) == 1}
    model_selectors = {target: spelling for spelling, target in short.items()}
    rates = {selector: rate for selector, _, _, rate in offerings}
    models: dict[str, str] = {}
    for slug, (_key, selectors) in identities.items():
        if not selectors:
            continue
        priced = [s for s in selectors if rates.get(s) is not None]
        models[slug] = min(priced, key=lambda s: (rates[s], selectors.index(s))) if priced else selectors[0]
    return SelectorIndex(full=full, short=short, models=models, model_selectors=model_selectors)
