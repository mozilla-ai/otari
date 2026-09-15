"""The short spellings a request may use for a model, and what they resolve to.

A selector is ``instance:model``, with ``model`` the provider's own id, and a
provider's id can be as long as ``accounts/fireworks/models/gpt-oss-120b``.
The catalog groups such ids by the model they name, and this index lets a
caller send what the catalog shows:

- a **short selector**, ``instance:<cleaned id>`` (``fireworks:glm-5.3``, spelled
  as the catalog spells the name), which resolves to the provider's full id
  on that instance; and
- a **model selector**, the catalog's id (``openai/gpt-oss-120b``, or the
  bare slug where the vendor is unknown), which resolves to the cheapest
  offering of that model the deployment serves, and only ever to one that
  vendor serves where the vendor is also a provider's name.

The index is process-wide and rebuilt from the deployment's own catalog view
(the configured instances, priced from the deployment's list and the
defaults) at startup and on a schedule. It is consulted after aliases and
static policies and before the ordinary split, and only for a selector that
names no offering already, so a real selector is never rewritten.

Resolution is synchronous because :func:`provider_kwargs.resolve_provider_selector`
is, which is why the index is a snapshot rather than a lookup.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

from any_llm import LLMProvider

# Every name any-llm would dispatch on. A slug whose vendor is one of these is
# also a legacy ``provider/model`` request, and the two readings must not be
# allowed to reach different providers; see :func:`build_selector_index`.
_PROVIDER_NAMES = frozenset(provider.value.lower() for provider in LLMProvider)


@dataclass(frozen=True)
class OfferingRow:
    """One selector the deployment serves, as the index needs to see it."""

    selector: str
    instance: str
    """The ``providers:`` name, which is what a selector is addressed by."""
    provider_type: str
    """The any-llm provider the instance dispatches to."""
    cleaned_id: str
    """The model id as the catalog spells it, for the short selector."""
    input_rate: float | None


def _names_a_provider(vendor: str, offerings: Sequence[OfferingRow]) -> bool:
    """Whether a slug's vendor could also be read as a provider to dispatch to."""
    lowered = vendor.lower()
    return lowered in _PROVIDER_NAMES or any(
        lowered in (row.instance.lower(), row.provider_type.lower()) for row in offerings
    )


@dataclass(frozen=True)
class SelectorIndex:
    """One snapshot of the short spellings in force."""

    full: frozenset[str] = frozenset()
    """Every selector the deployment serves, verbatim."""

    short: dict[str, str] = field(default_factory=dict)
    """``instance:<cleaned id>``, lowercased, to the full selector, where the short id is unambiguous there."""

    models: dict[str, str] = field(default_factory=dict)
    """Catalog id, lowercased, to the offering it resolves to."""

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
    # ``openai/gpt-oss-120b`` is both the legacy spelling of an offering on the
    # ``openai`` instance and, read as vendor/model, a catalog id. Where the
    # instance really serves that id, the caller meant the offering.
    prefix, slash, rest = model_selector.partition("/")
    if slash and f"{prefix}:{rest}" in index.full:
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
    offerings: Sequence[OfferingRow],
    identities: dict[str, tuple[str, tuple[str, ...]]],
) -> SelectorIndex:
    """Fold the deployment's offerings and the grouped identities into an index.

    A short spelling is kept only where one offering on the instance cleans to
    it, so two variants of a model on one provider (a base and a quantized
    build cleaning to the same id) get no short form rather than a wrong one.
    A slug resolves to its cheapest priced offering, ties to the first, and to
    the first offering when none is priced.

    A slug whose vendor is also the name of a provider (``openai/gpt-4o``, read
    the other way, is the legacy spelling of a request to OpenAI) is answered
    only from that provider's own offerings, and dropped where it serves none:
    a selector that names a provider reaches that provider or fails, and is
    never redirected to another one.
    """
    full = frozenset(row.selector for row in offerings)
    by_short: dict[str, list[str]] = {}
    for row in offerings:
        by_short.setdefault(f"{row.instance}:{row.cleaned_id}".lower(), []).append(row.selector)
    short = {spelling: targets[0] for spelling, targets in by_short.items() if len(targets) == 1}
    model_selectors = {target: spelling for spelling, target in short.items()}
    rates = {row.selector: row.input_rate for row in offerings}
    serves = {row.selector: {row.instance.lower(), row.provider_type.lower()} for row in offerings}
    models: dict[str, str] = {}
    for slug, (_key, selectors) in identities.items():
        vendor, slash, _rest = slug.partition("/")
        if slash and _names_a_provider(vendor, offerings):
            selectors = tuple(s for s in selectors if vendor.lower() in serves.get(s, frozenset()))
        if not selectors:
            continue
        priced = [s for s in selectors if rates.get(s) is not None]
        models[slug] = min(priced, key=lambda s: (rates[s], selectors.index(s))) if priced else selectors[0]
    return SelectorIndex(full=full, short=short, models=models, model_selectors=model_selectors)
