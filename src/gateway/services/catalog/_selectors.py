"""The spellings a request may use for a model, and what they resolve to.

A selector is ``instance:model``, with ``model`` the provider's own id, and a
provider's id can be as long as ``accounts/fireworks/models/gpt-oss-120b``.
The catalog groups such ids by the model they name, and this index lets a
caller send what the catalog shows:

- a **model selector**, the catalog's id (``deepseek/deepseek-v4.1-flash``, or
  the bare slug where the vendor is unknown), which resolves to the cheapest
  offering of that model the caller can reach. The vendor's own provider wins
  where it serves the model, so ``openai/gpt-4o`` reaches OpenAI while OpenAI is
  configured and the cheapest reseller otherwise;
- a **pinned selector**, ``instance:<catalog id>``
  (``nebius:deepseek/deepseek-v4.1-flash``), which resolves to that model's
  cheapest offering on that instance and never leaves it.

The index is process-wide and rebuilt at startup and on a schedule from the
deployment's own catalog view (the configured instances, priced from the
deployment's list and the defaults) plus one view per organization for the
models it offers on its own provider keys, priced at that organization's rates.
An organization's view is consulted only for its own callers, so a model one
tenant reaches through its key is never where another tenant's selector lands.
The index is consulted after aliases and static policies and before the
ordinary split, and only for a selector that names no offering already, so a
real selector is never rewritten.

Resolution is synchronous because :func:`provider_kwargs.resolve_provider_selector`
is, which is why the index is a snapshot rather than a lookup.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from gateway.services.model_identity import own_providers_for_vendor_slug


@dataclass(frozen=True)
class OfferingRow:
    """One selector the deployment serves, as the index needs to see it."""

    selector: str
    instance: str
    """The ``providers:`` name, or the provider of an organization's key, which is what a selector is addressed by."""
    provider_type: str
    """The any-llm provider the instance dispatches to."""
    input_rate: float | None


Identities = Mapping[str, tuple[str, tuple[str, ...]]]
"""Catalog id to the grouping key and the selectors of one model, as ``group_offerings`` folds them."""


@dataclass(frozen=True)
class _Maps:
    full: frozenset[str]
    pinned: dict[str, str]
    models: dict[str, str]
    model_selectors: dict[str, str]


@dataclass(frozen=True)
class OrganizationSelectors:
    """What one organization's own offerings add to the deployment's spellings.

    Built over the union of the deployment's offerings and the organization's,
    at the organization's rates, and narrowed to the instances and models the
    organization's offerings touch: an entry here is authoritative for that
    organization's callers, and anything absent falls through to the
    deployment's view.
    """

    full: frozenset[str] = frozenset()
    """The organization's own selectors, verbatim."""

    pinned: dict[str, str] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    model_selectors: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectorIndex:
    """One snapshot of the spellings in force."""

    full: frozenset[str] = frozenset()
    """Every selector the deployment serves, verbatim."""

    pinned: dict[str, str] = field(default_factory=dict)
    """``instance:<catalog id>``, lowercased, to the model's cheapest offering on that instance."""

    models: dict[str, str] = field(default_factory=dict)
    """Catalog id, lowercased, to the offering it resolves to."""

    model_selectors: dict[str, str] = field(default_factory=dict)
    """Full selector to the pinned spelling the catalog shows for it, where it is the offering that spelling reaches."""

    organizations: dict[uuid.UUID, OrganizationSelectors] = field(default_factory=dict)
    """Each organization's own view, keyed by organization."""

    workspace_organization: dict[uuid.UUID, uuid.UUID] = field(default_factory=dict)
    """Which organization a workspace belongs to, for a caller that knows only its workspace."""


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


def _organization_view(
    index: SelectorIndex, workspace_id: uuid.UUID | None, organization_id: uuid.UUID | None
) -> OrganizationSelectors | None:
    if organization_id is None and workspace_id is not None:
        organization_id = index.workspace_organization.get(workspace_id)
    if organization_id is None:
        return None
    return index.organizations.get(organization_id)


def resolve_catalog_selector(
    model_selector: str,
    *,
    workspace_id: uuid.UUID | None = None,
    organization_id: uuid.UUID | None = None,
) -> str | None:
    """The full selector a pinned or model selector stands for, or None.

    None for a selector that already names an offering (nothing to rewrite),
    for one the index does not know, and while the index is empty. The caller's
    organization, named directly or through its workspace, is answered from its
    own view first and the deployment's second.
    """
    index = _index
    view = _organization_view(index, workspace_id, organization_id)
    org_full = view.full if view is not None else frozenset()
    if model_selector in index.full or model_selector in org_full:
        return None
    # ``openai/gpt-4o`` is both the legacy spelling of an offering on the
    # ``openai`` instance and, read as vendor/model, a catalog id. Where the
    # instance really serves that id, the caller meant the offering.
    prefix, slash, rest = model_selector.partition("/")
    if slash and (f"{prefix}:{rest}" in index.full or f"{prefix}:{rest}" in org_full):
        return None
    # Spellings are case-insensitive: they are the catalog's, not the
    # provider's, and a provider's own casing is the thing they leave behind.
    spelling = model_selector.lower()
    for attr in ("pinned", "models"):
        if view is not None and (hit := getattr(view, attr).get(spelling)) is not None:
            return str(hit)
        if (hit := getattr(index, attr).get(spelling)) is not None:
            return str(hit)
    return None


def short_selector_for(full_selector: str, *, organization_id: uuid.UUID | None = None) -> str | None:
    """The spelling the catalog shows for an offering, when it has one."""
    view = _organization_view(_index, None, organization_id)
    if view is not None and (spelling := view.model_selectors.get(full_selector)) is not None:
        return spelling
    return _index.model_selectors.get(full_selector)


def model_selector_for_slug(slug: str, *, organization_id: uuid.UUID | None = None) -> str | None:
    """The offering a catalog id resolves to, when the index knows it."""
    view = _organization_view(_index, None, organization_id)
    if view is not None and (target := view.models.get(slug)) is not None:
        return target
    return _index.models.get(slug)


def _cheapest(selectors: Sequence[str], rates: Mapping[str, float | None]) -> str:
    """The cheapest priced selector, ties to the first; the first when none is priced."""
    priced = [selector for selector in selectors if rates.get(selector) is not None]
    if not priced:
        return selectors[0]
    return min(priced, key=lambda selector: (rates[selector], selectors.index(selector)))


def _build_maps(offerings: Sequence[OfferingRow], identities: Identities) -> _Maps:
    full = frozenset(row.selector for row in offerings)
    rates = {row.selector: row.input_rate for row in offerings}
    serves = {row.selector: {row.instance.lower(), row.provider_type.lower()} for row in offerings}
    instance_of = {row.selector: row.instance for row in offerings}
    pinned: dict[str, str] = {}
    models: dict[str, str] = {}
    for slug, (_key, members) in identities.items():
        selectors = tuple(selector for selector in members if selector in full)
        if not selectors:
            continue
        vendor, slash, _rest = slug.partition("/")
        candidates = selectors
        if slash:
            own = {vendor.lower()} | own_providers_for_vendor_slug(vendor)
            by_vendor = tuple(selector for selector in selectors if own & serves[selector])
            candidates = by_vendor or selectors
        models[slug] = _cheapest(candidates, rates)
        by_instance: dict[str, list[str]] = {}
        for selector in selectors:
            by_instance.setdefault(instance_of[selector], []).append(selector)
        for instance, siblings in by_instance.items():
            pinned[f"{instance}:{slug}".lower()] = _cheapest(siblings, rates)
    model_selectors = {target: spelling for spelling, target in pinned.items()}
    return _Maps(full=full, pinned=pinned, models=models, model_selectors=model_selectors)


def build_selector_index(
    offerings: Sequence[OfferingRow],
    identities: Identities,
    *,
    organizations: Mapping[uuid.UUID, OrganizationSelectors] | None = None,
    workspace_organization: Mapping[uuid.UUID, uuid.UUID] | None = None,
) -> SelectorIndex:
    """Fold the deployment's offerings and the grouped identities into an index.

    A catalog id resolves to its cheapest priced offering, ties to the first,
    and to the first offering when none is priced; a pinned spelling applies
    the same rule to one instance's offerings of the model (two builds of a
    model on one provider are one model by the catalog's grouping), and is
    what the catalog advertises for the offering it lands on.

    A catalog id whose vendor is also a provider (``openai/gpt-4o``) is
    answered from that provider's own offerings where it has any, because the
    caller who names OpenAI's model while OpenAI serves it means OpenAI's price,
    and from every offering otherwise, so the model is reached through whoever
    resells it.
    """
    maps = _build_maps(offerings, identities)
    return SelectorIndex(
        full=maps.full,
        pinned=maps.pinned,
        models=maps.models,
        model_selectors=maps.model_selectors,
        organizations=dict(organizations or {}),
        workspace_organization=dict(workspace_organization or {}),
    )


def build_organization_selectors(
    deployment: Sequence[OfferingRow],
    organization: Sequence[OfferingRow],
    identities: Identities,
) -> OrganizationSelectors:
    """One organization's view: the union of both offering sets, kept where the organization's touch it.

    ``identities`` groups the union, and every row carries the rate this
    organization pays, so a catalog id the organization offers a cheaper
    offering of resolves to its own key, and one it offers nothing of is left
    to the deployment's view.
    """
    maps = _build_maps([*deployment, *organization], identities)
    own = frozenset(row.selector for row in organization)
    instances = {row.instance.lower() for row in organization}
    touched = {slug for slug, (_key, members) in identities.items() if own.intersection(members)}
    touched_selectors = {selector for slug in touched for selector in identities[slug][1]} | own
    return OrganizationSelectors(
        full=own,
        pinned={
            spelling: target for spelling, target in maps.pinned.items() if spelling.partition(":")[0] in instances
        },
        models={slug: target for slug, target in maps.models.items() if slug in touched},
        model_selectors={
            selector: spelling for selector, spelling in maps.model_selectors.items() if selector in touched_selectors
        },
    )
