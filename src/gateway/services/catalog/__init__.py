"""The caller's model catalog, and the spellings a request may use for a model."""

from gateway.services.catalog._aliases import hide_alias_target
from gateway.services.catalog._dated import dated_builds
from gateway.services.catalog._limits import apply_model_limits
from gateway.services.catalog._metadata import load_catalog_metadata, metadata_entry
from gateway.services.catalog._query import query_catalog
from gateway.services.catalog._selectors import (
    Identities,
    OfferingRow,
    OrganizationSelectors,
    SelectorIndex,
    build_organization_selectors,
    build_selector_index,
    current_selector_index,
    model_selector_for_slug,
    reset_selector_index,
    resolve_catalog_offerings,
    resolve_catalog_selector,
    resolve_undated_selector,
    set_selector_index,
    short_selector_for,
)

__all__ = [
    "Identities",
    "OfferingRow",
    "OrganizationSelectors",
    "SelectorIndex",
    "apply_model_limits",
    "build_organization_selectors",
    "build_selector_index",
    "current_selector_index",
    "dated_builds",
    "hide_alias_target",
    "load_catalog_metadata",
    "metadata_entry",
    "model_selector_for_slug",
    "query_catalog",
    "reset_selector_index",
    "resolve_catalog_offerings",
    "resolve_catalog_selector",
    "resolve_undated_selector",
    "set_selector_index",
    "short_selector_for",
]
