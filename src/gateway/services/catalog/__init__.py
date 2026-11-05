"""The caller's model catalog, and the spellings a request may use for a model."""

from gateway.services.catalog._aliases import hide_alias_target
from gateway.services.catalog._dated import dated_builds
from gateway.services.catalog._limits import ModelArchitecture, apply_model_metadata
from gateway.services.catalog._metadata import load_catalog_metadata, metadata_entry
from gateway.services.catalog._model_type import (
    catalog_model_type,
    model_type,
    not_chat_detail,
    not_embedding_detail,
    offering_model_type,
)
from gateway.services.catalog._query import query_catalog
from gateway.services.catalog._selectors import (
    CatalogModel,
    Identities,
    OfferingRow,
    OrganizationSelectors,
    SelectorIndex,
    build_organization_selectors,
    build_selector_index,
    catalog_models,
    current_selector_index,
    model_selector_for_slug,
    offering_input_rate,
    organization_offerings,
    reset_selector_index,
    resolve_catalog_offerings,
    resolve_catalog_selector,
    resolve_undated_selector,
    set_selector_index,
    short_selector_for,
)
from gateway.types.model_type import ModelType

__all__ = [
    "CatalogModel",
    "Identities",
    "ModelArchitecture",
    "ModelType",
    "OfferingRow",
    "OrganizationSelectors",
    "SelectorIndex",
    "apply_model_metadata",
    "build_organization_selectors",
    "build_selector_index",
    "catalog_model_type",
    "catalog_models",
    "current_selector_index",
    "dated_builds",
    "hide_alias_target",
    "load_catalog_metadata",
    "metadata_entry",
    "model_selector_for_slug",
    "model_type",
    "not_chat_detail",
    "not_embedding_detail",
    "offering_model_type",
    "offering_input_rate",
    "organization_offerings",
    "query_catalog",
    "reset_selector_index",
    "resolve_catalog_offerings",
    "resolve_catalog_selector",
    "resolve_undated_selector",
    "set_selector_index",
    "short_selector_for",
]
