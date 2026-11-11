"""The caller's model catalog, and the spellings a request may use for a model."""

from gateway.services.catalog._aliases import hide_alias_target
from gateway.services.catalog._dated import dated_builds
from gateway.services.catalog._limits import ModelArchitecture, apply_model_metadata, served_context_window
from gateway.services.catalog._metadata import load_catalog_metadata, metadata_entry
from gateway.services.catalog._model_type import (
    catalog_model_type,
    model_type,
    not_chat_detail,
    not_embedding_detail,
    offered_model_types,
    offering_model_type,
)
from gateway.services.catalog._query import common_price_unit, query_catalog
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
from gateway.services.catalog._served import (
    EngineSettings,
    KeyOwner,
    load_served_models_at_startup,
    read_engine_settings,
    record_served_models,
    reset_served_models,
    run_served_model_refresher,
)
from gateway.types.model_type import ModelType

__all__ = [
    "CatalogModel",
    "EngineSettings",
    "Identities",
    "KeyOwner",
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
    "common_price_unit",
    "current_selector_index",
    "dated_builds",
    "hide_alias_target",
    "load_catalog_metadata",
    "load_served_models_at_startup",
    "metadata_entry",
    "model_selector_for_slug",
    "model_type",
    "not_chat_detail",
    "not_embedding_detail",
    "offered_model_types",
    "offering_model_type",
    "offering_input_rate",
    "organization_offerings",
    "query_catalog",
    "read_engine_settings",
    "record_served_models",
    "reset_selector_index",
    "reset_served_models",
    "resolve_catalog_offerings",
    "resolve_catalog_selector",
    "resolve_undated_selector",
    "run_served_model_refresher",
    "served_context_window",
    "set_selector_index",
    "short_selector_for",
]
