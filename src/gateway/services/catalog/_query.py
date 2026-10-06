"""Pure selection over authorized summaries assembled from the catalog's sources."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from gateway.schemas.catalog import (
    CatalogCapability,
    CatalogFacets,
    CatalogModelSummary,
    CatalogPage,
    CatalogQuery,
    CatalogVendorFacet,
)


def _has_capability(model: CatalogModelSummary, capability: CatalogCapability) -> bool:
    if capability == CatalogCapability.OPEN_WEIGHTS:
        return model.open_weights
    return bool(getattr(model.capabilities, capability.value))


def _pricing_matches(model: CatalogModelSummary, pricing: str) -> bool:
    match pricing:
        case "custom":
            return any(source in ("deployment", "organization") for source in model.price_sources)
        case "default":
            return "defaults" in model.price_sources
        case "priced":
            return model.min_input_price_per_million is not None
        case "unpriced":
            return model.unpriced_count > 0
        case _:
            return True


def _source_matches(model: CatalogModelSummary, source: str) -> bool:
    return source == "all" or model.discovered == (source == "discovered")


def _matching_models(
    models: Sequence[CatalogModelSummary], query: CatalogQuery, *, now: datetime | None = None
) -> list[CatalogModelSummary]:
    """Select over the complete authorized catalog, independently of its page."""
    term = (query.search or "").strip().casefold()
    today = (now or datetime.now(UTC)).date()
    released_after = (today - timedelta(days=query.released_within_days)).isoformat()
    released_before = today.isoformat()
    providers = set(query.provider)
    vendors = set(query.vendor)
    input_modalities = set(query.input_modality)
    output_modalities = set(query.output_modality)

    def matches(model: CatalogModelSummary) -> bool:
        if term and not any(
            term in value.casefold()
            for value in (model.name, model.id, model.vendor or "", *model.selectors, *model.providers)
        ):
            return False
        if providers and not providers.intersection(model.providers):
            return False
        if vendors and (model.vendor or "") not in vendors:
            return False
        if not input_modalities.issubset(model.input_modalities):
            return False
        if not output_modalities.issubset(model.output_modalities):
            return False
        if not all(_has_capability(model, capability) for capability in query.capability):
            return False
        if query.min_context and (model.context_window is None or model.context_window < query.min_context):
            return False
        if query.max_input is not None and (
            model.min_input_price_per_million is None or model.min_input_price_per_million > query.max_input
        ):
            return False
        if not _pricing_matches(model, query.pricing) or not _source_matches(model, query.source):
            return False
        return not query.released_within_days or (
            model.release_date is not None and released_after <= model.release_date <= released_before
        )

    return [model for model in models if matches(model)]


def _sorted_models(models: Sequence[CatalogModelSummary], query: CatalogQuery) -> list[CatalogModelSummary]:
    """Unknown values sort last in either direction; names and ids break ties."""
    ordered = sorted(models, key=lambda model: (model.name.casefold(), model.id))
    if query.sort == "name":
        return ordered if query.direction == "asc" else list(reversed(ordered))

    def value(model: CatalogModelSummary) -> float | int | str | None:
        match query.sort:
            case "released":
                return model.release_date or None
            case "input":
                return model.min_input_price_per_million
            case "output":
                return model.min_output_price_per_million
            case "context":
                return model.context_window
            case _:
                return model.provider_count

    known = [model for model in ordered if value(model) is not None]
    unknown = [model for model in ordered if value(model) is None]
    # Each sort column has one concrete value type; the presence flag leaves
    # the key comparable without turning a missing price into a real zero.
    known.sort(key=lambda model: value(model) or 0, reverse=query.direction == "desc")
    return known + unknown


def _catalog_facets(models: Sequence[CatalogModelSummary]) -> CatalogFacets:
    """Choices come from the whole catalog, so a filter that matches nothing can still be undone."""
    vendor_slugs: dict[str, str | None] = {}
    for model in models:
        vendor = model.vendor or ""
        slug = model.id.partition("/")[0] if model.vendor and "/" in model.id else None
        vendor_slugs.setdefault(vendor, slug)
    return CatalogFacets(
        total_count=len(models),
        providers=sorted({provider for model in models for provider in model.providers}),
        vendors=[
            CatalogVendorFacet(value=vendor, vendor_slug=vendor_slugs[vendor])
            for vendor in sorted(vendor_slugs, key=lambda vendor: (vendor.casefold(), vendor))
        ],
    )


def query_catalog(
    models: Sequence[CatalogModelSummary], query: CatalogQuery, *, now: datetime | None = None
) -> CatalogPage:
    """Filter and count the authorized catalog, then return one deterministically sorted page."""
    matched = _matching_models(models, query, now=now)
    facets = _catalog_facets(models) if query.include_facets else None
    ordered = _sorted_models(matched, query)
    return CatalogPage(count=len(matched), models=ordered[query.skip : query.skip + query.limit], facets=facets)
