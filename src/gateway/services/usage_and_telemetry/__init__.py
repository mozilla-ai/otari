"""The usage domain: the rows each request served or imported leaves behind.

``UsageService`` is its service. The bulk mutations are still in the old shape
(``services/usage_admin_service.py``) and move here when the rest of the domain does.
"""

from gateway.repositories.usage_and_telemetry import ServedTraffic, UsageFilters, UsageScope
from gateway.services.usage_and_telemetry._service import UsageService
from gateway.services.usage_and_telemetry._window import (
    MAX_SERIES_POINTS,
    SERIES_TOP_N,
    dense_series,
    error_class_for,
    parse_tag_filters,
    resolve_window,
)

__all__ = [
    "MAX_SERIES_POINTS",
    "SERIES_TOP_N",
    "ServedTraffic",
    "UsageFilters",
    "UsageScope",
    "UsageService",
    "dense_series",
    "error_class_for",
    "parse_tag_filters",
    "resolve_window",
]
