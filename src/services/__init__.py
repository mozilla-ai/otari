"""Backward-compatible aliases for legacy `services.*` imports in tests."""

import sys

from gateway.services import (
    bootstrap_service,
    budget_service,
    pricing_init_service,
    pricing_service,
)

sys.modules[f"{__name__}.bootstrap_service"] = bootstrap_service
sys.modules[f"{__name__}.budget_service"] = budget_service
sys.modules[f"{__name__}.pricing_init_service"] = pricing_init_service
sys.modules[f"{__name__}.pricing_service"] = pricing_service

__all__ = [
    "bootstrap_service",
    "budget_service",
    "pricing_init_service",
    "pricing_service",
]
