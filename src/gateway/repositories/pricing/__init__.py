"""Data access for the pricing tables.

A package rather than a module beside `base_repository.py`, which is what
`scripts/check_architecture.py` requires of new repository code: a repository
belongs to its domain's package.
"""

from gateway.repositories.pricing.organization_model_pricing_repository import (
    OrganizationModelPricingRepository,
)

__all__ = ["OrganizationModelPricingRepository"]
