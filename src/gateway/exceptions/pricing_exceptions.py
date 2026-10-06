"""Errors an organization's pricing overrides raise, and the HTTP status each carries."""

from gateway.exceptions import TenancyConflictError, TenancyForbiddenError, TenancyNotFoundError


class OrganizationPricingNotFoundError(TenancyNotFoundError):
    """No pricing override with this id in the caller's organization.

    One status for "never existed" and "belongs to another organization", the same
    rule the base class states: a distinguishable 404 would tell one tenant which
    ids exist in another.
    """

    def __init__(self, pricing_id: object):
        super().__init__(f"Pricing override {pricing_id} not found")


class OrganizationPricingOverlapError(TenancyConflictError):
    """A period that would overlap one this organization already priced.

    ``model_pricing`` lets a later row shadow an earlier one, because a catalog is
    re-imported wholesale. An override is a commitment for a period, so two
    periods covering one instant is an unanswerable question rather than a newest
    wins rule, and it is refused with both periods named.
    """

    def __init__(self, model_key: str, existing_period: str):
        super().__init__(
            f"An override for '{model_key}' already covers part of that period ({existing_period}). "
            "Change this period, or edit the existing override instead."
        )


class OrganizationPricingManagedModelError(TenancyForbiddenError):
    """An override aimed at a model the deployment, not the organization, pays for.

    An override is how an organization records what it pays for a model it
    supplies the provider key for. Two cases raise this: a model addressed through
    a ``config.providers`` instance, and a bare ``provider:model`` key that the
    bound ``ModelProviderPort`` would serve on a deployment-owned hosted credential
    because a workspace lacks a usable BYO key. Both mean the deployment holds the
    upstream credential and settles the upstream bill, so the rate is the
    deployment price list's and a tenant-set rate would decide what that
    deployment charges itself. A zero is the sharp end of it, since cost is also
    what a budget counts down.

    Withheld from an organization manager, not from everyone: a deployment
    operator is the party that pays, so the standalone deployment whose one
    administrator is also its only tenant keeps setting its own rates exactly as
    before (otari-ai#2095).
    """

    def __init__(self, model_key: str):
        super().__init__(
            f"'{model_key}' resolves on a credential this deployment, not your organization, supplies, "
            "so its rate is set on the deployment price list rather than per organization. An override "
            "applies to a model your organization supplies its own provider key for."
        )


__all__ = [
    "OrganizationPricingManagedModelError",
    "OrganizationPricingNotFoundError",
    "OrganizationPricingOverlapError",
]
