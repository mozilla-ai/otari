"""The tenant-scoped budget surface understands exactly the scopes enforcement does.

``organization_budget_service`` spells the five scope names out rather than
importing ``ScopeType`` from ``scoped_budget_service``, because that module
reaches ``workspace_scope`` and closes an import cycle back through
``tenancy/__init__`` (its own docstring, and
``test_service_module_imports.py``, carry the detail). A second roster is a
second thing to keep in step, and the failure it would cause is quiet: a scope
this surface accepts but resolution does not know is a ceiling that is created,
listed, and never enforced, which fails in the permissive direction.

So the copy is asserted rather than trusted. Importing both modules here is
fine: a test is not part of the graph the cycle runs through.
"""

from typing import get_args

from gateway.models.budgets import SCOPE_TYPES, ScopeType
from gateway.services.tenancy.organization_budget_service import (
    ORGANIZATION_SCOPE_TYPES,
    OrganizationScopeType,
)


def test_the_two_scope_rosters_are_the_same_set() -> None:
    assert set(ORGANIZATION_SCOPE_TYPES) == set(SCOPE_TYPES)


def test_the_literals_agree_so_the_schema_publishes_the_same_values() -> None:
    """Set equality is not enough: the ``Literal`` is what the OpenAPI schema carries.

    A value present in one ``Literal`` and not the other would be refused at one
    boundary and accepted at the other, whatever the tuples say.
    """
    assert set(get_args(OrganizationScopeType)) == set(get_args(ScopeType))


def test_every_scope_name_is_spelled_the_same_on_the_wire() -> None:
    """No renaming between the two surfaces.

    ``scope_type`` is a stored string, so a ceiling written by one surface is read
    by the other and by ``applicable_budgets``. A tidier spelling here would
    orphan every row the deployment surface wrote.
    """
    assert sorted(ORGANIZATION_SCOPE_TYPES) == sorted(SCOPE_TYPES)
