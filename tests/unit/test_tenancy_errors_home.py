"""The status-carrying error bases have one definition, in ``gateway.exceptions``."""

import pytest

from gateway import exceptions
from gateway.services.tenancy import errors

_BASES = (
    "TenancyError",
    "TenancyNotFoundError",
    "TenancyForbiddenError",
    "TenancyConflictError",
    "TenancyValidationError",
)


@pytest.mark.parametrize("name", _BASES)
def test_base_has_one_definition(name: str) -> None:
    assert getattr(errors, name) is getattr(exceptions, name)
