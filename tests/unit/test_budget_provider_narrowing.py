"""Guards on the ``provider_key_id`` field of the three create bodies that narrow to a provider."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from gateway.schemas.budgets import (
    CreateScopedBudgetRequest,
    OrganizationScopedBudgetCreate,
    WorkspaceMemberBudgetPolicyCreate,
)

_CEILING_BODY = {"scope_type": "workspace", "scope_id": "w1", "budget_id": "b1"}
_NARROWING_BODIES: list[tuple[type[BaseModel], dict[str, Any]]] = [
    (CreateScopedBudgetRequest, _CEILING_BODY),
    (OrganizationScopedBudgetCreate, _CEILING_BODY),
    (WorkspaceMemberBudgetPolicyCreate, {"budget_id": "b1"}),
]
_PUBLISHED_STRING_CONSTRAINTS = {"type": "string", "minLength": 1, "maxLength": 255, "pattern": r"^\S+$"}


@pytest.mark.parametrize(("body", "required"), _NARROWING_BODIES)
@pytest.mark.parametrize("refused", ["", "   ", "\t", "two words", "x" * 256])
def test_a_blank_spaced_or_overlong_narrowing_is_refused(
    body: type[BaseModel], required: dict[str, Any], refused: str
) -> None:
    with pytest.raises(ValidationError):
        body.model_validate({**required, "provider_key_id": refused})


@pytest.mark.parametrize(("body", "required"), _NARROWING_BODIES)
@pytest.mark.parametrize("accepted", [None, "x", "openai", "x" * 255])
def test_a_narrowing_is_optional_and_bounded(
    body: type[BaseModel], required: dict[str, Any], accepted: str | None
) -> None:
    assert body.model_validate({**required, "provider_key_id": accepted}).model_dump()["provider_key_id"] == accepted


@pytest.mark.parametrize(("body", "required"), _NARROWING_BODIES)
def test_an_omitted_narrowing_is_none(body: type[BaseModel], required: dict[str, Any]) -> None:
    assert body.model_validate(required).model_dump()["provider_key_id"] is None


@pytest.mark.parametrize("body", [body for body, _ in _NARROWING_BODIES])
def test_the_published_narrowing_constraints_are_the_same_on_every_body(body: type[BaseModel]) -> None:
    published = body.model_json_schema()["properties"]["provider_key_id"]
    assert published["anyOf"] == [_PUBLISHED_STRING_CONSTRAINTS, {"type": "null"}]
    assert published["default"] is None
