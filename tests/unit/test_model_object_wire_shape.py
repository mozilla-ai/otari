"""Which fields the OpenAI-compatible model responses may leave out."""

from collections.abc import Iterator
from typing import Any, Literal, get_args

import pytest
from pydantic import BaseModel

from gateway.api.routes.models import ModelListResponse
from gateway.services.merged_catalog_service import ModelObject

# Fields the two OpenAI-compatible model routes omit when the gateway does not
# know their value, per wire model reachable from either route's response model.
# Both routes serialize with ``response_model_exclude_none``, which recurses, so
# a field whose value is None is dropped wherever it sits, whether or not absence
# is the shape you want for it. A field where null is a known answer rather than
# an unknown one therefore needs a type that does not admit None.
#
# NOTE: A dict-typed value is opaque to the serializer, so a null inside one
# survives. ``pricing_tiers`` has such an arm, which the integration test covers
# by pricing a tiered model.
OMITTABLE_FIELDS = {
    "ModelArchitecture": set(),
    "ModelListResponse": set(),
    "ModelObject": {
        "architecture",
        "context_window",
        "max_output_tokens",
        "name",
        "pricing",
        "supported_parameters",
    },
    "ModelPricingInfo": {
        "cache_read_price_per_million",
        "cache_write_price_per_million",
    },
    "PricingTier": {
        "cache_read_price_per_million",
        "cache_write_price_per_million",
        "input_price_per_million",
        "output_price_per_million",
    },
}


def _admits_none(annotation: Any) -> bool:
    arguments = get_args(annotation)
    return annotation is type(None) or type(None) in arguments or None in arguments


def _models_in(annotation: Any) -> Iterator[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
    for argument in get_args(annotation):
        yield from _models_in(argument)


def _nullable_fields_by_model(*roots: type[BaseModel]) -> dict[str, set[str]]:
    """Each wire model reachable from ``roots``, mapped to the fields whose type admits None."""
    found: dict[str, set[str]] = {}
    pending = list(roots)
    while pending:
        model = pending.pop()
        if model.__name__ in found:
            continue
        found[model.__name__] = {name for name, field in model.model_fields.items() if _admits_none(field.annotation)}
        for field in model.model_fields.values():
            pending.extend(_models_in(field.annotation))
    return found


@pytest.mark.parametrize(
    ("annotation", "admits_none"),
    [
        (int | None, True),
        (type(None), True),
        (Literal["a", None], True),
        (str, False),
        (Literal["a"], False),
        (dict[str, float], False),
    ],
)
def test_a_type_that_admits_none_is_recognized(annotation: Any, admits_none: bool) -> None:
    assert _admits_none(annotation) is admits_none


def test_only_classified_fields_are_omitted_from_a_model_response() -> None:
    assert _nullable_fields_by_model(ModelListResponse, ModelObject) == OMITTABLE_FIELDS, (
        "A field whose type admits None is omitted from the model responses wherever its "
        "value is None. Add it to OMITTABLE_FIELDS if absence is the shape you want, or give "
        "it a type that does not admit None if the response should always send a value."
    )
