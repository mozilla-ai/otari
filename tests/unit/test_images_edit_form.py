"""The image edit route declares its form fields by hand, so it must keep up with any-llm's."""

from any_llm.types.image import ImageEditParams
from fastapi.routing import APIRoute

from gateway.api.routes import images
from gateway.api.routes._schema_derive import PARAM_FIELD_RENAMES
from gateway.core.provider_params import SENSITIVE_PARAM_FIELDS


def _edit_route() -> APIRoute:
    [route] = [r for r in images.router.routes if isinstance(r, APIRoute) and r.path == "/images/edits"]
    return route


def test_edit_form_accepts_every_image_edit_param() -> None:
    """A field any-llm adds to ``ImageEditParams`` fails here until the route accepts and forwards it."""
    expected = {
        PARAM_FIELD_RENAMES.get(name, name)
        for name in ImageEditParams.model_fields
        if name not in SENSITIVE_PARAM_FIELDS
    }
    accepted = {param.alias for param in _edit_route().dependant.body_params}
    assert expected <= accepted, f"images/edits does not accept: {sorted(expected - accepted)}"


def test_edit_form_takes_the_openai_multi_image_spelling() -> None:
    accepted = {param.alias for param in _edit_route().dependant.body_params}
    assert {"image", "image[]", "mask"} <= accepted
