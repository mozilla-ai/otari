"""The image edit route declares its form fields by hand, so it must keep up with any-llm's."""

from any_llm.types.image import ImageEditParams

from gateway.api.routes import images
from gateway.api.routes._schema_derive import PARAM_FIELD_RENAMES
from gateway.core.provider_params import SENSITIVE_PARAM_FIELDS
from gateway.schemas.inference import ImageEditForm


def _accepted() -> set[str]:
    return {*ImageEditForm.model_fields, *images.EDIT_FILE_FIELDS}


def test_edit_form_accepts_every_image_edit_param() -> None:
    """A field any-llm adds to ``ImageEditParams`` fails here until the route accepts and forwards it."""
    expected = {
        PARAM_FIELD_RENAMES.get(name, name)
        for name in ImageEditParams.model_fields
        if name not in SENSITIVE_PARAM_FIELDS
    }
    assert expected <= _accepted(), f"images/edits does not accept: {sorted(expected - _accepted())}"


def test_edit_form_takes_the_openai_multi_image_spelling() -> None:
    assert {"image", "image[]", "mask"} <= _accepted()


def test_edit_schema_names_every_accepted_field() -> None:
    """The OpenAPI body is written by hand too, so it must list what the route reads."""
    assert set(images._EDIT_UPLOAD_SCHEMA["properties"]) == _accepted()


def test_edit_schema_requires_neither_image_spelling() -> None:
    """Either ``image`` or ``image[]`` is enough, so the schema cannot require one of them."""
    assert {"model", "prompt"} == set(images._EDIT_UPLOAD_SCHEMA["required"])
