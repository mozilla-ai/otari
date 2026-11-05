"""Validate multipart form fields a route parses itself, after its auth check."""

from typing import Literal, TypeVar, overload

from fastapi.exceptions import RequestValidationError
from pydantic import TypeAdapter, ValidationError
from starlette.datastructures import FormData

MULTIPART_OVERHEAD_BYTES = 1024 * 1024
_T = TypeVar("_T")


@overload
def form_field(
    form: FormData,
    name: str,
    adapter: TypeAdapter[_T],
    *,
    required: Literal[True],
) -> _T: ...


@overload
def form_field(
    form: FormData,
    name: str,
    adapter: TypeAdapter[_T],
    *,
    default: _T,
    required: Literal[False] = False,
) -> _T: ...


@overload
def form_field(
    form: FormData,
    name: str,
    adapter: TypeAdapter[_T],
    *,
    required: Literal[False] = False,
) -> _T | None: ...


def form_field(
    form: FormData,
    name: str,
    adapter: TypeAdapter[_T],
    *,
    default: _T | None = None,
    required: bool = False,
) -> _T | None:
    """Validate one parsed form value with the same body location FastAPI uses."""
    if name not in form or form.get(name) == "":
        if required:
            raise RequestValidationError(
                [{"type": "missing", "loc": ("body", name), "msg": "Field required", "input": None}]
            )
        return default

    try:
        return adapter.validate_python(form.get(name))
    except ValidationError as exc:
        errors = [{**error, "loc": ("body", name, *error["loc"])} for error in exc.errors()]
        raise RequestValidationError(errors) from None
