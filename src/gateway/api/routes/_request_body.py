"""Bound and validate request bodies read by gateway routes after authorization."""

from typing import Literal, TypeVar, overload

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import TypeAdapter, ValidationError
from starlette.datastructures import FormData
from starlette.types import Message

MULTIPART_OVERHEAD_BYTES = 1024 * 1024
_T = TypeVar("_T")


def bounded_request(request: Request, *, max_bytes: int, detail: str) -> Request:
    """Return a request that refuses oversized bodies, trusting length only for early rejection."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            pass
        else:
            if declared_length > max_bytes:
                raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail)

    received = 0
    original_receive = request.receive

    async def receive() -> Message:
        nonlocal received
        message = await original_receive()
        body = message.get("body", b"")
        if len(body) > max_bytes - received:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail)
        received += len(body)
        return message

    return Request(request.scope, receive)


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
    if name not in form:
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
