from io import BytesIO
from typing import cast

import pytest
from fastapi import HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import TypeAdapter
from starlette.datastructures import FormData
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.types import Message, Receive, Scope

from gateway.api.routes._helpers import bounded_body
from gateway.api.routes._request_body import form_field


def _request(receive: Receive, content_length: str | None = None) -> Request:
    headers = [] if content_length is None else [(b"content-length", content_length.encode())]
    return Request(cast(Scope, {"type": "http", "headers": headers}), receive)


@pytest.mark.asyncio
@pytest.mark.parametrize("content_length", [None, "1"])
async def test_received_bytes_are_limited_without_trusting_content_length(content_length: str | None) -> None:
    messages: list[Message] = [
        {"type": "http.request", "body": b"123", "more_body": True},
        {"type": "http.request", "body": b"45", "more_body": False},
    ]
    receive_count = 0

    async def receive() -> Message:
        nonlocal receive_count
        message = messages[receive_count]
        receive_count += 1
        return message

    request = _request(receive, content_length)
    limited = bounded_body(request, 4, "too large")

    with pytest.raises(HTTPException) as raised:
        await limited.body()

    assert raised.value.status_code == 413
    assert receive_count == 2


def test_declared_oversize_is_rejected_without_receiving_the_body() -> None:
    receive_count = 0

    async def receive() -> Message:
        nonlocal receive_count
        receive_count += 1
        return {"type": "http.request", "body": b"12345", "more_body": False}

    with pytest.raises(HTTPException) as raised:
        bounded_body(_request(receive, "5"), 4, "too large")

    assert raised.value.status_code == 413
    assert receive_count == 0


def test_form_field_preserves_fastapi_upload_file_values() -> None:
    upload = StarletteUploadFile(filename="notes.txt", file=BytesIO(b"notes"))
    form = FormData([("file", upload)])

    assert form_field(form, "file", TypeAdapter(UploadFile), required=True) is upload


def test_missing_form_field_reports_a_body_validation_error() -> None:
    with pytest.raises(RequestValidationError) as raised:
        form_field(FormData(), "file", TypeAdapter(UploadFile), required=True)

    assert raised.value.errors()[0]["loc"] == ("body", "file")


@pytest.mark.parametrize("form", [FormData(), FormData([("model", "")])])
def test_missing_or_empty_required_scalar_reports_a_body_validation_error(form: FormData) -> None:
    with pytest.raises(RequestValidationError) as raised:
        form_field(form, "model", TypeAdapter(str), required=True)

    assert raised.value.errors()[0]["type"] == "missing"
    assert raised.value.errors()[0]["loc"] == ("body", "model")


@pytest.mark.parametrize("form", [FormData(), FormData([("temperature", "")])])
def test_missing_or_empty_optional_scalar_uses_default(form: FormData) -> None:
    assert form_field(form, "temperature", TypeAdapter(float), default=0.5) == 0.5


def test_nonempty_scalar_is_validated() -> None:
    assert form_field(FormData([("temperature", "0")]), "temperature", TypeAdapter(float)) == 0.0
    with pytest.raises(RequestValidationError) as raised:
        form_field(FormData([("temperature", "invalid")]), "temperature", TypeAdapter(float))
    assert raised.value.errors()[0]["type"] == "float_parsing"
    assert raised.value.errors()[0]["loc"] == ("body", "temperature")
