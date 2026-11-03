from io import BytesIO
from typing import cast

import pytest
from fastapi import HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import TypeAdapter
from starlette.datastructures import FormData
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.types import Message, Receive, Scope

from gateway.api.routes._request_body import bounded_request, form_field


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
    limited = bounded_request(request, max_bytes=4, detail="too large")

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
        bounded_request(_request(receive, "5"), max_bytes=4, detail="too large")

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
