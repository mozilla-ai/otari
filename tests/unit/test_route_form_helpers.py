"""The helpers a route reads its own multipart form with, after its auth check."""

import io

import pytest
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.datastructures import FormData, UploadFile

from gateway.api.routes._helpers import form_fields, form_file, multipart_request_body
from gateway.schemas.files import FileUploadForm
from gateway.schemas.inference import TranscriptionForm


class _Form(BaseModel):
    model: str
    n: int | None = None


def _upload() -> UploadFile:
    return UploadFile(io.BytesIO(b"bytes"), filename="a.bin", size=5)


def test_form_fields_reads_the_text_fields_a_model_names() -> None:
    form = FormData([("model", "m"), ("n", "2"), ("extra", "ignored")])
    assert form_fields(form, _Form) == _Form(model="m", n=2)


@pytest.mark.parametrize(
    ("form", "field", "error_type"),
    [
        pytest.param(FormData([("n", "2")]), "model", "missing", id="missing"),
        pytest.param(FormData([("model", "m"), ("n", "two")]), "n", "int_parsing", id="wrong-type"),
        pytest.param(FormData([("model", _upload())]), "model", "value_error", id="sent-as-file"),
    ],
)
def test_form_fields_refuses_as_fastapi_does(form: FormData, field: str, error_type: str) -> None:
    with pytest.raises(RequestValidationError) as refused:
        form_fields(form, _Form)
    [error] = refused.value.errors()
    assert error["loc"] == ("body", field)
    assert error["type"] == error_type


def test_form_file_returns_the_upload() -> None:
    upload = _upload()
    assert form_file(FormData([("file", upload)]), "file") is upload


@pytest.mark.parametrize(
    ("form", "error_type"),
    [
        pytest.param(FormData([]), "missing", id="missing"),
        pytest.param(FormData([("file", "/etc/passwd")]), "value_error", id="sent-as-text"),
    ],
)
def test_form_file_refuses_a_missing_or_text_file(form: FormData, error_type: str) -> None:
    with pytest.raises(RequestValidationError) as refused:
        form_file(form, "file")
    [error] = refused.value.errors()
    assert error["loc"] == ("body", "file")
    assert error["type"] == error_type


@pytest.mark.parametrize(
    ("model", "required"),
    [
        pytest.param(TranscriptionForm, ["model", "file"], id="transcription"),
        pytest.param(FileUploadForm, ["file"], id="file-upload"),
        pytest.param(None, ["file"], id="file-only"),
    ],
)
def test_multipart_request_body_names_every_field_the_route_reads(
    model: type[BaseModel] | None, required: list[str]
) -> None:
    body = multipart_request_body(model, {"file": "The file."}, required_files=("file",))
    schema = body["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert set(schema["properties"]) == {*(model.model_fields if model is not None else ()), "file"}
    assert schema["required"] == required
    assert schema["properties"]["file"] == {
        "type": "string",
        "contentMediaType": "application/octet-stream",
        "description": "The file.",
    }
