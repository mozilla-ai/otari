"""Native envelopes stay at the API boundary, including their discriminators."""

from datetime import UTC, datetime

import pytest
from starlette.requests import Request

from gateway.api.routes._file_formats import AnthropicFilesFormat, OpenAIFilesFormat, files_format
from gateway.services.provider_files.contracts import FileMetadata, FilePage, FilesError


def test_metadata_discriminators_survive_exclude_unset() -> None:
    data = FileMetadata(
        id="file_test",
        filename="input.csv",
        size_bytes=4,
        mime_type="text/csv",
        created_at=datetime.now(UTC),
        downloadable=True,
        purpose="user_data",
        status="processed",
    )
    native = AnthropicFilesFormat().metadata(data).model_dump(exclude_unset=True)
    assert native["type"] == "file"
    assert "purpose" not in native and "status" not in native
    native = OpenAIFilesFormat().metadata(data).model_dump(exclude_unset=True)
    assert native["object"] == "file"
    assert native["purpose"] == "user_data" and native["status"] == "processed"


def test_empty_openai_page_is_native() -> None:
    page = OpenAIFilesFormat().page(FilePage(data=[])).model_dump(exclude_unset=True)
    assert page == {"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False}


@pytest.mark.parametrize("provider", ["gemini", "unknown_provider", "", "openai,anthropic"])
def test_unavailable_provider_is_rejected(provider: str) -> None:
    request = Request({"type": "http", "headers": [(b"x-otari-files-provider", provider.encode())]})
    with pytest.raises(FilesError) as error:
        files_format(request)
    assert error.value.status_code == 400


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"purpose": " "},
        {"purpose": "user_data", "expires_after[anchor]": "created_at"},
        {"purpose": "user_data", "expires_after[anchor]": "last_active_at", "expires_after[seconds]": "3600"},
        {"purpose": "user_data", "expires_after[anchor]": "created_at", "expires_after[seconds]": "7776000"},
    ],
)
def test_invalid_openai_upload_options_fail_locally(fields: dict[str, str]) -> None:
    with pytest.raises(FilesError):
        OpenAIFilesFormat().upload_options(fields)
