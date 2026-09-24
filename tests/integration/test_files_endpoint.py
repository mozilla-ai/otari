"""Integration tests for the /api/v1/files API and end-to-end file understanding.

The headline test exercises the full path: upload a file, send a chat request
that references it by ``file_id``, and assert the gateway extracted the file to
text and that text reached the provider — proving a text-only local model can
"read" an uploaded file.

The markitdown extraction boundary is monkeypatched for determinism so the test
needs no optional extraction deps; everything else (HTTP, DB, blob store,
file_id resolution, capability resolution, message rewriting) is real.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
from anthropic import Anthropic, BadRequestError
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.adapters.file_storage_adapter import LocalDirFileStore
from gateway.core.config import API_ROOT, API_VERSION, GatewayConfig
from gateway.models.files import FileObject
from gateway.services.file_extractors import ExtractionResult

from .conftest import build_test_client


@pytest.fixture
def tmp_file_store(client: TestClient, tmp_path: Path) -> None:
    """Point the app's blob store at a temp dir (default writes to cwd)."""
    cast(Any, client.app).state.file_store = LocalDirFileStore(str(tmp_path))


@pytest.fixture
def files_off_client(test_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    """A client on a deployment that does not serve files."""
    yield from build_test_client(test_config.model_copy(update={"files_enabled": False}))


def _make_completion() -> Any:
    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionMessage,
        Choice,
        CompletionUsage,
    )

    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model="llama3",
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="ok"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )


def test_upload_requires_auth(client: TestClient) -> None:
    resp = client.post(f"{API_ROOT}/files", files={"file": ("a.txt", b"x", "text/plain")})
    assert resp.status_code == 401


def test_upload_empty_file_rejected(client: TestClient, api_key_header: dict[str, str], tmp_file_store: None) -> None:
    resp = client.post(f"{API_ROOT}/files", headers=api_key_header, files={"file": ("empty.txt", b"", "text/plain")})
    assert resp.status_code == 400


def test_upload_too_large_rejected(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    test_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(test_config, "files_max_bytes", 16)
    resp = client.post(
        f"{API_ROOT}/files",
        headers=api_key_header,
        files={"file": ("big.bin", b"x" * 64, "application/octet-stream")},
    )
    assert resp.status_code == 413


def test_files_are_user_scoped(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    tmp_file_store: None,
) -> None:
    # Upload as the first key's user (the shared "default" user: this key was
    # created without an explicit user_id).
    up = client.post(f"{API_ROOT}/files", headers=api_key_header, files={"file": ("a.txt", b"hi", "text/plain")})
    file_id = up.json()["id"]

    # A key owned by a DIFFERENT user must not see it. File isolation is per-user,
    # not per-key, and keys created without a user_id all share the "default" user
    # (see keys.py / get_or_create_default_user), so give this key its own owner.
    other = client.post(
        f"{API_ROOT}/keys", json={"key_name": "other", "user_id": "other-user"}, headers=master_key_header
    )
    # Match the fixture's auth scheme: the gateway requires a "Bearer " prefix on
    # every header form, including Otari-Key (see api_key_header / deps.py
    # extract_credential_token), so reuse the fixture's header name with a Bearer value.
    other_header = {next(iter(api_key_header)): f"Bearer {other.json()['key']}"}

    assert client.get(f"{API_ROOT}/files/{file_id}", headers=other_header).status_code == 404
    assert client.get(f"{API_ROOT}/files/{file_id}/content", headers=other_header).status_code == 404
    # Owner still sees it.
    assert client.get(f"{API_ROOT}/files/{file_id}", headers=api_key_header).status_code == 200


def test_no_owner_keys_share_default_user_files(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    tmp_file_store: None,
) -> None:
    # Two keys created without an explicit user_id both attach to the shared
    # "default" user, so they intentionally share files (and budget/usage). This
    # pins that deliberate behavior: no-owner keys are NOT isolated from each other;
    # isolation requires giving keys distinct owners (see test_files_are_user_scoped).
    up = client.post(f"{API_ROOT}/files", headers=api_key_header, files={"file": ("a.txt", b"hi", "text/plain")})
    file_id = up.json()["id"]

    sibling = client.post(f"{API_ROOT}/keys", json={"key_name": "sibling"}, headers=master_key_header)
    sibling_header = {next(iter(api_key_header)): f"Bearer {sibling.json()['key']}"}

    assert client.get(f"{API_ROOT}/files/{file_id}", headers=sibling_header).status_code == 200
    assert client.get(f"{API_ROOT}/files/{file_id}/content", headers=sibling_header).status_code == 200


def test_download_filename_header_is_injection_safe(
    client: TestClient, api_key_header: dict[str, str], tmp_file_store: None
) -> None:
    up = client.post(
        f"{API_ROOT}/files",
        headers=api_key_header,
        files={"file": ('ev"il\r\nX-Injected: 1.txt', b"data", "text/plain")},
    )
    assert up.status_code == 200
    file_id = up.json()["id"]

    resp = client.get(f"{API_ROOT}/files/{file_id}/content", headers=api_key_header)
    assert resp.status_code == 200
    cd = resp.headers["content-disposition"]
    # No raw CR/LF or unescaped quotes leaked into the header.
    assert "\r" not in cd and "\n" not in cd
    assert "X-Injected" not in resp.headers
    assert "filename*=UTF-8''" in cd


@pytest.mark.parametrize(
    ("filename", "payload", "media_type"),
    [("notes.txt", b"hello world", "text/plain"), ("data.bin", b"\x00\xff\x80", "application/octet-stream")],
)
def test_upload_get_list_delete_roundtrip(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    filename: str,
    payload: bytes,
    media_type: str,
) -> None:
    up = client.post(
        f"{API_ROOT}/files",
        headers=api_key_header,
        data={"purpose": "user_data"},
        files={"file": (filename, payload, media_type)},
    )
    assert up.status_code == 200, up.text
    obj = up.json()
    file_id = obj["id"]
    assert obj["object"] == "file"
    assert obj["bytes"] == len(payload)
    assert obj["filename"] == filename

    got = client.get(f"{API_ROOT}/files/{file_id}", headers=api_key_header)
    assert got.status_code == 200
    assert got.json()["id"] == file_id

    content = client.get(f"{API_ROOT}/files/{file_id}/content", headers=api_key_header)
    assert content.status_code == 200
    assert content.content == payload
    assert content.headers["content-type"].split(";")[0] == media_type
    assert content.headers["content-disposition"].startswith("attachment;")
    assert f'filename="{filename}"' in content.headers["content-disposition"]
    assert f"filename*=UTF-8''{filename}" in content.headers["content-disposition"]

    listed = client.get(f"{API_ROOT}/files", headers=api_key_header)
    assert listed.status_code == 200
    assert any(f["id"] == file_id for f in listed.json()["data"])

    deleted = client.delete(f"{API_ROOT}/files/{file_id}", headers=api_key_header)
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    # Gone after delete.
    assert client.get(f"{API_ROOT}/files/{file_id}", headers=api_key_header).status_code == 404


def test_file_id_extracted_reaches_provider(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upload → chat with file_id on a text-only model → extracted text is sent."""

    async def fake_extract(data: bytes, mime: str, filename: str | None) -> ExtractionResult:
        return ExtractionResult(f"EXTRACTED::{data.decode()}", True, "ok")

    monkeypatch.setattr("gateway.services.content_normalizer.extract_text_from_file", fake_extract)

    up = client.post(
        f"{API_ROOT}/files",
        headers=api_key_header,
        files={"file": ("report.txt", b"quarterly numbers", "text/plain")},
    )
    assert up.status_code == 200, up.text
    file_id = up.json()["id"]

    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_completion()

    # ollama is a local provider → capabilities default to "extract" (text-only),
    # so the file block must be turned into text before the provider call.
    body = {
        "model": "ollama:llama3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize the attached file."},
                    {"type": "file", "file": {"file_id": file_id}},
                ],
            }
        ],
    }

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = client.post(f"{API_ROOT}/chat/completions", headers=api_key_header, json=body)

    assert resp.status_code == 200, resp.text

    sent = json.dumps(captured["messages"])
    # The extracted text — framed with the filename header — reached the provider...
    assert "EXTRACTED::quarterly numbers" in sent
    assert "report.txt" in sent
    # ...and the raw file block did not (it was replaced by a text block).
    assert "file_id" not in sent


def test_budget_rejection_skips_normalization(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request rejected at the budget gate must not run extraction/vision.

    Locks in the ordering fix: normalization happens only after reserve_budget,
    so a blocked/over-budget user can't trigger extraction or vision side-calls.
    """
    from fastapi import HTTPException, status

    extracted = {"called": False}

    async def spy_extract(*args: Any, **kwargs: Any) -> ExtractionResult:
        extracted["called"] = True
        return ExtractionResult("should-not-run", True, "ok")

    async def deny_budget(*args: Any, **kwargs: Any) -> Any:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="over budget")

    monkeypatch.setattr("gateway.services.content_normalizer.extract_text_from_file", spy_extract)
    monkeypatch.setattr("gateway.api.routes._pipeline.reserve_budget", deny_budget)

    pdf_url = "data:application/pdf;base64," + base64.b64encode(b"%PDF-1.4 fake").decode("ascii")
    body = {
        "model": "ollama:llama3",  # text-only → would extract if normalization ran
        "messages": [
            {"role": "user", "content": [{"type": "file", "file": {"file_data": pdf_url, "filename": "x.pdf"}}]}
        ],
    }

    resp = client.post(f"{API_ROOT}/chat/completions", headers=api_key_header, json=body)
    assert resp.status_code == 403
    assert extracted["called"] is False


def test_native_model_passes_file_through(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A natively-capable model gets the file forwarded (file_id inlined), not extracted."""

    async def fail_extract(*args: Any, **kwargs: Any) -> ExtractionResult:  # pragma: no cover
        raise AssertionError("extraction must not run for a native model")

    monkeypatch.setattr("gateway.services.content_normalizer.extract_text_from_file", fail_extract)

    up = client.post(
        f"{API_ROOT}/files",
        headers=api_key_header,
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    file_id = up.json()["id"]

    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_completion()

    body = {
        "model": "openai:gpt-4o",  # hosted, natively multimodal → passthrough
        "messages": [{"role": "user", "content": [{"type": "file", "file": {"file_id": file_id}}]}],
    }

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = client.post(f"{API_ROOT}/chat/completions", headers=api_key_header, json=body)

    assert resp.status_code == 200, resp.text
    sent = json.dumps(captured["messages"])
    # file_id was resolved to an inline data: URL the provider understands.
    assert "data:application/pdf;base64," in sent


def test_expired_file_returns_404(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    db_session: Session,
) -> None:
    """A file past its retention window is no longer served (404)."""
    up = client.post(
        f"{API_ROOT}/files",
        headers=api_key_header,
        files={"file": ("a.txt", b"hi", "text/plain")},
    )
    file_id = up.json()["id"]

    # Backdate the stored expiry into the past, simulating retention elapsing.
    record = db_session.get(FileObject, file_id)
    assert record is not None
    record.expires_at = datetime.now(UTC) - timedelta(hours=1)
    db_session.commit()

    assert client.get(f"{API_ROOT}/files/{file_id}", headers=api_key_header).status_code == 404
    assert client.get(f"{API_ROOT}/files/{file_id}/content", headers=api_key_header).status_code == 404


def test_an_expired_file_is_not_listed(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    db_session: Session,
) -> None:
    """The listing hides what every other verb 404s, rather than waiting for the sweep."""
    live = client.post(
        f"{API_ROOT}/files", headers=api_key_header, files={"file": ("live.txt", b"hi", "text/plain")}
    ).json()["id"]
    gone = client.post(
        f"{API_ROOT}/files", headers=api_key_header, files={"file": ("gone.txt", b"hi", "text/plain")}
    ).json()["id"]

    record = db_session.get(FileObject, gone)
    assert record is not None
    record.expires_at = datetime.now(UTC) - timedelta(hours=1)
    db_session.commit()

    listed = client.get(f"{API_ROOT}/files", headers=api_key_header)
    assert listed.status_code == 200, listed.text
    ids = {row["id"] for row in listed.json()["data"]}
    assert live in ids
    assert gone not in ids


def test_vision_describe_side_call_is_billed(
    client: TestClient,
    master_key_header: dict[str, str],
    tmp_file_store: None,
    test_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A describe side-call for a text-only model is metered and billed.

    Captioning an image for a text-only target model issues a side-call to the
    configured vision model. Its cost must land on the user's spend even though
    the target model itself has no pricing.
    """
    from any_llm.types.completion import CompletionUsage

    monkeypatch.setattr(test_config, "vision_strategy", "describe")
    monkeypatch.setattr(test_config, "vision_describe_model", "openai:gpt-4o-mini")

    # Pricing for the vision model → the side-call has a non-zero cost.
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openai:gpt-4o-mini",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )
    client.post(f"{API_ROOT}/users", json={"user_id": "vision-user"}, headers=master_key_header)

    async def fake_describe(config: Any, data_url: str) -> tuple[str | None, CompletionUsage | None]:
        return "a chart of revenue", CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    monkeypatch.setattr("gateway.services.content_normalizer.describe_image", fake_describe)

    captured: dict[str, Any] = {}

    async def mock_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_completion()

    image = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
    body = {
        "model": "ollama:llama3",  # text-only local → image is described, not forwarded
        "user": "vision-user",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]}],
    }

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = client.post(f"{API_ROOT}/chat/completions", headers=master_key_header, json=body)

    assert resp.status_code == 200, resp.text
    # The caption reached the provider in place of the image block...
    assert "a chart of revenue" in json.dumps(captured["messages"])
    # ...and the describe side-call's cost (1000/1e6*2.5 + 500/1e6*10) was billed.
    user = client.get(f"{API_ROOT}/users/vision-user", headers=master_key_header).json()
    assert user["spend"] == pytest.approx(0.0075)


def test_files_user_mismatch_rejected_by_default(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
) -> None:
    """Strict default: a non-master key naming a different 'user' is rejected."""
    resp = client.post(
        f"{API_ROOT}/files",
        headers=api_key_header,
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"user": "someone-else"},
    )
    assert resp.status_code == 403
    assert "does not match" in resp.json()["detail"]


def test_files_user_mismatch_ignored_when_lenient(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    tmp_file_store: None,
    test_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With reject_user_mismatch=False, a mismatched 'user' is ignored.

    The upload succeeds and ownership still binds to the key's own user, like
    every other pass-through route.
    """
    monkeypatch.setattr(test_config, "reject_user_mismatch", False)
    resp = client.post(
        f"{API_ROOT}/files",
        headers=api_key_header,
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"user": "someone-else"},
    )
    assert resp.status_code == 200, resp.text
    file_id = resp.json()["id"]

    # The file is scoped to the key's user, not the mismatched name: listing
    # with the key (which resolves to its own user) returns it.
    listing = client.get(f"{API_ROOT}/files", headers=api_key_header)
    assert listing.status_code == 200
    assert any(f["id"] == file_id for f in listing.json()["data"])


_ANTHROPIC = {"anthropic-version": "2023-06-01"}


def test_anthropic_sdk_headers_get_anthropic_shapes(
    client: TestClient, api_key_header: dict[str, str], tmp_file_store: None
) -> None:
    """The Anthropic SDK sends ``anthropic-version`` on every call and reads ``FileMetadata``."""
    headers = {**api_key_header, **_ANTHROPIC}
    up = client.post(f"{API_ROOT}/files", headers=headers, files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})
    assert up.status_code == 200, up.text
    meta = up.json()
    assert meta["type"] == "file"
    assert meta["size_bytes"] == len(b"%PDF-1.4")
    assert meta["mime_type"] == "application/pdf"
    assert meta["downloadable"] is True
    assert meta["created_at"].endswith("Z")
    assert "expires_at" in meta
    assert meta["expires_at"] is None
    assert "object" not in meta and "bytes" not in meta

    got = client.get(f"{API_ROOT}/files/{meta['id']}", headers=headers)
    assert got.status_code == 200
    assert got.json()["size_bytes"] == len(b"%PDF-1.4")

    listed = client.get(f"{API_ROOT}/files", headers=headers)
    assert listed.status_code == 200
    page = listed.json()
    assert set(page) == {"data", "next_page"}
    assert page["next_page"] is None
    assert [f["id"] for f in page["data"]] == [meta["id"]]

    # The same file, read with OpenAI's headers, is the OpenAI object.
    assert client.get(f"{API_ROOT}/files/{meta['id']}", headers=api_key_header).json()["object"] == "file"

    deleted = client.delete(f"{API_ROOT}/files/{meta['id']}", headers=headers)
    assert deleted.status_code == 200
    assert deleted.json() == {"id": meta["id"], "type": "file_deleted"}


def test_list_is_cursor_paged(client: TestClient, api_key_header: dict[str, str], tmp_file_store: None) -> None:
    ids = [
        client.post(
            f"{API_ROOT}/files", headers=api_key_header, files={"file": (f"{n}.txt", b"x", "text/plain")}
        ).json()["id"]
        for n in range(3)
    ]

    first = client.get(f"{API_ROOT}/files", headers=api_key_header, params={"limit": 2}).json()
    assert first["object"] == "list"
    assert len(first["data"]) == 2
    assert first["has_more"] is True
    assert first["first_id"] == first["data"][0]["id"]
    assert first["last_id"] == first["data"][1]["id"]

    second = client.get(
        f"{API_ROOT}/files", headers=api_key_header, params={"limit": 2, "after": first["last_id"]}
    ).json()
    assert len(second["data"]) == 1
    assert second["has_more"] is False
    seen = [f["id"] for f in first["data"] + second["data"]]
    assert sorted(seen) == sorted(ids)
    assert len(set(seen)) == 3

    asc = client.get(f"{API_ROOT}/files", headers=api_key_header, params={"limit": 3, "order": "asc"}).json()
    assert [f["id"] for f in asc["data"]] == list(reversed(seen))
    tail = client.get(
        f"{API_ROOT}/files", headers=api_key_header, params={"after": asc["data"][0]["id"], "order": "asc"}
    ).json()
    assert [f["id"] for f in tail["data"]] == [f["id"] for f in asc["data"][1:]]

    # A cursor that has since been deleted is still a position: the usual
    # "list a page, delete each, list again from last_id" loop must not 404
    # on its second page.
    assert client.delete(f"{API_ROOT}/files/{first['last_id']}", headers=api_key_header).status_code == 200
    after_deleted = client.get(
        f"{API_ROOT}/files", headers=api_key_header, params={"limit": 2, "after": first["last_id"]}
    )
    assert after_deleted.status_code == 200
    assert [f["id"] for f in after_deleted.json()["data"]] == [second["data"][0]["id"]]

    # A cursor the caller never owned answers like a direct read of it would.
    assert client.get(f"{API_ROOT}/files", headers=api_key_header, params={"after": "file-nope"}).status_code == 404
    assert client.get(f"{API_ROOT}/files", headers=api_key_header, params={"limit": 0}).status_code == 422


def _upload_text_files(client: TestClient, headers: dict[str, str], count: int) -> list[str]:
    return [
        client.post(f"{API_ROOT}/files", headers=headers, files={"file": (f"{n}.txt", b"x", "text/plain")}).json()["id"]
        for n in range(count)
    ]


def test_anthropic_listing_pages_with_next_page(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    tmp_file_store: None,
) -> None:
    headers = {**api_key_header, **_ANTHROPIC}
    ids = _upload_text_files(client, headers, 3)

    first = client.get(f"{API_ROOT}/files", headers=headers, params={"limit": 2}).json()
    assert set(first) == {"data", "next_page"}
    assert len(first["data"]) == 2
    assert first["next_page"].startswith("page_")

    second = client.get(f"{API_ROOT}/files", headers=headers, params={"limit": 2, "page": first["next_page"]}).json()
    assert second["next_page"] is None
    assert sorted(f["id"] for f in first["data"] + second["data"]) == sorted(ids)

    # A token is opaque, so a file ID or a token this gateway never issued is a bad request.
    # page_AA decodes to a NUL character, which must not reach the database.
    for page in ("page_nope", "page_é", "page_AA", ids[0]):
        refused = client.get(f"{API_ROOT}/files", headers=headers, params={"page": page})
        assert refused.status_code == 400, refused.text

    # A token issued to another user names a file this caller cannot see.
    other = client.post(
        f"{API_ROOT}/keys", json={"key_name": "other", "user_id": "other-user"}, headers=master_key_header
    )
    other_headers = {next(iter(api_key_header)): f"Bearer {other.json()['key']}", **_ANTHROPIC}
    _upload_text_files(client, other_headers, 2)
    foreign = client.get(f"{API_ROOT}/files", headers=other_headers, params={"limit": 1}).json()["next_page"]
    assert client.get(f"{API_ROOT}/files", headers=headers, params={"page": foreign}).status_code == 400


def test_anthropic_listing_reads_named_ids_in_one_page(
    client: TestClient, api_key_header: dict[str, str], tmp_file_store: None
) -> None:
    headers = {**api_key_header, **_ANTHROPIC}
    first, _, third = _upload_text_files(client, headers, 3)

    named = client.get(
        f"{API_ROOT}/files", headers=headers, params={"ids[]": [first, third, "file-missing", first]}
    ).json()
    assert named["next_page"] is None
    assert sorted(f["id"] for f in named["data"]) == sorted([first, third])

    # The cap counts distinct IDs.
    repeated = client.get(f"{API_ROOT}/files", headers=headers, params={"ids[]": [first] * 150})
    assert repeated.status_code == 200, repeated.text
    assert [f["id"] for f in repeated.json()["data"]] == [first]

    # An ID with a NUL names no file, so it is left out like any other miss.
    with_nul = client.get(f"{API_ROOT}/files", headers=headers, params={"ids[]": [first, "file-\x00"]})
    assert with_nul.status_code == 200, with_nul.text
    assert [f["id"] for f in with_nul.json()["data"]] == [first]

    for extra in ({"limit": 1}, {"page": "page_x"}):
        mixed = client.get(f"{API_ROOT}/files", headers=headers, params={"ids[]": [first], **extra})
        assert mixed.status_code == 400, mixed.text

    too_many = client.get(f"{API_ROOT}/files", headers=headers, params={"ids[]": [f"file-{n}" for n in range(101)]})
    assert too_many.status_code == 400, too_many.text


@pytest.mark.parametrize("cursor", ["after_id", "before_id"])
def test_anthropic_listing_refuses_the_beta_cursors(
    client: TestClient, api_key_header: dict[str, str], tmp_file_store: None, cursor: str
) -> None:
    (file_id,) = _upload_text_files(client, api_key_header, 1)
    refused = client.get(f"{API_ROOT}/files", headers={**api_key_header, **_ANTHROPIC}, params={cursor: file_id})
    assert refused.status_code == 400
    assert "page" in refused.json()["detail"]


_FILES_BETA = "files-api-2025-04-14"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/files"),
        ("GET", "/files"),
        ("GET", "/files/file-any"),
        ("GET", "/files/file-any/content"),
        ("DELETE", "/files/file-any"),
    ],
)
@pytest.mark.parametrize("anthropic_version", [True, False])
def test_the_files_beta_header_is_refused(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    method: str,
    path: str,
    anthropic_version: bool,
) -> None:
    headers = {**api_key_header, "anthropic-beta": f"code-execution-2025-08-25, {_FILES_BETA}"}
    if anthropic_version:
        headers.update(_ANTHROPIC)
    upload = {"file": ("a.txt", b"x", "text/plain")} if method == "POST" else None

    refused = client.request(method, f"{API_ROOT}{path}", headers=headers, files=upload)

    assert refused.status_code == 400
    assert _FILES_BETA in refused.json()["detail"]


def test_other_anthropic_betas_are_served(
    client: TestClient, api_key_header: dict[str, str], tmp_file_store: None
) -> None:
    headers = {**api_key_header, **_ANTHROPIC, "anthropic-beta": "code-execution-2025-08-25"}
    listed = client.get(f"{API_ROOT}/files", headers=headers)
    assert listed.status_code == 200, listed.text
    assert set(listed.json()) == {"data", "next_page"}


@pytest.fixture
def anthropic_sdk(client: TestClient, api_key_obj: dict[str, Any]) -> Generator[Anthropic]:
    """Anthropic's SDK, sending its requests through the test app.

    Gotcha: the SDK accepts only an ``httpx.Client``, and ``TestClient`` is built on ``httpx2``.
    """

    def forward(request: httpx.Request) -> httpx.Response:
        sent = client.request(
            request.method, str(request.url), headers=request.headers.multi_items(), content=request.read()
        )
        return httpx.Response(sent.status_code, headers=sent.headers.multi_items(), content=sent.content)

    with Anthropic(
        base_url=f"{client.base_url}{API_ROOT.removesuffix(f'/{API_VERSION}')}",
        api_key=api_key_obj["key"],
        http_client=httpx.Client(transport=httpx.MockTransport(forward)),
        max_retries=0,
    ) as sdk:
        yield sdk


def test_anthropic_sdk_files_client_pages_and_reads_expires_at(
    anthropic_sdk: Anthropic, tmp_file_store: None, db_session: Session
) -> None:
    uploaded = [anthropic_sdk.files.upload(file=(f"{n}.txt", b"x", "text/plain")).id for n in range(3)]

    first = anthropic_sdk.files.list(limit=2)
    assert len(first.data) == 2
    assert first.has_next_page()
    assert sorted(f.id for f in anthropic_sdk.files.list(limit=2)) == sorted(uploaded)

    expires_at = datetime.now(UTC).replace(microsecond=0) + timedelta(days=1)
    record = db_session.get(FileObject, uploaded[0])
    assert record is not None
    record.expires_at = expires_at
    db_session.commit()
    assert anthropic_sdk.files.retrieve_metadata(uploaded[0]).expires_at == expires_at
    assert anthropic_sdk.files.retrieve_metadata(uploaded[1]).expires_at is None

    assert anthropic_sdk.files.download(uploaded[1]).read() == b"x"
    assert anthropic_sdk.files.delete(uploaded[1]).type == "file_deleted"

    with pytest.raises(BadRequestError, match=_FILES_BETA):
        anthropic_sdk.beta.files.list(betas=[_FILES_BETA])


def test_sweep_reclaims_expired_and_deleted_files(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    tmp_path: Path,
    db_session: Session,
    test_config: Any,
) -> None:
    """Expiry hides a file; the sweep takes its bytes and row, and a deleted file's row with them."""
    import asyncio

    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from gateway.core.unit_of_work import UnitOfWork
    from gateway.repositories.files import FileRepository
    from gateway.services.files import sweep_files

    def _upload(name: str) -> str:
        resp = client.post(
            f"{API_ROOT}/files", headers=api_key_header, files={"file": (name, b"payload", "text/plain")}
        )
        assert resp.status_code == 200, resp.text
        return str(resp.json()["id"])

    expired, deleted, live = _upload("expired.txt"), _upload("deleted.txt"), _upload("live.txt")
    # Every row here is an upload, so its blob ref is set; ``storage_ref`` is
    # nullable only for a file a provider holds.
    refs = {
        row.id: str(row.storage_ref)
        for row in db_session.query(FileObject).filter(FileObject.id.in_([expired, deleted, live])).all()
    }
    db_session.query(FileObject).filter(FileObject.id == expired).update(
        {"expires_at": datetime.now(UTC) - timedelta(hours=1)}
    )
    db_session.commit()
    assert client.delete(f"{API_ROOT}/files/{deleted}", headers=api_key_header).status_code == 200
    assert (tmp_path / refs[expired]).exists()

    store = LocalDirFileStore(str(tmp_path))

    async def _sweep() -> int:
        engine = create_async_engine(make_url(test_config.database_url).set(drivername="postgresql+asyncpg"))
        try:
            async with async_sessionmaker(engine)() as db, UnitOfWork(db) as uow:
                batch = await sweep_files(FileRepository(uow), store, batch_size=10)
                return batch.reclaimed
        finally:
            await engine.dispose()

    assert asyncio.run(_sweep()) == 2
    db_session.expire_all()
    remaining = {row.id for row in db_session.query(FileObject).all()}
    assert expired not in remaining and deleted not in remaining and live in remaining
    assert not (tmp_path / refs[expired]).exists()
    assert (tmp_path / refs[live]).exists()
    assert client.get(f"{API_ROOT}/files/{live}", headers=api_key_header).status_code == 200


def test_sweep_pages_past_rows_whose_blob_will_not_delete(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    tmp_path: Path,
    db_session: Session,
    test_config: Any,
) -> None:
    """A row whose blob keeps failing must not park at the head and hide the rows behind it."""
    import asyncio

    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from gateway.core.unit_of_work import UnitOfWork
    from gateway.repositories.files import FileRepository
    from gateway.services.files import sweep_files

    ids = []
    for name in ("stuck-1.txt", "stuck-2.txt", "fine.txt"):
        resp = client.post(
            f"{API_ROOT}/files", headers=api_key_header, files={"file": (name, b"payload", "text/plain")}
        )
        assert resp.status_code == 200, resp.text
        ids.append(str(resp.json()["id"]))
    db_session.query(FileObject).filter(FileObject.id.in_(ids)).update(
        {"expires_at": datetime.now(UTC) - timedelta(hours=1)}
    )
    db_session.commit()
    refs = {row.id: str(row.storage_ref) for row in db_session.query(FileObject).filter(FileObject.id.in_(ids)).all()}
    stuck = {refs[ids[0]], refs[ids[1]]}

    class _StickyStore(LocalDirFileStore):
        async def delete(self, storage_ref: str) -> None:
            if storage_ref in stuck:
                raise PermissionError(storage_ref)
            await super().delete(storage_ref)

    store = _StickyStore(str(tmp_path))

    async def _sweep_two_batches() -> list[tuple[int, int]]:
        engine = create_async_engine(make_url(test_config.database_url).set(drivername="postgresql+asyncpg"))
        try:
            async with async_sessionmaker(engine)() as db, UnitOfWork(db) as uow:
                first = await sweep_files(FileRepository(uow), store, batch_size=2)
                second = await sweep_files(FileRepository(uow), store, batch_size=2, after=first.cursor)
                return [(first.seen, first.reclaimed), (second.seen, second.reclaimed)]
        finally:
            await engine.dispose()

    # The first batch is the two stuck rows and reclaims nothing; the second,
    # started past them, reaches the one that can go.
    assert asyncio.run(_sweep_two_batches()) == [(2, 0), (1, 1)]
    db_session.expire_all()
    remaining = {row.id for row in db_session.query(FileObject).filter(FileObject.id.in_(ids)).all()}
    assert remaining == {ids[0], ids[1]}
    assert not (tmp_path / refs[ids[2]]).exists()


def test_a_deployment_that_does_not_serve_files_refuses_every_verb(
    files_off_client: TestClient, master_key_header: dict[str, str]
) -> None:
    """``files_enabled`` off answers 404 everywhere, so the API reads as unmounted."""
    upload = files_off_client.post(
        f"{API_ROOT}/files",
        headers=master_key_header,
        files={"file": ("a.txt", b"x", "text/plain")},
        data={"user": "someone"},
    )
    listing = files_off_client.get(f"{API_ROOT}/files", headers=master_key_header, params={"user": "someone"})
    read = files_off_client.get(f"{API_ROOT}/files/file-x", headers=master_key_header, params={"user": "someone"})

    assert [resp.status_code for resp in (upload, listing, read)] == [404, 404, 404]
    assert upload.json()["detail"] == "File uploads are disabled"


def test_a_disabled_deployment_refuses_before_it_reads_the_request(
    files_off_client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A surface that is not served answers the same way to every caller.

    The refusal comes before the request is parsed and before the caller is
    authenticated, so a disabled deployment cannot be told from an unmounted one
    by sending a request it would otherwise refuse for another reason.
    """
    # A master-key request with no ``user`` is a 400 on a deployment that serves files.
    no_user = files_off_client.post(
        f"{API_ROOT}/files", headers=master_key_header, files={"file": ("a.txt", b"x", "text/plain")}
    )
    # ``ids[]`` with ``limit`` is a 400 on a deployment that serves files.
    bad_params = files_off_client.get(
        f"{API_ROOT}/files",
        headers={**master_key_header, "anthropic-version": "2023-06-01"},
        params={"ids[]": "file-a", "limit": 5, "user": "someone"},
    )
    unauthenticated = files_off_client.get(f"{API_ROOT}/files/file-x")

    assert [resp.status_code for resp in (no_user, bad_params, unauthenticated)] == [404, 404, 404]
    assert no_user.json()["detail"] == "File uploads are disabled"


def test_a_storage_failure_answers_a_generic_500(
    client: TestClient, api_key_header: dict[str, str], tmp_file_store: None, tmp_path: Path
) -> None:
    """A 5xx names no internals: the condition goes to the log, the caller gets the house detail."""

    class _UnreadableStore(LocalDirFileStore):
        def get_stream(self, storage_ref: str) -> Any:
            raise OSError("disk gone")

    resp = client.post(
        f"{API_ROOT}/files", headers=api_key_header, files={"file": ("a.txt", b"payload", "text/plain")}
    )
    assert resp.status_code == 200, resp.text
    file_id = resp.json()["id"]

    cast(Any, client.app).state.file_store = _UnreadableStore(str(tmp_path))
    failed = client.get(f"{API_ROOT}/files/{file_id}/content", headers=api_key_header)

    assert failed.status_code == 500
    assert failed.json() == {"detail": "Internal server error"}


def test_a_file_id_that_could_name_nothing_is_a_404(client: TestClient, api_key_header: dict[str, str]) -> None:
    """A NUL in an ID reaches a bind parameter, so the refusal must not become a driver error."""
    resp = client.get(f"{API_ROOT}/files/file-%00x", headers=api_key_header)

    assert resp.status_code == 404, resp.text


def test_one_unusable_id_does_not_fail_a_batch_lookup(
    client: TestClient, api_key_header: dict[str, str], tmp_file_store: None, test_config: Any
) -> None:
    """A provider that announces an impossible ID must not cost the batch its valid files."""
    import asyncio

    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from gateway.core.unit_of_work import UnitOfWork
    from gateway.repositories.files import FileRepository

    resp = client.post(
        f"{API_ROOT}/files", headers=api_key_header, files={"file": ("a.txt", b"payload", "text/plain")}
    )
    assert resp.status_code == 200, resp.text
    stored = str(resp.json()["id"])

    async def _lookup() -> set[str]:
        engine = create_async_engine(make_url(test_config.database_url).set(drivername="postgresql+asyncpg"))
        try:
            async with async_sessionmaker(engine)() as db, UnitOfWork(db) as uow:
                return await FileRepository(uow).existing_ids([stored, "file-\x00x"])
        finally:
            await engine.dispose()

    assert asyncio.run(_lookup()) == {stored}
