"""Integration tests for the /v1/files API and end-to-end file understanding.

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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import FileObject
from gateway.services.file_extractors import ExtractionResult
from gateway.services.file_store import LocalDirFileStore


@pytest.fixture
def tmp_file_store(client: TestClient, tmp_path: Path) -> None:
    """Point the app's blob store at a temp dir (default writes to cwd)."""
    cast(Any, client.app).state.file_store = LocalDirFileStore(str(tmp_path))


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
        choices=[
            Choice(index=0, message=ChatCompletionMessage(role="assistant", content="ok"), finish_reason="stop")
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )


def test_upload_requires_auth(client: TestClient) -> None:
    resp = client.post("/v1/files", files={"file": ("a.txt", b"x", "text/plain")})
    assert resp.status_code == 401


def test_upload_empty_file_rejected(
    client: TestClient, api_key_header: dict[str, str], tmp_file_store: None
) -> None:
    resp = client.post("/v1/files", headers=api_key_header, files={"file": ("empty.txt", b"", "text/plain")})
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
        "/v1/files",
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
    # Upload as the first key's user.
    up = client.post("/v1/files", headers=api_key_header, files={"file": ("a.txt", b"hi", "text/plain")})
    file_id = up.json()["id"]

    # A second, independent key (its own auto-created user) must not see it.
    other = client.post("/v1/keys", json={"key_name": "other"}, headers=master_key_header)
    # Match the fixture's auth scheme: the gateway requires a "Bearer " prefix on
    # every header form, including Otari-Key (see api_key_header / deps.py
    # _extract_bearer_token), so reuse the fixture's header name with a Bearer value.
    other_header = {next(iter(api_key_header)): f"Bearer {other.json()['key']}"}

    assert client.get(f"/v1/files/{file_id}", headers=other_header).status_code == 404
    assert client.get(f"/v1/files/{file_id}/content", headers=other_header).status_code == 404
    # Owner still sees it.
    assert client.get(f"/v1/files/{file_id}", headers=api_key_header).status_code == 200


def test_download_filename_header_is_injection_safe(
    client: TestClient, api_key_header: dict[str, str], tmp_file_store: None
) -> None:
    up = client.post(
        "/v1/files",
        headers=api_key_header,
        files={"file": ('ev"il\r\nX-Injected: 1.txt', b"data", "text/plain")},
    )
    assert up.status_code == 200
    file_id = up.json()["id"]

    resp = client.get(f"/v1/files/{file_id}/content", headers=api_key_header)
    assert resp.status_code == 200
    cd = resp.headers["content-disposition"]
    # No raw CR/LF or unescaped quotes leaked into the header.
    assert "\r" not in cd and "\n" not in cd
    assert "X-Injected" not in resp.headers
    assert "filename*=UTF-8''" in cd


def test_upload_get_list_delete_roundtrip(
    client: TestClient, api_key_header: dict[str, str], tmp_file_store: None
) -> None:
    up = client.post(
        "/v1/files",
        headers=api_key_header,
        data={"purpose": "user_data"},
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert up.status_code == 200, up.text
    obj = up.json()
    file_id = obj["id"]
    assert obj["object"] == "file"
    assert obj["bytes"] == len(b"hello world")
    assert obj["filename"] == "notes.txt"

    got = client.get(f"/v1/files/{file_id}", headers=api_key_header)
    assert got.status_code == 200
    assert got.json()["id"] == file_id

    content = client.get(f"/v1/files/{file_id}/content", headers=api_key_header)
    assert content.status_code == 200
    assert content.content == b"hello world"

    listed = client.get("/v1/files", headers=api_key_header)
    assert listed.status_code == 200
    assert any(f["id"] == file_id for f in listed.json()["data"])

    deleted = client.delete(f"/v1/files/{file_id}", headers=api_key_header)
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    # Gone after delete.
    assert client.get(f"/v1/files/{file_id}", headers=api_key_header).status_code == 404


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
        "/v1/files",
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
        resp = client.post("/v1/chat/completions", headers=api_key_header, json=body)

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

    resp = client.post("/v1/chat/completions", headers=api_key_header, json=body)
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
        "/v1/files",
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
        "messages": [
            {"role": "user", "content": [{"type": "file", "file": {"file_id": file_id}}]}
        ],
    }

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = client.post("/v1/chat/completions", headers=api_key_header, json=body)

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
        "/v1/files",
        headers=api_key_header,
        files={"file": ("a.txt", b"hi", "text/plain")},
    )
    file_id = up.json()["id"]

    # Backdate the stored expiry into the past, simulating retention elapsing.
    record = db_session.get(FileObject, file_id)
    assert record is not None
    record.expires_at = datetime.now(UTC) - timedelta(hours=1)
    db_session.commit()

    assert client.get(f"/v1/files/{file_id}", headers=api_key_header).status_code == 404
    assert client.get(f"/v1/files/{file_id}/content", headers=api_key_header).status_code == 404


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
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o-mini",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )
    client.post("/v1/users", json={"user_id": "vision-user"}, headers=master_key_header)

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
        resp = client.post("/v1/chat/completions", headers=master_key_header, json=body)

    assert resp.status_code == 200, resp.text
    # The caption reached the provider in place of the image block...
    assert "a chart of revenue" in json.dumps(captured["messages"])
    # ...and the describe side-call's cost (1000/1e6*2.5 + 500/1e6*10) was billed.
    user = client.get("/v1/users/vision-user", headers=master_key_header).json()
    assert user["spend"] == pytest.approx(0.0075)
