"""Unit tests for the content normalizer.

The heavy extraction/vision boundaries (markitdown, vision side-call, file
store) are monkeypatched so these tests run with no optional deps and no DB.
"""

from __future__ import annotations

import base64
from typing import Any, cast

import pytest
from any_llm.types.completion import CompletionUsage

from gateway.core.config import GatewayConfig
from gateway.models.entities import FileObject
from gateway.services import content_normalizer as cn
from gateway.services.content_normalizer import normalize_messages
from gateway.services.file_extractors import ExtractionResult
from gateway.services.model_capabilities import Capabilities

_NATIVE = Capabilities(image=True, pdf=True, source="test")
_TEXT_ONLY = Capabilities(image=False, pdf=False, source="test")

_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")
_PNG_DATA_URL = f"data:image/png;base64,{_PNG_B64}"


def _image_msg(url: str = _PNG_DATA_URL) -> list[dict[str, Any]]:
    content = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": url}}]
    return [{"role": "user", "content": content}]


@pytest.mark.asyncio
async def test_string_content_untouched() -> None:
    cfg = GatewayConfig()
    msgs = [{"role": "user", "content": "plain text"}]
    out, stats = await normalize_messages(
        msgs, config=cfg, caps=_TEXT_ONLY, fmt="openai", db=None, file_store=None, user_id="u"
    )
    assert out == msgs
    assert not stats.touched


@pytest.mark.asyncio
async def test_bare_string_messages_untouched() -> None:
    # The Responses endpoint accepts a bare-string ``input``. Iterating it would
    # walk the string character-by-character; it must be returned verbatim.
    cfg = GatewayConfig()
    text = "What is the capital of France?"
    out, stats = await normalize_messages(
        cast("list[dict[str, Any]]", text),
        config=cfg,
        caps=_TEXT_ONLY,
        fmt="responses",
        db=None,
        file_store=None,
        user_id="u",
    )
    assert cast("Any", out) == text
    assert not stats.touched


@pytest.mark.asyncio
async def test_native_image_passthrough() -> None:
    cfg = GatewayConfig()
    msg = _image_msg()
    out, stats = await normalize_messages(
        msg, config=cfg, caps=_NATIVE, fmt="openai", db=None, file_store=None, user_id="u"
    )
    block = out[0]["content"][1]
    assert block["type"] == "image_url"
    # An already-inline block must pass through byte-identical — no decode/
    # re-encode round-trip for a native model.
    assert block["image_url"]["url"] == _PNG_DATA_URL
    assert not stats.touched


@pytest.mark.asyncio
async def test_text_only_image_described(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_describe(
        config: GatewayConfig, data_url: str
    ) -> tuple[str | None, CompletionUsage | None]:
        return "a red circle", CompletionUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18)

    monkeypatch.setattr(cn, "describe_image", fake_describe)
    cfg = GatewayConfig(vision_strategy="describe")
    out, stats = await normalize_messages(
        _image_msg(), config=cfg, caps=_TEXT_ONLY, fmt="openai", db=None, file_store=None, user_id="u"
    )
    block = out[0]["content"][1]
    assert block["type"] == "text"
    assert "a red circle" in block["text"]
    assert stats.images_described == 1
    # The describe side-call's usage is surfaced so the pipeline can bill it.
    usage = stats.vision_usage()
    assert usage is not None
    assert usage.prompt_tokens == 11
    assert usage.completion_tokens == 7


@pytest.mark.asyncio
async def test_text_only_image_dropped_when_off() -> None:
    cfg = GatewayConfig(vision_strategy="off")
    out, stats = await normalize_messages(
        _image_msg(), config=cfg, caps=_TEXT_ONLY, fmt="openai", db=None, file_store=None, user_id="u"
    )
    block = out[0]["content"][1]
    assert block["type"] == "text"
    assert "omitted" in block["text"]
    assert stats.dropped == 1


@pytest.mark.asyncio
async def test_remote_image_url_not_fetched_and_passed_through() -> None:
    cfg = GatewayConfig(vision_strategy="off")
    out, _ = await normalize_messages(
        _image_msg("https://example.com/a.png"),
        config=cfg,
        caps=_TEXT_ONLY,
        fmt="openai",
        db=None,
        file_store=None,
        user_id="u",
    )
    # Remote URLs are never fetched (SSRF-safe); the block is left for the model
    # turn marker since it can't be rendered.
    assert out[0]["content"][1]["type"] == "text"


@pytest.mark.asyncio
async def test_document_extracted_to_text(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_extract(data: bytes, mime: str, filename: str | None) -> ExtractionResult:
        return ExtractionResult("Quarterly numbers up.", True, "ok")

    monkeypatch.setattr(cn, "extract_text_from_file", fake_extract)
    cfg = GatewayConfig()
    pdf_data_url = "data:application/pdf;base64," + base64.b64encode(b"%PDF-1.4").decode("ascii")
    msgs = [
        {
            "role": "user",
            "content": [{"type": "file", "file": {"file_data": pdf_data_url, "filename": "q3.pdf"}}],
        }
    ]
    out, stats = await normalize_messages(
        msgs, config=cfg, caps=_TEXT_ONLY, fmt="openai", db=None, file_store=None, user_id="u"
    )
    block = out[0]["content"][0]
    assert block["type"] == "text"
    assert "q3.pdf" in block["text"]
    assert "Quarterly numbers up." in block["text"]
    assert stats.files_extracted == 1


@pytest.mark.asyncio
async def test_anthropic_image_passthrough_native() -> None:
    cfg = GatewayConfig()
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _PNG_B64}}
            ],
        }
    ]
    out, stats = await normalize_messages(
        msgs, config=cfg, caps=_NATIVE, fmt="anthropic", db=None, file_store=None, user_id="u"
    )
    assert out[0]["content"][0]["type"] == "image"
    assert out[0]["content"][0]["source"]["type"] == "base64"
    assert not stats.touched


@pytest.mark.asyncio
async def test_responses_uses_input_text_block() -> None:
    cfg = GatewayConfig(vision_strategy="off")
    msgs = [{"role": "user", "content": [{"type": "input_image", "image_url": _PNG_DATA_URL}]}]
    out, _ = await normalize_messages(
        msgs, config=cfg, caps=_TEXT_ONLY, fmt="responses", db=None, file_store=None, user_id="u"
    )
    assert out[0]["content"][0]["type"] == "input_text"


@pytest.mark.asyncio
async def test_file_id_resolved_and_inlined_for_native(monkeypatch: pytest.MonkeyPatch) -> None:
    record = FileObject(
        id="file-x",
        user_id="u",
        filename="pic.png",
        mime_type="image/png",
        bytes=8,
        purpose="user_data",
        storage_ref="x/file-x",
    )

    async def fake_fetch(db, file_id, user_id):  # type: ignore[no-untyped-def]
        assert file_id == "file-x"
        return record

    async def fake_read(file_store, rec):  # type: ignore[no-untyped-def]
        return b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr(cn, "fetch_file", fake_fetch)
    monkeypatch.setattr(cn, "read_file_bytes", fake_read)

    cfg = GatewayConfig()
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}, "file_id": "file-x"}]}]
    out, _ = await normalize_messages(
        msgs,
        config=cfg,
        caps=_NATIVE,
        fmt="openai",
        db=cast(Any, object()),
        file_store=cast(Any, object()),
        user_id="u",
    )
    block = out[0]["content"][0]
    # file_id rewritten to an inline data URL the upstream provider understands.
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_disabled_is_noop() -> None:
    cfg = GatewayConfig(file_understanding_enabled=False, vision_strategy="off")
    out, stats = await normalize_messages(
        _image_msg(), config=cfg, caps=_TEXT_ONLY, fmt="openai", db=None, file_store=None, user_id="u"
    )
    assert out == _image_msg()
    assert not stats.touched
