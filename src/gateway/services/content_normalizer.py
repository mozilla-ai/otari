"""Normalize file/image content blocks for the target model.

For each non-text content block in the request messages, decide — based on the
resolved :class:`~gateway.services.model_capabilities.Capabilities` — whether to:

* **pass through** to a natively-capable provider (resolving any ``file_id`` to
  inline bytes first, since the upstream provider doesn't know our file ids), or
* **extract to text** for a text-only model: documents via markitdown, images
  via a vision side-call / OCR, scanned PDFs via rasterize-then-describe, or
* **stage into the code-execution sandbox** when the request runs one: an
  Anthropic ``container_upload`` block names a file for the sandbox rather than
  the model, so it is recorded on the stats for the sandbox backend to seed and
  replaced by a short text marker telling the model the file is there.

The normalizer is format-aware (OpenAI chat, Anthropic messages, OpenAI
Responses) because each wire shape names its blocks and its text block
differently. It is defensive by construction: any per-block failure leaves the
original block untouched and is logged — it must never turn a chat request into
a 500.
"""

from __future__ import annotations

import base64
import binascii
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from any_llm.types.completion import CompletionUsage
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.file_extractors import extract_text_from_file, ocr_image, rasterize_pdf
from gateway.services.file_service import StagedFile, fetch_file, read_file_bytes
from gateway.services.file_store import FileStore
from gateway.services.model_capabilities import Capabilities
from gateway.services.vision import describe_image

WireFormat = Literal["openai", "anthropic", "responses"]

# Kinds of content we normalize.
_IMAGE = "image"
_DOCUMENT = "document"
# Anthropic's block for a file the code-execution container should see. Not a
# kind the model reads: with a sandbox in the request it is staged, without one
# it is treated as a document.
_CONTAINER = "container_upload"


@dataclass
class NormalizationStats:
    """Per-request tally, surfaced into usage-log metadata for observability."""

    files_extracted: int = 0
    images_described: int = 0
    dropped: int = 0
    chars_added: int = 0
    # Token usage of any vision describe side-calls made while extracting images
    # for a text-only model. Accumulated across every describe call (including
    # per-page calls for a scanned PDF) so the caller can meter and bill them.
    vision_prompt_tokens: int = 0
    vision_completion_tokens: int = 0
    details: list[str] = field(default_factory=list)
    # Uploads the request referenced for the code-execution sandbox, in message
    # order and without repeats. Only filled when the caller said a sandbox runs.
    sandbox_inputs: list[StagedFile] = field(default_factory=list)

    def stage(self, staged: StagedFile) -> None:
        if all(existing.file_id != staged.file_id for existing in self.sandbox_inputs):
            self.sandbox_inputs.append(staged)

    @property
    def touched(self) -> bool:
        return bool(self.files_extracted or self.images_described or self.dropped)

    def vision_usage(self) -> CompletionUsage | None:
        """The summed describe-call usage, or ``None`` if no describe call ran."""
        if not (self.vision_prompt_tokens or self.vision_completion_tokens):
            return None
        return CompletionUsage(
            prompt_tokens=self.vision_prompt_tokens,
            completion_tokens=self.vision_completion_tokens,
            total_tokens=self.vision_prompt_tokens + self.vision_completion_tokens,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "files_extracted": self.files_extracted,
            "images_described": self.images_described,
            "dropped": self.dropped,
            "chars_added": self.chars_added,
        }


@dataclass
class _Source:
    """A resolved content block's bytes + descriptor."""

    kind: str  # _IMAGE | _DOCUMENT
    data: bytes | None  # None for a remote URL we won't fetch (SSRF-safe)
    mime: str
    filename: str | None
    data_url: str | None  # original/remote data: or http(s) URL when present
    # True only when bytes came from a stored file_id, so a native model needs
    # the block rewritten to inline data. Already-inline / remote blocks pass
    # through unchanged (no wasteful decode→re-encode round-trip).
    needs_inline: bool = False
    # The stored upload behind a file_id block, so it can be staged into the
    # sandbox. None for inline and remote sources, which have no file to stage.
    staged: StagedFile | None = None


def _text_block(fmt: WireFormat, text: str) -> dict[str, Any]:
    if fmt == "responses":
        return {"type": "input_text", "text": text}
    return {"type": "text", "text": text}


def _decode_data_url(url: str) -> tuple[bytes | None, str]:
    """Return (bytes, mime) for a ``data:`` URL, or (None, '') if not one."""
    if not url.startswith("data:"):
        return None, ""
    header, _, payload = url[5:].partition(",")
    mime = header.split(";", 1)[0] or "application/octet-stream"
    if "base64" not in header:
        return payload.encode("utf-8"), mime
    try:
        return base64.b64decode(payload), mime
    except (binascii.Error, ValueError):
        return None, mime


def _to_data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


@dataclass
class _Resolved:
    """A descriptor resolved to bytes. ``staged`` is set when they came from a stored upload."""

    data: bytes | None
    mime: str
    filename: str | None
    staged: StagedFile | None = None

    def source(self, kind: str) -> _Source:
        return _Source(kind, self.data, self.mime, self.filename, None, self.staged is not None, self.staged)


async def _resolve_from_ref(
    ref: dict[str, Any],
    *,
    db: AsyncSession | None,
    file_store: FileStore | None,
    user_id: str | None,
    workspace_id: uuid.UUID | None,
    read_bytes: bool = True,
) -> _Resolved | None:
    """Resolve a ``{file_data|url|file_id, filename}`` descriptor to bytes.

    ``read_bytes=False`` resolves a ``file_id`` to its record without loading
    the blob, for a block that is only staged into the sandbox and never shown
    to the model.
    """
    filename = ref.get("filename")
    file_id = ref.get("file_id")
    if file_id and db is not None and file_store is not None:
        record = await fetch_file(db, str(file_id), user_id, workspace_id=workspace_id)
        if record is None:
            logger.warning("content normalizer: file_id %s not found for user %s", file_id, user_id)
            return None
        staged = StagedFile(record.id, record.filename, record.mime_type, record.storage_ref)
        data = await read_file_bytes(file_store, record) if read_bytes else None
        return _Resolved(data, record.mime_type, record.filename, staged)

    url = ref.get("file_data") or ref.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        decoded, mime = _decode_data_url(url)
        if decoded is not None:
            return _Resolved(decoded, mime, filename)
    return None


async def _classify(
    block: dict[str, Any],
    fmt: WireFormat,
    *,
    db: AsyncSession | None,
    file_store: FileStore | None,
    user_id: str | None,
    workspace_id: uuid.UUID | None,
    sandbox_requested: bool = False,
) -> _Source | None:
    """Identify an image/document/container block and resolve its bytes, or return None."""
    btype = block.get("type")

    # --- sandbox input blocks ------------------------------------------
    if fmt == "anthropic" and btype == _CONTAINER:
        # Read the bytes only when there is no sandbox to stage into and the
        # block falls back to being a document the model reads.
        resolved = await _resolve_from_ref(
            block,
            db=db,
            file_store=file_store,
            user_id=user_id,
            workspace_id=workspace_id,
            read_bytes=not sandbox_requested,
        )
        return resolved.source(_CONTAINER) if resolved else None

    # --- image blocks ---------------------------------------------------
    if (fmt in ("openai", "responses") and btype in ("image_url", "input_image")) or (
        fmt == "anthropic" and btype == "image"
    ):
        if fmt == "anthropic":
            src = block.get("source", {})
            if src.get("type") == "base64":
                data = _decode_data_url(f"data:{src.get('media_type', '')};base64,{src.get('data', '')}")[0]
                return _Source(_IMAGE, data, src.get("media_type", "image/png"), None, None)
            if src.get("type") == "file":
                resolved = await _resolve_from_ref(
                    src, db=db, file_store=file_store, user_id=user_id, workspace_id=workspace_id
                )
                if resolved:
                    return resolved.source(_IMAGE)
            return _Source(_IMAGE, None, "image/png", None, src.get("url"))
        # openai / responses image
        image_url = block.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if block.get("file_id"):
            resolved = await _resolve_from_ref(
                block, db=db, file_store=file_store, user_id=user_id, workspace_id=workspace_id
            )
            if resolved:
                return resolved.source(_IMAGE)
        if isinstance(url, str):
            data, mime = _decode_data_url(url)
            return _Source(_IMAGE, data, mime or "image/png", None, None if data else url)
        return None

    # --- document/file blocks ------------------------------------------
    if (fmt in ("openai", "responses") and btype in ("file", "input_file")) or (
        fmt == "anthropic" and btype == "document"
    ):
        if fmt == "anthropic":
            src = block.get("source", {})
            if src.get("type") == "text":
                return _Source(_DOCUMENT, str(src.get("data", "")).encode("utf-8"), "text/plain", None, None)
            if src.get("type") == "base64":
                data = _decode_data_url(f"data:{src.get('media_type', '')};base64,{src.get('data', '')}")[0]
                return _Source(_DOCUMENT, data, src.get("media_type", "application/pdf"), None, None)
            if src.get("type") == "file":
                resolved = await _resolve_from_ref(
                    src, db=db, file_store=file_store, user_id=user_id, workspace_id=workspace_id
                )
                if resolved:
                    return resolved.source(_DOCUMENT)
            return _Source(_DOCUMENT, None, "application/pdf", None, src.get("url"))
        ref = block.get("file", block) if btype == "file" else block
        resolved = await _resolve_from_ref(
            ref, db=db, file_store=file_store, user_id=user_id, workspace_id=workspace_id
        )
        if resolved:
            return resolved.source(_DOCUMENT)
        return None

    return None


def _inline_passthrough(block: dict[str, Any], fmt: WireFormat, src: _Source) -> dict[str, Any]:
    """Rewrite a (possibly file_id) block to inline bytes for a native provider."""
    if src.data is None:  # remote URL or unresolved — forward unchanged
        return block
    data_url = _to_data_url(src.data, src.mime)
    if fmt == "anthropic":
        atype = "image" if src.kind == _IMAGE else "document"
        return {
            "type": atype,
            "source": {"type": "base64", "media_type": src.mime, "data": base64.b64encode(src.data).decode("ascii")},
        }
    if src.kind == _IMAGE:
        if fmt == "responses":
            return {"type": "input_image", "image_url": data_url}
        return {"type": "image_url", "image_url": {"url": data_url}}
    # document
    file_obj = {"file_data": data_url}
    if src.filename:
        file_obj["filename"] = src.filename
    if fmt == "responses":
        return {"type": "input_file", "file_data": data_url, **({"filename": src.filename} if src.filename else {})}
    return {"type": "file", "file": file_obj}


async def _extract_to_text(src: _Source, config: GatewayConfig, stats: NormalizationStats) -> str | None:
    """Produce a text rendering of a non-text block for a text-only model."""
    if src.data is None:
        stats.details.append(f"skip remote {src.kind} url (not fetched)")
        return None

    if src.kind == _DOCUMENT:
        result = await extract_text_from_file(src.data, src.mime, src.filename)
        if result.ok:
            label = src.filename or "document"
            return f"[Attached file: {label}]\n{result.text}\n[End of {label}]"
        # Scanned/image-only PDF → rasterize and describe pages.
        if src.mime.split(";", 1)[0] == "application/pdf":
            return await _describe_pdf_pages(src, config, stats)
        stats.details.append(f"document extract failed: {result.detail}")
        return None

    # image
    return await _render_image(src, config, stats)


async def _render_image(src: _Source, config: GatewayConfig, stats: NormalizationStats) -> str | None:
    assert src.data is not None
    strategy = config.vision_strategy
    if strategy == "describe":
        described, usage = await describe_image(config, _to_data_url(src.data, src.mime))
        if usage is not None:
            stats.vision_prompt_tokens += usage.prompt_tokens or 0
            stats.vision_completion_tokens += usage.completion_tokens or 0
        if described:
            return f"[Attached image description]\n{described}"
        # Fall through to OCR as a cheaper recovery, then give up.
    if strategy in ("describe", "ocr"):
        text = await ocr_image(src.data)
        if text:
            return f"[Attached image — extracted text]\n{text}"
    stats.details.append(f"image dropped (strategy={strategy}, no description available)")
    return None


async def _describe_pdf_pages(src: _Source, config: GatewayConfig, stats: NormalizationStats) -> str | None:
    assert src.data is not None
    pages = await rasterize_pdf(src.data)
    if not pages:
        stats.details.append("scanned PDF: rasterization unavailable")
        return None
    descriptions: list[str] = []
    for index, page_png in enumerate(pages, start=1):
        page_src = _Source(_IMAGE, page_png, "image/png", None, None)
        rendered = await _render_image(page_src, config, stats)
        if rendered:
            descriptions.append(f"-- Page {index} --\n{rendered}")
    if not descriptions:
        return None
    label = src.filename or "document.pdf"
    return f"[Attached scanned file: {label}]\n" + "\n".join(descriptions) + f"\n[End of {label}]"


async def _normalize_block(
    block: Any,
    fmt: WireFormat,
    caps: Capabilities,
    config: GatewayConfig,
    stats: NormalizationStats,
    *,
    db: AsyncSession | None,
    file_store: FileStore | None,
    user_id: str | None,
    workspace_id: uuid.UUID | None,
    sandbox_requested: bool = False,
) -> Any:
    if not isinstance(block, dict):
        return block
    try:
        src = await _classify(
            block,
            fmt,
            db=db,
            file_store=file_store,
            user_id=user_id,
            workspace_id=workspace_id,
            sandbox_requested=sandbox_requested,
        )
    except Exception as exc:  # noqa: BLE001 — never fail the request over a block
        logger.warning("content normalizer: failed to classify block: %s", exc)
        return block
    if src is None:
        return block

    if sandbox_requested and src.staged is not None:
        stats.stage(src.staged)
    if src.kind == _CONTAINER:
        if sandbox_requested and src.staged is not None:
            # The sandbox gets the bytes; the model gets told where they are.
            return _text_block(fmt, f"[File available in the code execution sandbox: {src.staged.filename}]")
        src.kind = _DOCUMENT

    native = caps.image if src.kind == _IMAGE else caps.pdf
    if native:
        # Only rewrite when bytes came from a stored file_id (the provider can't
        # resolve our ids); already-inline / remote blocks pass through as-is.
        return _inline_passthrough(block, fmt, src) if src.needs_inline else block

    try:
        text = await _extract_to_text(src, config, stats)
    except Exception as exc:  # noqa: BLE001
        logger.warning("content normalizer: extraction failed for %s: %s", src.kind, exc)
        return block

    if text is None:
        stats.dropped += 1
        logger.info("content normalizer: dropped a %s block (no text rendering)", src.kind)
        # Drop the block entirely; leaving it would let the provider silently
        # ignore it. A short marker keeps the turn coherent for the model.
        return _text_block(fmt, f"[Unprocessable {src.kind} attachment omitted]")

    if src.kind == _IMAGE:
        stats.images_described += 1
    else:
        stats.files_extracted += 1
    stats.chars_added += len(text)
    return _text_block(fmt, text)


# Content parts the Responses API also accepts as bare ``input`` items.
_RESPONSES_ITEM_TYPES = frozenset({"input_file", "input_image"})


def _wrap_bare_responses_item(item: Any) -> Any:
    """Put a bare item the normalizer turned into text back into a valid position.

    A file or image item is valid at the top level of a Responses ``input``, but
    the text it extracts to is not: ``input_text`` only lives inside a message.
    """
    if isinstance(item, dict) and item.get("type") == "input_text":
        return {"role": "user", "content": [item]}
    return item


async def normalize_messages(
    messages: list[dict[str, Any]],
    *,
    config: GatewayConfig,
    caps: Capabilities,
    fmt: WireFormat,
    db: AsyncSession | None,
    file_store: FileStore | None,
    user_id: str | None,
    workspace_id: uuid.UUID | None = None,
    sandbox_requested: bool = False,
) -> tuple[list[dict[str, Any]], NormalizationStats]:
    """Return (possibly-rewritten messages, stats).

    ``workspace_id`` confines ``file_id`` resolution to one workspace, and is the
    workspace of the API key that authenticated the request. ``None`` means "any",
    which is what a master-key request gets: the operator acting deployment-wide.

    ``sandbox_requested`` says the request runs the gateway's code-execution
    sandbox. Every stored upload the messages reference is then also recorded on
    ``stats.sandbox_inputs`` for the sandbox to seed, and ``container_upload``
    blocks are staged instead of read.

    Messages whose ``content`` is a plain string are returned untouched (the
    common, zero-overhead path). Only list-content messages are walked, plus, on
    the Responses format, a file or image item placed directly in ``input``
    rather than inside a message.

    The Responses endpoint accepts a bare-string ``input``; iterating that would
    walk it character-by-character, so non-list ``messages`` are returned as-is.
    """
    stats = NormalizationStats()
    if not config.file_understanding_enabled or not isinstance(messages, list):
        return messages, stats

    async def _block(block: Any) -> Any:
        return await _normalize_block(
            block,
            fmt,
            caps,
            config,
            stats,
            db=db,
            file_store=file_store,
            user_id=user_id,
            workspace_id=workspace_id,
            sandbox_requested=sandbox_requested,
        )

    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            if fmt == "responses" and isinstance(message, dict) and message.get("type") in _RESPONSES_ITEM_TYPES:
                out.append(_wrap_bare_responses_item(await _block(message)))
            else:
                out.append(message)
            continue
        new_content = [await _block(block) for block in content]
        out.append({**message, "content": new_content})

    if stats.touched:
        logger.info(
            "content normalizer: extracted=%d described=%d dropped=%d (+%d chars)",
            stats.files_extracted,
            stats.images_described,
            stats.dropped,
            stats.chars_added,
        )
    return out, stats
