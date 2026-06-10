"""Extract a text representation of an uploaded file for text-only models.

Text/office/PDF extraction goes through `markitdown` (MIT) which converts a
broad set of formats to Markdown. When a PDF carries no extractable text (a
scanned/image-only PDF), :func:`rasterize_pdf` renders its pages to PNGs via
`pypdfium2` (Apache-2.0) so the caller can fall back to the vision path —
deliberately avoiding AGPL-licensed PDF libraries since the gateway is a
network service.

All heavy dependencies are imported lazily so the gateway starts (and the rest
of the test suite runs) even when the optional extras aren't installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import mimetypes
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from gateway.log_config import logger

# MIME → file extension hints for formats markitdown keys off the suffix.
_MIME_TO_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/msword": ".doc",
    "text/csv": ".csv",
    "text/markdown": ".md",
    "text/plain": ".txt",
    "text/html": ".html",
    "application/json": ".json",
    "application/xml": ".xml",
    "text/xml": ".xml",
}


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of a text-extraction attempt."""

    text: str
    ok: bool
    detail: str


def _resolve_extension(mime_type: str, filename: str | None) -> str:
    if filename and (suffix := Path(filename).suffix):
        return suffix.lower()
    mime = (mime_type or "").split(";", 1)[0].strip().lower()
    if mime in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime]
    return mimetypes.guess_extension(mime) or ".bin"


def _extract_sync(data: bytes, extension: str) -> ExtractionResult:
    try:
        from markitdown import MarkItDown
    except ImportError:
        return ExtractionResult("", False, "markitdown not installed")

    # markitdown's stable API is path-based; write to a temp file with the right
    # suffix so format detection works across library versions. We close the
    # handle before markitdown re-opens the path and unlink it ourselves —
    # NamedTemporaryFile's auto-delete-on-close would race the re-open on Windows.
    tmp = tempfile.NamedTemporaryFile(suffix=extension, delete=False)  # noqa: SIM115
    try:
        tmp.write(data)
        tmp.close()
        result = MarkItDown(enable_plugins=False).convert(tmp.name)
    except Exception as exc:  # noqa: BLE001 — surface as a failed extraction, not a 500
        logger.warning("markitdown extraction failed for %s: %s", extension, exc)
        return ExtractionResult("", False, f"extraction error: {exc}")
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp.name)

    text = (result.text_content or "").strip()
    if not text:
        return ExtractionResult("", False, "no extractable text")
    return ExtractionResult(text, True, "ok")


async def extract_text_from_file(data: bytes, mime_type: str, filename: str | None) -> ExtractionResult:
    """Extract a Markdown/text representation of ``data``.

    Returns ``ok=False`` (with bytes intact for a possible vision fallback) when
    the format is unsupported, the library is missing, or no text could be read.
    """
    extension = _resolve_extension(mime_type, filename)
    return await asyncio.to_thread(_extract_sync, data, extension)


def _rasterize_sync(data: bytes, max_pages: int) -> list[bytes]:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning("pypdfium2 not installed; cannot rasterize scanned PDF")
        return []

    images: list[bytes] = []
    pdf = pdfium.PdfDocument(data)
    try:
        for index in range(min(len(pdf), max_pages)):
            page = pdf[index]
            bitmap = page.render(scale=2.0)
            pil_image = bitmap.to_pil()
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            images.append(buffer.getvalue())
    finally:
        pdf.close()
    return images


async def rasterize_pdf(data: bytes, max_pages: int = 10) -> list[bytes]:
    """Render up to ``max_pages`` PDF pages to PNG bytes for the vision fallback."""
    return await asyncio.to_thread(_rasterize_sync, data, max_pages)


def _ocr_sync(image_data: bytes) -> str:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        logger.warning("rapidocr_onnxruntime not installed; OCR unavailable")
        return ""

    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    result, _ = RapidOCR()(np.asarray(image))
    if not result:
        return ""
    return "\n".join(str(line[1]) for line in result).strip()


async def ocr_image(image_data: bytes) -> str:
    """Extract text from a raster image via RapidOCR (returns '' if unavailable)."""
    return await asyncio.to_thread(_ocr_sync, image_data)
