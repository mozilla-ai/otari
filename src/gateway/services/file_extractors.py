"""Extract a text representation of an uploaded file for text-only models.

PDF text, and the page images :func:`rasterize_pdf` renders when a PDF carries
no text (a scanned or image-only PDF) for the caller's vision fallback, come
from `pypdfium2` (Apache-2.0), deliberately avoiding AGPL-licensed PDF libraries
since the gateway is a network service. Office documents (docx, pptx, xlsx) are
zip archives of XML and are read with the standard library: a text-only model
needs their text, not their layout.

pypdfium2 and the OCR backend are imported lazily so the gateway starts even
when they are not installed.
"""

from __future__ import annotations

import asyncio
import codecs
import io
import mimetypes
import posixpath
import re
import threading
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import IO, Literal
from xml.etree import ElementTree

from gateway.log_config import logger

# PDFium is not thread-safe, not even across separate documents, and
# extractions and rasterizations run concurrently in the default executor.
_PDFIUM_LOCK = threading.Lock()

# The most text one extraction returns, far past any model's context window.
_MAX_TEXT_BYTES = 16 * 1024 * 1024
# The most of a text or HTML upload that is decoded to produce that text.
_MAX_TEXT_INPUT_BYTES = 4 * _MAX_TEXT_BYTES
# The most XML one office file may inflate to. A small upload can decompress to
# gigabytes, and the text ceiling does not stop markup that holds no text.
_MAX_ARCHIVE_XML_BYTES = 256 * 1024 * 1024
# The most XML elements held in memory at once: a whole part when it is parsed
# outright, one paragraph or row when it is streamed. Bytes alone do not bound
# memory, since each tiny element costs far more parsed than it does as text.
_MAX_HELD_ELEMENTS = 1_000_000
# What one kept string or row costs in memory beyond its text, counted against
# ``_MAX_TEXT_BYTES`` so that millions of tiny ones cannot pass for little.
_HELD_OBJECT_BYTES = 64
# What one spreadsheet cell slot costs whether or not it holds text: its pointer
# in the row's list and the " | " that renders it.
_CELL_SLOT_BYTES = 11
# The most pages an uploaded PDF is read for. Every PDFium call in the process
# shares one lock, so this bounds how long one upload holds up every other.
_MAX_PDF_PAGES = 1_000
# The share of a web page's text, chrome left out, that <main> or its <article>s
# must hold to be taken as the page's content rather than one part of it.
_MAIN_CONTENT_SHARE = 0.25
# Column XFD, the last one a spreadsheet may address.
_MAX_SHEET_COLUMNS = 16_384

# The number formats a spreadsheet application builds in for dates and times
# (ECMA-376 Part 1, 18.8.30); a workbook's own formats it declares in full.
_BUILTIN_DATE_FORMATS = frozenset({"14", "15", "16", "17", "18", "19", "20", "21", "22", "45", "46", "47"})
# Day zero of a workbook's serial dates. 1899-12-30 rather than 1900-01-01
# absorbs the leap day 1900 never had, which the format counts for every date
# after February 1900.
_EPOCH_1900 = datetime(1899, 12, 30)
_EPOCH_1904 = datetime(1904, 1, 1)

# MIME → file extension, for a block that carries no filename.
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

# Formats read as text even when they are not UTF-8. Any other extension is
# read as text only when it decodes as UTF-8, so a binary upload is refused.
_TEXT_EXTENSIONS = frozenset(
    {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml", ".yml", ".log"}
)
_HTML_EXTENSIONS = frozenset({".html", ".htm"})

# Subtrees whose text would repeat or mislead: an alternate-content fallback
# repeats its choice, paragraph and run properties hold tab stops rather than
# tabs, and phonetic runs annotate East Asian spreadsheet text.
_SKIPPED = frozenset({"Fallback", "pPr", "rPr", "rPh"})


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of a text-extraction attempt."""

    text: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class BoundedPDFExtractionResult:
    """Text-bearing PDF output for the remote retrieval worker."""

    text: str
    ok: bool
    truncated: bool = False


def _resolve_extension(mime_type: str, filename: str | None) -> str:
    if filename and (suffix := Path(filename).suffix):
        return suffix.lower()
    mime = (mime_type or "").split(";", 1)[0].strip().lower()
    if mime in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime]
    return mimetypes.guess_extension(mime) or ".bin"


class _ExtractionTooLarge(Exception):
    """An office file inflated past ``_MAX_ARCHIVE_XML_BYTES``."""


class _TextSink:
    """Output lines, up to ``_MAX_TEXT_BYTES``."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._remaining = _MAX_TEXT_BYTES
        self.full = False

    def add(self, line: str) -> bool:
        """Append ``line``, cutting it at the ceiling; False once the ceiling is reached."""
        if self.full:
            return False
        self._remaining, self.full = _append_text_with_byte_limit(self._parts, line + "\n", self._remaining)
        return not self.full

    def text(self) -> str:
        return "".join(self._parts).strip()


class _Archive:
    """An office file's zip, every byte of XML read counted against ``_MAX_ARCHIVE_XML_BYTES``."""

    def __init__(self, data: bytes) -> None:
        self._zip = zipfile.ZipFile(io.BytesIO(data))
        self._names = frozenset(self._zip.namelist())
        self.remaining = _MAX_ARCHIVE_XML_BYTES

    def has(self, name: str) -> bool:
        return name in self._names

    def parse(self, name: str) -> ElementTree.Element:
        root: ElementTree.Element | None = None
        with self._zip.open(name) as raw:
            for held, (_, element) in enumerate(ElementTree.iterparse(_CappedReader(raw, self), events=("start",))):
                if held >= _MAX_HELD_ELEMENTS:
                    raise _ExtractionTooLarge
                root = element if root is None else root
        if root is None:
            raise ValueError(f"{name} is empty")
        return root

    def iter_at_depth(self, name: str, depth: int) -> Iterator[ElementTree.Element]:
        """Yield each element ``depth`` levels below the root once it is complete, then free it.

        Streams the member, so a long document or sheet holds one paragraph or
        row in memory at a time rather than its whole tree.
        """
        stack: list[ElementTree.Element] = []
        held = 0
        with self._zip.open(name) as raw:
            for event, element in ElementTree.iterparse(_CappedReader(raw, self), events=("start", "end")):
                if event == "start":
                    stack.append(element)
                    held += 1
                    if held > _MAX_HELD_ELEMENTS:
                        raise _ExtractionTooLarge
                    continue
                stack.pop()
                if len(stack) == depth:
                    yield element
                    stack[-1].remove(element)
                    held = len(stack)


class _CappedReader:
    """A zip member's stream, drawing on its archive's budget."""

    def __init__(self, raw: IO[bytes], archive: _Archive) -> None:
        self._raw = raw
        self._archive = archive

    def read(self, size: int = -1, /) -> bytes:
        limit = self._archive.remaining + 1
        chunk = self._raw.read(limit if size < 0 else min(size, limit))
        self._archive.remaining -= len(chunk)
        if self._archive.remaining < 0:
            raise _ExtractionTooLarge
        return chunk


def _local(tag: str) -> str:
    """``tag`` without its namespace, so transitional and strict OOXML read alike."""
    return tag.rpartition("}")[2]


def _relationship_id(element: ElementTree.Element) -> str | None:
    """The namespaced ``r:id`` attribute, which a slide or sheet entry carries beside a plain ``id``."""
    return next((value for key, value in element.attrib.items() if key.startswith("{") and _local(key) == "id"), None)


def _relationships(archive: _Archive, part: str) -> list[tuple[str, str, str]]:
    """``(id, type, member)`` for each internal relationship of ``part``; ``""`` is the package itself."""
    base = posixpath.dirname(part)
    rels = posixpath.join(base, "_rels", posixpath.basename(part) + ".rels")
    if not archive.has(rels):
        return []
    found = []
    for relationship in archive.parse(rels):
        target = relationship.get("Target") or ""
        if not target or relationship.get("TargetMode") == "External":
            continue
        member = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(base, target))
        found.append((relationship.get("Id") or "", relationship.get("Type") or "", member))
    return found


def _main_part(archive: _Archive, default: str) -> str:
    return next(
        (member for _, kind, member in _relationships(archive, "") if kind.endswith("/officeDocument")), default
    )


def _inline_text(element: ElementTree.Element, limit: int | None = None) -> str:
    """The text of ``element``'s runs, each nested paragraph on a line of its own.

    Collection stops at ``limit`` characters (``_MAX_TEXT_BYTES`` by default),
    so a run too long to keep is never copied whole.
    """
    remaining = _MAX_TEXT_BYTES if limit is None else limit
    parts: list[str] = []

    def add(text: str) -> None:
        nonlocal remaining
        parts.append(text[:remaining])
        remaining -= len(parts[-1])

    def visit(node: ElementTree.Element) -> None:
        for child in node:
            if remaining <= 0:
                return
            name = _local(child.tag)
            if name in _SKIPPED:
                continue
            if name == "t":
                add(child.text or "")
            elif name == "tab":
                add("\t")
            elif name in ("br", "cr"):
                add("\n")
            else:
                if name == "p" and parts:
                    add("\n")
                visit(child)

    visit(element)
    return "".join(parts)


def _markdown_table(rows: list[list[str]]) -> Iterator[str]:
    """The table's lines, one at a time, so a caller writing to a sink stops building them once it is full.

    Every row is padded to the widest one, so a single wide row multiplies the
    size of all the others.
    """
    rows = [row for row in rows if any(row)]
    if not rows:
        return
    width = max(len(row) for row in rows)
    for index, row in enumerate(rows):
        cells = [" ".join(cell.split()).replace("|", "\\|") for cell in row] + [""] * (width - len(row))
        yield "| " + " | ".join(cells) + " |"
        if index == 0:
            yield "|" + " --- |" * width
    yield ""


def _table_rows(table: ElementTree.Element) -> list[list[str]]:
    return [
        [_inline_text(cell) for cell in row if _local(cell.tag) == "tc"] for row in table if _local(row.tag) == "tr"
    ]


def _word_prefix(paragraph: ElementTree.Element) -> str:
    """The Markdown a Word paragraph's style stands for: a heading level or a list item."""
    properties = next((child for child in paragraph if _local(child.tag) == "pPr"), None)
    for child in properties if properties is not None else ():
        name = _local(child.tag)
        if name == "pStyle":
            style = next((value for key, value in child.attrib.items() if _local(key) == "val"), "").lower()
            if style == "title":
                return "# "
            if match := re.fullmatch(r"heading(\d)", style):
                return "#" * min(int(match.group(1)), 6) + " "
        elif name == "numPr":
            return "- "
    return ""


def _write_blocks(elements: Iterable[ElementTree.Element], sink: _TextSink, *, word: bool) -> bool:
    """Write the paragraphs and tables under ``elements`` to ``sink``; False once it is full."""
    for element in elements:
        name = _local(element.tag)
        if name in _SKIPPED:
            continue
        if name == "p":
            text = _inline_text(element).strip()
            if text and not sink.add((_word_prefix(element) if word else "") + text):
                return False
        elif name == "tbl":
            if not all(sink.add(line) for line in _markdown_table(_table_rows(element))):
                return False
        elif not _write_blocks(element, sink, word=word):
            return False
    return True


def _docx_text(archive: _Archive, sink: _TextSink) -> None:
    # document > body > paragraph or table
    for block in archive.iter_at_depth(_main_part(archive, "word/document.xml"), 2):
        if not _write_blocks([block], sink, word=True):
            return


def _pptx_text(archive: _Archive, sink: _TextSink) -> None:
    presentation = _main_part(archive, "ppt/presentation.xml")
    targets = {rid: member for rid, _, member in _relationships(archive, presentation)}
    slide_ids = (_relationship_id(e) for e in archive.parse(presentation).iter() if _local(e.tag) == "sldId")
    slides = [targets[rid] for rid in slide_ids if rid in targets and archive.has(targets[rid])]
    for number, member in enumerate(slides, start=1):
        if not sink.add(f"## Slide {number}") or not _write_blocks([archive.parse(member)], sink, word=False):
            return
        sink.add("")


@dataclass(frozen=True)
class _Workbook:
    """What reading a cell needs from outside its sheet."""

    shared: list[str]
    # Indexes of the cell styles whose number format shows a date or a time.
    date_styles: frozenset[int]
    date1904: bool


def _is_date_format(code: str) -> bool:
    """Whether a custom number format shows a date or time, ignoring its literals, colors and locales."""
    bare = re.sub(r'"[^"]*"|\[[^\]]*\]|[\\_*].', "", code).lower()
    return re.search(r"[dyhs]", bare) is not None


def _date_styles(archive: _Archive, styles: str | None) -> frozenset[int]:
    if styles is None or not archive.has(styles):
        return frozenset()
    root = archive.parse(styles)
    custom = {e.get("numFmtId"): e.get("formatCode") or "" for e in root.iter() if _local(e.tag) == "numFmt"}
    cell_formats = next((child for child in root if _local(child.tag) == "cellXfs"), None)
    dates = set()
    for index, cell_format in enumerate(cell_formats if cell_formats is not None else ()):
        format_id = cell_format.get("numFmtId") or "0"
        code = custom.get(format_id)
        if _is_date_format(code) if code is not None else format_id in _BUILTIN_DATE_FORMATS:
            dates.add(index)
    return frozenset(dates)


def _serial_date(value: str, date1904: bool) -> str:
    """A date cell's stored day count as ISO 8601: a date, a time of day, or both."""
    try:
        serial = float(value)
        moment = (_EPOCH_1904 if date1904 else _EPOCH_1900) + timedelta(seconds=round(serial * 86_400))
    except (ValueError, OverflowError):
        return value
    if serial.is_integer():
        return moment.date().isoformat()
    if 0 <= serial < 1:
        return moment.time().isoformat()
    return moment.isoformat(sep=" ")


def _cell_value(cell: ElementTree.Element, book: _Workbook, limit: int) -> str:
    """The cell's text, at most ``limit`` characters of it.

    The limit applies to the resolved text, never to the stored value, which
    for a shared string is an index that a cut would turn into another one.
    """
    limit = max(0, limit)
    kind = cell.get("t")
    if kind == "inlineStr":
        inline = next((child for child in cell if _local(child.tag) == "is"), None)
        return "" if inline is None else _inline_text(inline, limit)
    value = next((child.text or "" for child in cell if _local(child.tag) == "v"), "")
    style = cell.get("s") or ""
    if kind == "s":
        value = book.shared[int(value)] if value.isdecimal() and int(value) < len(book.shared) else ""
    elif kind == "b":
        value = {"0": "FALSE", "1": "TRUE"}.get(value, value)
    elif value and kind in (None, "n") and style.isdecimal() and int(style) in book.date_styles:
        value = _serial_date(value, book.date1904)
    return value[:limit]


def _column_index(reference: str | None, default: int) -> int:
    letters = re.match(r"[A-Za-z]{1,3}", reference or "")
    if letters is None:
        return default
    index = 0
    for letter in letters.group().upper():
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _shared_strings(archive: _Archive, member: str) -> list[str]:
    """The workbook's string table, up to ``_MAX_TEXT_BYTES``; a cell naming an index past it reads as empty."""
    shared: list[str] = []
    size = 0
    # sst > si
    for item in archive.iter_at_depth(member, 1):
        text = _inline_text(item, _MAX_TEXT_BYTES - size)
        size += _HELD_OBJECT_BYTES + len(text)
        if size > _MAX_TEXT_BYTES:
            break
        shared.append(text)
    return shared


def _sheet_rows(archive: _Archive, member: str, book: _Workbook) -> list[list[str]]:
    rows: list[list[str]] = []
    size = 0
    # worksheet > sheetData > row
    for row in archive.iter_at_depth(member, 2):
        if _local(row.tag) != "row":
            continue
        cells: list[str] = []
        size += _HELD_OBJECT_BYTES
        for cell in row:
            if _local(cell.tag) != "c":
                continue
            column = _column_index(cell.get("r"), len(cells))
            value = _cell_value(cell, book, _MAX_TEXT_BYTES - size)
            if not value or column >= _MAX_SHEET_COLUMNS:
                continue
            # A cell far to the right pads the row with every slot before it,
            # so each slot is charged, not only the ones holding text.
            size += max(0, column + 1 - len(cells)) * _CELL_SLOT_BYTES + len(value)
            if size > _MAX_TEXT_BYTES:
                break
            if column >= len(cells):
                cells.extend([""] * (column - len(cells) + 1))
            cells[column] = value
        # An empty row renders as nothing, and keeping it would cost memory the
        # size check below never sees.
        if not cells:
            size -= _HELD_OBJECT_BYTES
            continue
        if size > _MAX_TEXT_BYTES:
            break
        rows.append(cells)
    return rows


def _xlsx_text(archive: _Archive, sink: _TextSink) -> None:
    workbook = _main_part(archive, "xl/workbook.xml")
    relationships = _relationships(archive, workbook)
    targets = {rid: member for rid, _, member in relationships}
    parts = {kind.rpartition("/")[2]: member for _, kind, member in relationships}
    strings = parts.get("sharedStrings")
    root = archive.parse(workbook)
    properties = next((e for e in root if _local(e.tag) == "workbookPr"), None)
    book = _Workbook(
        shared=_shared_strings(archive, strings) if strings and archive.has(strings) else [],
        date_styles=_date_styles(archive, parts.get("styles")),
        date1904=properties is not None and properties.get("date1904") in ("1", "true"),
    )
    for sheet in root.iter():
        member = targets.get(_relationship_id(sheet) or "") if _local(sheet.tag) == "sheet" else None
        if member is None or not archive.has(member):
            continue
        rows = _sheet_rows(archive, member, book)
        if not sink.add(f"## {sheet.get('name') or 'Sheet'}") or not all(
            sink.add(line) for line in _markdown_table(rows)
        ):
            return


_OFFICE_READERS: dict[str, Callable[[_Archive, _TextSink], None]] = {
    ".docx": _docx_text,
    ".docm": _docx_text,
    ".pptx": _pptx_text,
    ".pptm": _pptx_text,
    ".xlsx": _xlsx_text,
    ".xlsm": _xlsx_text,
}


class _HTMLText(HTMLParser):
    """The visible text of an HTML document as light Markdown: headings, list items and table rows.

    One pass fills four renderings, and :meth:`main_text` picks among them:
    everything, everything but the page's chrome, ``<main>``, and its ``<article>``s.
    """

    _HIDDEN = frozenset({"script", "style", "noscript", "template"})
    _BLOCKS = frozenset(
        {
            "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
            "footer", "header", "hr", "main", "nav", "ol", "p", "pre", "section", "table", "title", "ul",
        }
    )  # fmt: skip
    _OPENERS = {"li": "\n- ", "tr": "\n|", "td": " ", "th": " "} | {f"h{n}": "\n" + "#" * n + " " for n in range(1, 7)}
    _CLOSERS = {"td": " |", "th": " |"}
    # Elements that never close, so they can open no region.
    _VOID = frozenset(
        {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
    )
    # The page's chrome rather than its content, by element and by ARIA role.
    # <header> and <footer> count only outside <main> and <article>, where they
    # are the page's banner and footer rather than an article's own.
    _CHROME = frozenset({"nav", "aside", "dialog"})
    _PAGE_CHROME = frozenset({"header", "footer"})
    _CHROME_ROLES = frozenset(
        {"navigation", "banner", "contentinfo", "complementary", "search", "dialog", "alertdialog"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._everything: list[str] = []
        self._content: list[str] = []
        self._main: list[str] = []
        self._articles: list[str] = []
        # How many of each tag are open, and the regions each tag opened, as
        # (that count when it opened, its kind). A region closes when the count
        # of its tag drops below the one it opened at, so a <div role=navigation>
        # closes at the </div> that balances it. Every step is constant time,
        # whatever the nesting.
        self._open: Counter[str] = Counter()
        self._regions: dict[str, list[tuple[int, str]]] = {}
        self._kinds: Counter[str] = Counter()
        self._hidden = 0
        self._preformatted = 0

    def _within(self, kind: str) -> bool:
        return self._kinds[kind] > 0

    def _append(self, text: str) -> None:
        self._everything.append(text)
        if self._within("chrome"):
            return
        self._content.append(text)
        if self._within("main"):
            self._main.append(text)
        if self._within("article"):
            self._articles.append(text)

    def _region_kind(self, tag: str, attrs: list[tuple[str, str | None]]) -> str | None:
        attributes = dict(attrs)
        role = (attributes.get("role") or "").lower()
        if "hidden" in attributes or attributes.get("aria-hidden") == "true":
            return "chrome"
        if tag == "main" or role == "main":
            return "main"
        if tag == "article" or role == "article":
            return "article"
        if tag in self._CHROME or role in self._CHROME_ROLES:
            return "chrome"
        if tag in self._PAGE_CHROME and not (self._within("main") or self._within("article")):
            return "chrome"
        return None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self._VOID:
            self._open[tag] += 1
            if kind := self._region_kind(tag, attrs):
                self._regions.setdefault(tag, []).append((self._open[tag], kind))
                self._kinds[kind] += 1
        if tag == "pre":
            self._preformatted += 1
        if tag in self._HIDDEN:
            self._hidden += 1
        elif tag in self._OPENERS:
            self._append(self._OPENERS[tag])
        elif tag in self._BLOCKS:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self._preformatted = max(0, self._preformatted - 1)
        if tag in self._HIDDEN:
            self._hidden = max(0, self._hidden - 1)
        elif tag in self._CLOSERS:
            self._append(self._CLOSERS[tag])
        elif tag in self._BLOCKS or tag in self._OPENERS:
            self._append("\n")
        if self._open[tag] > 0:
            self._open[tag] -= 1
            regions = self._regions.get(tag)
            while regions and regions[-1][0] > self._open[tag]:
                self._kinds[regions.pop()[1]] -= 1

    def handle_data(self, data: str) -> None:
        # Outside <pre>, a line break in the source is only whitespace.
        if not self._hidden:
            self._append(data if self._preformatted else " ".join(data.splitlines()))

    def text(self) -> str:
        """Every piece of visible text on the page."""
        return _render_html_text(self._everything)

    def main_text(self) -> str:
        """The page's content without its chrome: navigation, banners, footers and sidebars.

        The longer of ``<main>`` and the page's ``<article>``s is the content
        where it holds a fair share of the page's text: a front page may keep
        only its featured story in ``<main>`` and the rest as articles outside
        it. A page marking neither keeps all of its text but its chrome, unless
        leaving the chrome out removed nearly everything, which is how markup
        that never closes a region shows.
        """
        content = _render_html_text(self._content)
        marked = max(_render_html_text(self._main), _render_html_text(self._articles), key=len)
        if marked and len(marked) >= _MAIN_CONTENT_SHARE * len(content):
            return marked
        everything = self.text()
        return content if len(content) >= _MAIN_CONTENT_SHARE * len(everything) else everything


def _render_html_text(parts: list[str]) -> str:
    lines: list[str] = []
    marker = ""
    for raw in "".join(parts).splitlines():
        line = " ".join(raw.split())
        # A list item or heading whose text sits in a nested block leaves
        # its marker alone on a line; it belongs to the next one.
        if line == "-" or (line and set(line) == {"#"}):
            marker = line + " "
        elif line:
            lines.append(marker + line)
            marker = ""
    return "\n".join(lines)


def html_main_text(html: str) -> str:
    """The content of a web page as light Markdown, its navigation and other chrome left out."""
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    return parser.main_text()


def _decode_text(data: bytes, *, lenient: bool) -> str | None:
    """The start of ``data`` as text, or None when it is not text.

    ``lenient`` falls back to Windows-1252 for a format known to be text.
    The decoders are incremental so a character the cut splits is held back
    rather than failing the decode.
    """
    head = data[:_MAX_TEXT_INPUT_BYTES]
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return codecs.getincrementaldecoder("utf-16")(errors="replace").decode(head)
    if b"\x00" in head[:8192]:
        return None
    try:
        return codecs.getincrementaldecoder("utf-8-sig")().decode(head)
    except UnicodeDecodeError:
        return head.decode("cp1252", errors="replace") if lenient else None


def _extract_sync(data: bytes, extension: str) -> ExtractionResult:
    sink = _TextSink()
    truncated = False
    try:
        if extension == ".pdf":
            pdf = extract_bounded_pdf_text_sync(
                data, max_pages=_MAX_PDF_PAGES, max_text_bytes=_MAX_TEXT_BYTES, past_max_pages="truncate"
            )
            sink.add(pdf.text.replace("\r\n", "\n"))
            truncated = pdf.truncated
        elif reader := _OFFICE_READERS.get(extension):
            reader(_Archive(data), sink)
        else:
            text = _decode_text(data, lenient=extension in _TEXT_EXTENSIONS or extension in _HTML_EXTENSIONS)
            if text is None:
                return ExtractionResult("", False, f"unsupported format {extension}")
            if extension in _HTML_EXTENSIONS:
                parser = _HTMLText()
                parser.feed(text)
                parser.close()
                text = parser.text()
            sink.add(text)
    except _ExtractionTooLarge:
        return ExtractionResult("", False, "document inflates past the extraction limit")
    except Exception as exc:  # noqa: BLE001 — surface as a failed extraction, not a 500
        logger.warning("text extraction failed for %s: %s", extension, exc)
        return ExtractionResult("", False, f"extraction error: {type(exc).__name__}")

    text = sink.text()
    if not text:
        return ExtractionResult("", False, "no extractable text")
    return ExtractionResult(text, True, "truncated" if truncated or sink.full else "ok")


async def extract_text_from_file(data: bytes, mime_type: str, filename: str | None) -> ExtractionResult:
    """Extract a Markdown/text representation of ``data``.

    Returns ``ok=False`` (with bytes intact for a possible vision fallback) when
    the format is unsupported, the document is too large, or no text could be read.
    """
    extension = _resolve_extension(mime_type, filename)
    return await asyncio.to_thread(_extract_sync, data, extension)


def _append_text_with_byte_limit(parts: list[str], value: str, remaining: int) -> tuple[int, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= remaining:
        parts.append(value)
        return remaining - len(encoded), False
    parts.append(encoded[:remaining].decode("utf-8", errors="ignore"))
    return 0, True


def extract_bounded_pdf_text_sync(
    data: bytes,
    *,
    max_pages: int,
    max_text_bytes: int,
    past_max_pages: Literal["refuse", "truncate"] = "refuse",
) -> BoundedPDFExtractionResult:
    """Extract PDF text up to fixed bounds, never rasterizing.

    A PDF of more than ``max_pages`` pages is refused before any page text is
    read, or with ``past_max_pages="truncate"`` read for its first ``max_pages``
    pages and reported truncated. Output stops at ``max_text_bytes`` of UTF-8.
    """
    if max_pages <= 0 or max_text_bytes <= 0:
        raise ValueError("PDF extraction limits must be positive")

    try:
        import pypdfium2 as pdfium
    except ImportError:
        return BoundedPDFExtractionResult("", False)

    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(data)
        try:
            if len(pdf) > max_pages and past_max_pages == "refuse":
                return BoundedPDFExtractionResult("", False)

            pages = min(len(pdf), max_pages)
            parts: list[str] = []
            remaining = max_text_bytes
            truncated = pages < len(pdf)
            for index in range(pages):
                page = pdf[index]
                try:
                    text_page = page.get_textpage()
                    try:
                        page_text = text_page.get_text_range().strip()
                    finally:
                        text_page.close()
                finally:
                    page.close()
                if not page_text:
                    continue
                prefix = "\n\n" if parts else ""
                remaining, prefix_truncated = _append_text_with_byte_limit(parts, prefix, remaining)
                if prefix_truncated or remaining == 0:
                    truncated = index < len(pdf) - 1 or bool(page_text)
                    break
                remaining, page_truncated = _append_text_with_byte_limit(parts, page_text, remaining)
                if page_truncated or remaining == 0:
                    truncated = page_truncated or index < len(pdf) - 1
                    break
        finally:
            pdf.close()

    text = "".join(parts).strip()
    if not text:
        return BoundedPDFExtractionResult("", False)
    return BoundedPDFExtractionResult(text, True, truncated)


def _rasterize_sync(data: bytes, max_pages: int) -> list[bytes]:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning("pypdfium2 not installed; cannot rasterize scanned PDF")
        return []

    images: list[bytes] = []
    with _PDFIUM_LOCK:
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
