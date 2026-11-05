"""Unit tests for `gateway.services.file_extractors`.

Office documents are built here as the zips of XML they are, holding only the
parts the extractor reads, so each test states the markup it covers.
"""

from __future__ import annotations

import asyncio
import ctypes
import io
import zipfile
from typing import Any

import pytest

from gateway.services import file_extractors

_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
_P = 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
_S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
_R = f'xmlns:r="{_REL}"'


def _zip(parts: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _rels(*relationships: tuple[str, str, str]) -> str:
    entries = "".join(
        f'<Relationship Id="{rid}" Type="{_REL}/{kind}" Target="{target}"/>' for rid, kind, target in relationships
    )
    return (
        f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{entries}</Relationships>'
    )


def _paragraph(text: str, style: str | None = None, *, numbered: bool = False) -> str:
    properties = (f'<w:pStyle w:val="{style}"/>' if style else "") + ("<w:numPr/>" if numbered else "")
    return f"<w:p><w:pPr>{properties}<w:tabs><w:tab/></w:tabs></w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>"


def _docx(body: str) -> bytes:
    return _zip(
        {
            "_rels/.rels": _rels(("rId1", "officeDocument", "word/document.xml")),
            "word/document.xml": f"<w:document {_W}><w:body>{body}</w:body></w:document>",
        }
    )


def test_docx_keeps_headings_lists_and_tables() -> None:
    cell = "<w:tc><w:p><w:r><w:t>{}</w:t></w:r></w:p></w:tc>"
    body = (
        _paragraph("Report", "Heading1")
        + _paragraph("Revenue grew.")
        + _paragraph("First item", numbered=True)
        + "<w:tbl>"
        + "".join(f"<w:tr>{cell.format(a)}{cell.format(b)}</w:tr>" for a, b in (("Region", "Revenue"), ("EU", "1|0")))
        + "</w:tbl>"
        + "<w:p><w:r><w:t>Kept</w:t></w:r><w:del><w:r><w:delText>deleted</w:delText></w:r></w:del></w:p>"
    )

    result = file_extractors._extract_sync(_docx(body), ".docx")

    assert result.ok is True
    assert result.text == (
        "# Report\nRevenue grew.\n- First item\n| Region | Revenue |\n| --- | --- |\n| EU | 1\\|0 |\n\nKept"
    )


def test_docx_reads_one_alternate_content_branch() -> None:
    """A text box is written twice, as the choice and as its fallback; only one is read."""
    body = (
        '<w:p xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"><w:r><mc:AlternateContent>'
        "<mc:Choice><w:t>Boxed</w:t></mc:Choice><mc:Fallback><w:t>Boxed</w:t></mc:Fallback>"
        "</mc:AlternateContent></w:r></w:p>"
    )

    assert file_extractors._extract_sync(_docx(body), ".docx").text == "Boxed"


def test_docx_keeps_tabs_breaks_titles_and_a_cell_of_several_paragraphs() -> None:
    cell = "<w:tc><w:p><w:r><w:t>Line one</w:t></w:r></w:p><w:p><w:r><w:t>Line two</w:t></w:r></w:p></w:tc>"
    body = (
        "<w:p><w:r><w:t>Name</w:t><w:tab/><w:t>Value</w:t><w:br/><w:t>Next line</w:t></w:r></w:p>"
        + _paragraph("Plan", "Title")
        + _paragraph("Deep", "Heading9")
        + f"<w:tbl><w:tr>{cell}</w:tr></w:tbl>"
    )

    result = file_extractors._extract_sync(_docx(body), ".docx")

    assert result.text == "Name\tValue\nNext line\n# Plan\n###### Deep\n| Line one Line two |\n| --- |"


def test_docx_stops_reading_once_the_text_ceiling_is_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markup past the ceiling is never parsed, so a broken tail far beyond it does not fail the extraction."""
    monkeypatch.setattr(file_extractors, "_MAX_TEXT_BYTES", 30)
    data = _zip(
        {
            "_rels/.rels": _rels(("rId1", "officeDocument", "word/document.xml")),
            "word/document.xml": f"<w:document {_W}><w:body>{_paragraph('first paragraph') * 1000}<w:p>",
        }
    )

    result = file_extractors._extract_sync(data, ".docx")

    assert result.ok is True
    assert result.detail == "truncated"
    assert result.text == "first paragraph\nfirst paragrap"


def test_pptx_follows_the_presentation_order_not_the_file_names() -> None:
    def slide(text: str) -> str:
        return (
            f"<p:sld {_P} {_A}><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p>"
            "</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
        )

    data = _zip(
        {
            "_rels/.rels": _rels(("rId1", "officeDocument", "ppt/presentation.xml")),
            "ppt/presentation.xml": (
                f'<p:presentation {_P} {_R}><p:sldIdLst><p:sldId id="256" r:id="rId3"/>'
                '<p:sldId id="257" r:id="rId2"/></p:sldIdLst></p:presentation>'
            ),
            "ppt/_rels/presentation.xml.rels": _rels(
                ("rId2", "slide", "slides/slide1.xml"), ("rId3", "slide", "slides/slide2.xml")
            ),
            "ppt/slides/slide1.xml": slide("Shown second"),
            "ppt/slides/slide2.xml": slide("Shown first"),
        }
    )

    result = file_extractors._extract_sync(data, ".pptx")

    assert result.text == "## Slide 1\nShown first\n\n## Slide 2\nShown second"


def _xlsx(sheet_data: str, *, date1904: bool = False) -> bytes:
    return _zip(
        {
            "_rels/.rels": _rels(("rId1", "officeDocument", "xl/workbook.xml")),
            "xl/workbook.xml": (
                f'<workbook {_S} {_R}><workbookPr date1904="{int(date1904)}"/>'
                '<sheets><sheet name="Sales" sheetId="1" r:id="rId1"/></sheets></workbook>'
            ),
            "xl/_rels/workbook.xml.rels": _rels(
                ("rId1", "worksheet", "worksheets/sheet1.xml"),
                ("rId2", "sharedStrings", "sharedStrings.xml"),
                ("rId3", "styles", "styles.xml"),
            ),
            "xl/sharedStrings.xml": (
                f"<sst {_S}><si><t>Region</t></si><si><r><t>Ro</t></r><r><t>w</t></r><rPh><t>ro</t></rPh></si></sst>"
            ),
            # Style 1 is a built-in date format, 2 a custom one, 3 a custom
            # number whose quoted literal holds a "d".
            "xl/styles.xml": (
                f'<styleSheet {_S}><numFmts><numFmt numFmtId="164" formatCode="dd/mm/yyyy hh:mm"/>'
                '<numFmt numFmtId="165" formatCode="0.0&quot; days&quot;"/></numFmts>'
                '<cellXfs><xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="164"/><xf numFmtId="165"/></cellXfs>'
                "</styleSheet>"
            ),
            "xl/worksheets/sheet1.xml": f"<worksheet {_S}><sheetData>{sheet_data}</sheetData></worksheet>",
        }
    )


def test_xlsx_reads_every_cell_kind_as_a_table() -> None:
    sheet = (
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>Inline</t></is></c></row>'
        '<row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2" t="b"><v>1</v></c><c r="D2"><v>2.5</v></c></row>'
        '<row r="3"><c r="A3" s="1"><v>46321</v></c><c r="B3" s="2"><v>46321.5</v></c>'
        '<c r="C3" s="3"><v>4</v></c></row>'
    )

    result = file_extractors._extract_sync(_xlsx(sheet), ".xlsx")

    assert result.text == (
        "## Sales\n"
        "| Region | Inline |  |  |\n"
        "| --- | --- | --- | --- |\n"
        "| Row | TRUE |  | 2.5 |\n"
        "| 2026-10-26 | 2026-10-26 12:00:00 | 4 |  |"
    )


def test_xlsx_dates_follow_the_1904_system() -> None:
    sheet = '<row r="1"><c r="A1" s="1"><v>0</v></c></row>'

    result = file_extractors._extract_sync(_xlsx(sheet, date1904=True), ".xlsx")

    assert result.text == "## Sales\n| 1904-01-01 |\n| --- |"


def test_xlsx_shows_a_time_of_day_and_keeps_a_date_cell_that_is_not_a_number() -> None:
    sheet = '<row r="1"><c r="A1" s="1"><v>0.75</v></c><c r="B1" s="1"><v>soon</v></c></row>'

    result = file_extractors._extract_sync(_xlsx(sheet), ".xlsx")

    assert result.text == "## Sales\n| 18:00:00 | soon |\n| --- | --- |"


def test_xlsx_places_a_cell_without_a_reference_after_the_one_before_it() -> None:
    sheet = '<row><c><v>1</v></c><c r="C1"><v>3</v></c><c><v>4</v></c></row>'

    assert (
        file_extractors._extract_sync(_xlsx(sheet), ".xlsx").text
        == "## Sales\n| 1 |  | 3 | 4 |\n| --- | --- | --- | --- |"
    )


def test_xlsx_parses_no_sheet_past_the_text_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_extractors, "_MAX_TEXT_BYTES", 2000)
    # One wide row pads every other row's rendering, so the table outgrows the
    # ceiling while the rows held to build it stay within it.
    rows = '<row><c r="A1"><v>1</v></c><c r="AX1"><v>2</v></c></row>' + "".join(
        f'<row><c r="A{n}"><v>{n}</v></c></row>' for n in range(2, 41)
    )
    data = _zip(
        {
            "xl/workbook.xml": (
                f'<workbook {_S} {_R}><sheets><sheet name="One" sheetId="1" r:id="rId1"/>'
                '<sheet name="Two" sheetId="2" r:id="rId2"/></sheets></workbook>'
            ),
            "xl/_rels/workbook.xml.rels": _rels(
                ("rId1", "worksheet", "worksheets/sheet1.xml"), ("rId2", "worksheet", "worksheets/sheet2.xml")
            ),
            "xl/worksheets/sheet1.xml": f"<worksheet {_S}><sheetData>{rows}</sheetData></worksheet>",
            "xl/worksheets/sheet2.xml": "<worksheet",
        }
    )

    result = file_extractors._extract_sync(data, ".xlsx")

    assert result.detail == "truncated"
    assert result.text.startswith("## One\n| 1 |")
    assert "## Two" not in result.text


def test_xlsx_ignores_a_column_past_the_last_one() -> None:
    sheet = '<row r="1"><c r="A1"><v>1</v></c><c r="ZZZ1"><v>2</v></c></row>'

    assert file_extractors._extract_sync(_xlsx(sheet), ".xlsx").text == "## Sales\n| 1 |\n| --- |"


def test_xlsx_keeps_no_empty_rows() -> None:
    """An empty row adds no text, so keeping it would cost memory no ceiling counts."""
    archive = file_extractors._Archive(_xlsx("<row/>" * 1000 + '<row r="1001"><c r="A1001"><v>1</v></c></row>'))
    book = file_extractors._Workbook(shared=[], date_styles=frozenset(), date1904=False)

    assert file_extractors._sheet_rows(archive, "xl/worksheets/sheet1.xml", book) == [["1"]]


def test_xlsx_rows_count_their_memory_against_the_text_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_extractors, "_MAX_TEXT_BYTES", 10 * file_extractors._HELD_OBJECT_BYTES)
    archive = file_extractors._Archive(_xlsx('<row><c r="A1"><v>1</v></c></row>' * 100))
    book = file_extractors._Workbook(shared=[], date_styles=frozenset(), date1904=False)

    assert len(file_extractors._sheet_rows(archive, "xl/worksheets/sheet1.xml", book)) <= 10


def test_xlsx_does_not_keep_the_row_that_passes_the_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_extractors, "_MAX_TEXT_BYTES", 1000)
    sheet = '<row><c r="A1"><v>1</v></c></row><row><c r="A2" t="inlineStr"><is><t>' + "x" * 5000 + "</t></is></c></row>"
    archive = file_extractors._Archive(_xlsx(sheet))
    book = file_extractors._Workbook(shared=[], date_styles=frozenset(), date1904=False)

    assert file_extractors._sheet_rows(archive, "xl/worksheets/sheet1.xml", book) == [["1"]]


def test_a_shared_string_index_is_resolved_before_the_cell_is_cut() -> None:
    """Cutting the stored index ``10`` to ``1`` would read another string."""
    from xml.etree import ElementTree

    book = file_extractors._Workbook(shared=["", "wrong", *[""] * 8, "right"], date_styles=frozenset(), date1904=False)
    cell = ElementTree.fromstring('<c t="s"><v>10</v></c>')

    assert file_extractors._cell_value(cell, book, 1) == "r"


def test_xlsx_charges_every_slot_a_far_right_cell_pads(monkeypatch: pytest.MonkeyPatch) -> None:
    row_cost = file_extractors._HELD_OBJECT_BYTES + 16_384 * file_extractors._CELL_SLOT_BYTES + 1
    monkeypatch.setattr(file_extractors, "_MAX_TEXT_BYTES", 2 * row_cost)
    archive = file_extractors._Archive(_xlsx('<row><c r="XFD1"><v>1</v></c></row>' * 10))
    book = file_extractors._Workbook(shared=[], date_styles=frozenset(), date1904=False)

    rows = file_extractors._sheet_rows(archive, "xl/worksheets/sheet1.xml", book)

    assert len(rows) == 2
    assert len(rows[0]) == 16_384


def test_a_table_is_rendered_one_line_at_a_time() -> None:
    """One wide row pads every other one, so the lines are not all built before the sink stops."""
    lines = file_extractors._markdown_table([["a"], ["b"]])

    assert next(lines) == "| a |"


def test_run_text_stops_at_its_limit() -> None:
    from xml.etree import ElementTree

    element = ElementTree.fromstring("<si><t>abc</t><t>defgh</t><t>ijk</t></si>")

    assert file_extractors._inline_text(element, 5) == "abcde"
    assert file_extractors._inline_text(element, 0) == ""


def test_a_cell_is_cut_at_the_budget_rather_than_read_whole() -> None:
    from xml.etree import ElementTree

    book = file_extractors._Workbook(shared=[], date_styles=frozenset(), date1904=False)
    inline = ElementTree.fromstring('<c t="inlineStr"><is><t>' + "x" * 100 + "</t></is></c>")
    number = ElementTree.fromstring("<c><v>123456</v></c>")

    assert file_extractors._cell_value(inline, book, 10) == "x" * 10
    assert file_extractors._cell_value(number, book, 3) == "123"
    assert file_extractors._cell_value(number, book, -5) == ""


def test_the_shared_string_table_stops_at_the_text_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_extractors, "_MAX_TEXT_BYTES", 3 * (file_extractors._HELD_OBJECT_BYTES + 1))
    data = _zip({"sst.xml": f"<sst {_S}>" + "<si><t>a</t></si>" * 100 + "</sst>"})

    assert file_extractors._shared_strings(file_extractors._Archive(data), "sst.xml") == ["a", "a", "a"]


def test_an_office_file_inflating_past_its_budget_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_extractors, "_MAX_ARCHIVE_XML_BYTES", 4096)
    # Kilobytes of markup that compress to almost nothing and hold no text.
    data = _docx("<w:p/>" * 10_000)

    result = file_extractors._extract_sync(data, ".docx")

    assert result.ok is False
    assert result.detail == "document inflates past the extraction limit"


def test_one_paragraph_of_too_many_elements_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_extractors, "_MAX_HELD_ELEMENTS", 1000)

    result = file_extractors._extract_sync(_docx("<w:p>" + "<w:r/>" * 2000 + "</w:p>"), ".docx")

    assert result.detail == "document inflates past the extraction limit"


def test_streaming_frees_each_paragraph_so_many_small_ones_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_extractors, "_MAX_HELD_ELEMENTS", 1000)

    result = file_extractors._extract_sync(_docx(_paragraph("x") * 2000), ".docx")

    assert result.ok is True
    assert result.text == "\n".join(["x"] * 2000)


def test_a_slide_the_presentation_lists_but_does_not_hold_is_skipped() -> None:
    data = _zip(
        {
            "ppt/presentation.xml": (
                f'<p:presentation {_P} {_R}><p:sldIdLst><p:sldId id="256" r:id="rId2"/>'
                '<p:sldId id="257" r:id="rId3"/></p:sldIdLst></p:presentation>'
            ),
            "ppt/_rels/presentation.xml.rels": _rels(
                ("rId2", "slide", "slides/missing.xml"), ("rId3", "slide", "slides/slide1.xml")
            ),
            "ppt/slides/slide1.xml": f"<p:sld {_P} {_A}><a:p><a:r><a:t>Here</a:t></a:r></a:p></p:sld>",
        }
    )

    assert file_extractors._extract_sync(data, ".pptx").text == "## Slide 1\nHere"


def test_pptx_parses_no_slide_past_the_text_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_extractors, "_MAX_TEXT_BYTES", 20)
    data = _zip(
        {
            "ppt/presentation.xml": (
                f'<p:presentation {_P} {_R}><p:sldIdLst><p:sldId id="256" r:id="rId2"/>'
                '<p:sldId id="257" r:id="rId3"/></p:sldIdLst></p:presentation>'
            ),
            "ppt/_rels/presentation.xml.rels": _rels(
                ("rId2", "slide", "slides/slide1.xml"), ("rId3", "slide", "slides/slide2.xml")
            ),
            "ppt/slides/slide1.xml": f"<p:sld {_P} {_A}><a:p><a:r><a:t>{'x' * 50}</a:t></a:r></a:p></p:sld>",
            "ppt/slides/slide2.xml": "<p:sld",
        }
    )

    result = file_extractors._extract_sync(data, ".pptx")

    assert result.detail == "truncated"
    assert result.text == "## Slide 1\n" + "x" * 9


def test_relationships_resolve_each_target_form_and_skip_external_ones() -> None:
    rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{_REL}/worksheet" Target="worksheets/sheet1.xml"/>'
        f'<Relationship Id="rId2" Type="{_REL}/styles" Target="/xl/styles.xml"/>'
        f'<Relationship Id="rId3" Type="{_REL}/extended-properties" Target="../docProps/app.xml"/>'
        f'<Relationship Id="rId4" Type="{_REL}/hyperlink" Target="https://example.com/x" TargetMode="External"/>'
        f'<Relationship Id="rId5" Type="{_REL}/image" Target=""/>'
        "</Relationships>"
    )
    archive = file_extractors._Archive(_zip({"xl/_rels/workbook.xml.rels": rels}))

    found = file_extractors._relationships(archive, "xl/workbook.xml")

    assert [(rid, member) for rid, _, member in found] == [
        ("rId1", "xl/worksheets/sheet1.xml"),
        ("rId2", "xl/styles.xml"),
        ("rId3", "docProps/app.xml"),
    ]


def test_a_file_that_is_not_the_zip_its_name_claims_fails_cleanly() -> None:
    result = file_extractors._extract_sync(b"not a zip", ".xlsx")

    assert result.ok is False
    assert result.detail == "extraction error: BadZipFile"


def test_text_past_the_ceiling_is_cut_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_extractors, "_MAX_TEXT_BYTES", 10)

    result = file_extractors._extract_sync(("é" * 20).encode(), ".txt")

    assert result.ok is True
    assert result.detail == "truncated"
    assert result.text == "é" * 5


@pytest.mark.parametrize(
    ("data", "extension", "expected"),
    [
        ("naïve,1\n".encode("cp1252"), ".csv", "naïve,1"),
        ("﻿hello".encode(), ".md", "hello"),
        ("hi".encode("utf-16"), ".txt", "hi"),
        (b"print('x')\n", ".py", "print('x')"),
    ],
)
def test_text_formats_decode(data: bytes, extension: str, expected: str) -> None:
    result = file_extractors._extract_sync(data, extension)

    assert result.ok is True
    assert result.text == expected


@pytest.mark.parametrize(("data", "extension"), [(b"\xd0\xcf\x11\xe0\x00\x00", ".doc"), (b"\xc3\x28\xa0\xa1", ".bin")])
def test_binary_with_an_unknown_extension_is_unsupported(data: bytes, extension: str) -> None:
    result = file_extractors._extract_sync(data, extension)

    assert result.ok is False
    assert result.detail == f"unsupported format {extension}"


def test_html_keeps_structure_and_drops_what_is_not_shown() -> None:
    html = b"""<html><head><title>T</title><style>p{}</style></head><body>
    <script>alert(1)</script><h2>Heading</h2><p>One
    paragraph</p><ul><li><p>Item</p></li></ul>
    <table><tr>
      <th>A</th>
      <th>B</th>
    </tr><tr><td>1</td><td>2</td></tr></table><pre>line 1
line 2</pre></body></html>"""

    result = file_extractors._extract_sync(html, ".html")

    assert result.text == "T\n## Heading\nOne paragraph\n- Item\n| A | B |\n| 1 | 2 |\nline 1\nline 2"


def test_an_uploaded_html_file_keeps_its_chrome() -> None:
    """Only a fetched page has its chrome left out; an upload is the caller's own document."""
    html = b"<body><nav>Menu</nav><main><p>Body</p></main></body>"

    assert file_extractors._extract_sync(html, ".html").text == "Menu\nBody"


_ARTICLE = "<p>" + "The article's own text. " * 20 + "</p>"


def test_a_web_page_is_its_main_without_the_chrome() -> None:
    html = (
        "<header>Site banner</header><nav><a>Home</a></nav>"
        f"<main><article><header><h1>Title</h1></header>{_ARTICLE}<footer>By someone</footer></article>"
        "<div role='navigation'><div>Breadcrumbs</div></div><p hidden>Modal</p>"
        "<span aria-hidden='true'>icon</span></main>"
        "<aside>Related</aside><footer>Copyright</footer>"
    )

    text = file_extractors.html_main_text(html)

    # An article's own header and footer are its content; the page's are chrome.
    assert text.startswith("# Title\nThe article's own text.")
    assert text.endswith("By someone")
    for chrome in ("Site banner", "Home", "Breadcrumbs", "Modal", "icon", "Related", "Copyright"):
        assert chrome not in text


def test_the_longer_of_main_and_the_articles_is_the_content() -> None:
    """A front page may keep only its featured story in <main> and the others as articles beside it."""
    html = f"<main><p>Featured</p></main><article>{_ARTICLE}</article><article>{_ARTICLE}</article>"

    text = file_extractors.html_main_text(html)

    assert "Featured" not in text
    assert text.count("The article's own text.") == 40


def test_a_main_holding_little_of_the_page_is_not_taken_for_its_content() -> None:
    html = f"<main><p>Hero</p></main><div>{_ARTICLE}</div>"

    assert file_extractors.html_main_text(html).startswith("Hero\nThe article's own text.")


def test_a_page_without_landmarks_keeps_everything_but_its_chrome() -> None:
    html = f"<div id='top'><nav>Menu</nav></div><div>{_ARTICLE}</div><div role='contentinfo'>Copyright</div>"

    text = file_extractors.html_main_text(html)

    assert text.startswith("The article's own text.")
    assert "Menu" not in text and "Copyright" not in text


def test_chrome_that_never_closes_falls_back_to_the_whole_page() -> None:
    """An unclosed <nav> would otherwise swallow everything after it."""
    html = f"<nav>Menu<div>{_ARTICLE}</div>"

    assert file_extractors.html_main_text(html).startswith("Menu\nThe article's own text.")


def _pdf(*pages: str | None) -> bytes:
    """A PDF of one page per argument, holding that text, or none for ``None``."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    pdf = pdfium.PdfDocument.new()
    for text in pages:
        page = pdf.new_page(300, 300)
        if text is None:
            continue
        obj = raw.FPDFPageObj_NewTextObj(pdf.raw, b"Helvetica", 12.0)
        encoded = (text + "\x00").encode("utf-16-le")
        buffer = ctypes.create_string_buffer(encoded, len(encoded))
        raw.FPDFText_SetText(obj, ctypes.cast(buffer, ctypes.POINTER(raw.FPDF_WCHAR)))
        raw.FPDFPageObj_Transform(obj, 1, 0, 0, 1, 20, 150)
        raw.FPDFPage_InsertObject(page.raw, obj)
        raw.FPDFPage_GenerateContent(page.raw)
    out = io.BytesIO()
    pdf.save(out)
    pdf.close()
    return out.getvalue()


def test_pdf_text_is_extracted() -> None:
    result = file_extractors._extract_sync(_pdf("Invoice total 42"), ".pdf")

    assert result.ok is True
    assert result.text == "Invoice total 42"


def test_an_uploaded_pdf_past_the_page_cap_is_read_for_its_first_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_extractors, "_MAX_PDF_PAGES", 2)

    result = file_extractors._extract_sync(_pdf("One", "Two", "Three"), ".pdf")

    assert result.text == "One\n\nTwo"
    assert result.detail == "truncated"


def test_the_web_fetch_reader_still_refuses_a_pdf_past_its_page_cap() -> None:
    result = file_extractors.extract_bounded_pdf_text_sync(_pdf("One", "Two"), max_pages=1, max_text_bytes=1024)

    assert result.ok is False


def test_a_pdf_without_text_reports_none_so_the_caller_can_rasterize() -> None:
    result = file_extractors._extract_sync(_pdf(None), ".pdf")

    assert result.ok is False
    assert result.detail == "no extractable text"


def test_pdfium_runs_under_the_module_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDFium is not thread-safe, and extractions run concurrently in worker threads."""
    held: list[bool] = []

    class FakeDocument:
        def __init__(self, _data: bytes) -> None:
            held.append(file_extractors._PDFIUM_LOCK.locked())

        def __len__(self) -> int:
            return 0

        def close(self) -> None:
            pass

    monkeypatch.setitem(__import__("sys").modules, "pypdfium2", type("m", (), {"PdfDocument": FakeDocument}))

    file_extractors._extract_sync(b"%PDF", ".pdf")
    file_extractors._rasterize_sync(b"%PDF", 1)

    assert held == [True, True]


def test_the_extension_comes_from_the_mime_type_when_there_is_no_filename() -> None:
    data = _xlsx('<row r="1"><c r="A1"><v>7</v></c></row>')
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    result = asyncio.run(file_extractors.extract_text_from_file(data, mime, None))

    assert result.text == "## Sales\n| 7 |\n| --- |"


@pytest.mark.parametrize(
    ("mime", "filename", "expected"),
    [
        ("text/plain", "Report.PDF", ".pdf"),
        ("application/pdf", "notes", ".pdf"),
        ("text/plain; charset=utf-8", None, ".txt"),
        ("image/png", None, ".png"),
        ("application/x-otari-unknown", None, ".bin"),
        ("", None, ".bin"),
    ],
)
def test_the_filename_suffix_wins_over_the_mime_type(mime: str, filename: str | None, expected: str) -> None:
    assert file_extractors._resolve_extension(mime, filename) == expected


@pytest.mark.parametrize(("max_pages", "max_text_bytes"), [(0, 1024), (1, 0), (-1, 1024)])
def test_bounded_pdf_refuses_a_limit_that_is_not_positive(max_pages: int, max_text_bytes: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        file_extractors.extract_bounded_pdf_text_sync(b"%PDF", max_pages=max_pages, max_text_bytes=max_text_bytes)


def test_bounded_pdf_rejects_page_count_before_extracting(monkeypatch: pytest.MonkeyPatch) -> None:
    indexed = False

    class FakeDocument:
        def __init__(self, _data: bytes) -> None:
            pass

        def __len__(self) -> int:
            return 101

        def __getitem__(self, _index: int) -> Any:
            nonlocal indexed
            indexed = True
            raise AssertionError("over-limit PDF must not read a page")

        def close(self) -> None:
            pass

    monkeypatch.setitem(
        __import__("sys").modules,
        "pypdfium2",
        type("m", (), {"PdfDocument": FakeDocument}),
    )

    result = file_extractors.extract_bounded_pdf_text_sync(b"%PDF", max_pages=100, max_text_bytes=1024)

    assert result.ok is False
    assert indexed is False


def test_bounded_pdf_stops_at_utf8_byte_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTextPage:
        def get_text_range(self) -> str:
            return "é" * 100

        def close(self) -> None:
            pass

    class FakePage:
        def get_textpage(self) -> FakeTextPage:
            return FakeTextPage()

        def close(self) -> None:
            pass

    class FakeDocument:
        def __init__(self, _data: bytes) -> None:
            pass

        def __len__(self) -> int:
            return 2

        def __getitem__(self, _index: int) -> FakePage:
            return FakePage()

        def close(self) -> None:
            pass

    monkeypatch.setitem(
        __import__("sys").modules,
        "pypdfium2",
        type("m", (), {"PdfDocument": FakeDocument}),
    )

    result = file_extractors.extract_bounded_pdf_text_sync(b"%PDF", max_pages=100, max_text_bytes=51)

    assert result.ok is True
    assert result.truncated is True
    assert len(result.text.encode("utf-8")) <= 51
