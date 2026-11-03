"""The typed models.dev catalog stored as JSON and read back."""

import json
from pathlib import Path
from typing import Any

import pytest

from gateway.types.models_dev import (
    CATALOG_FORMAT,
    ModelsDevCatalog,
    catalog_document,
    parse_catalog,
    parse_catalog_v2,
    read_catalog_document,
    snapshot_document,
)

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _load(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((FIXTURES / name).read_text("utf-8"))
    return document


def _stored(catalog: ModelsDevCatalog, **meta: object) -> dict[str, Any]:
    """The catalog as it is written to a row or a file, and read back from text."""
    document: dict[str, Any] = json.loads(json.dumps(catalog_document(catalog, **meta)))
    return document


@pytest.mark.parametrize(
    ("fixture", "parse"),
    [("models_dev_v2_sample.json", parse_catalog_v2), ("models_dev_v1_sample.json", parse_catalog)],
)
def test_a_stored_catalog_reads_back_to_the_same_catalog(fixture: str, parse: Any) -> None:
    catalog = parse(_load(fixture))

    assert read_catalog_document(_stored(catalog)) == catalog


def test_a_stored_catalog_keeps_what_the_download_could_not_read() -> None:
    raw = _load("models_dev_v2_sample.json")
    raw["openai"]["models"]["gpt-4o"]["cost"]["text"]["input"]["price"] = "2.5"
    catalog = parse_catalog_v2(raw)
    assert catalog.issues.count

    assert read_catalog_document(_stored(catalog)) == catalog


def test_a_stored_catalog_keeps_the_units_it_does_not_know() -> None:
    catalog = parse_catalog_v2({"p": {"models": {"m": {"cost": {"text": {"input": {"price": 1, "unit": "parsec"}}}}}}})
    assert catalog.issues.unknown_units == ("parsec",)

    document = _stored(catalog)

    assert document["issues"] == {"models": 1, "unknown_units": ["parsec"]}
    assert read_catalog_document(document) == catalog


def test_a_stored_catalog_carries_its_marker_and_meta_and_leaves_out_defaults() -> None:
    catalog = parse_catalog_v2({"p": {"models": {"m": {}}}})

    document = _stored(catalog, source="https://models.dev/experimental/v2.0/api.json?type=all")

    assert document["_meta"] == {
        "format": CATALOG_FORMAT,
        "source": "https://models.dev/experimental/v2.0/api.json?type=all",
    }
    assert document["providers"] == {"p": {"id": "p", "name": None, "doc": None, "models": {"m": {"id": "m"}}}}


def test_a_stored_v1_model_keeps_its_missing_type() -> None:
    """The default type is ``chat``, so a v1 model's ``None`` is written out and read back."""
    document = _stored(parse_catalog({"p": {"models": {"m": {}}}}))

    assert document["providers"]["p"]["models"]["m"] == {"id": "m", "type": None}
    assert read_catalog_document(document).model_for("p", "m").type is None  # type: ignore[union-attr]


def test_a_document_without_the_marker_is_an_older_price_snapshot() -> None:
    catalog = parse_catalog_v2(_load("models_dev_v2_sample.json"))
    snapshot = json.loads(json.dumps(snapshot_document(catalog)))

    assert read_catalog_document(snapshot) == parse_catalog(snapshot)


def test_a_bundled_price_snapshot_with_its_meta_reads_without_the_meta() -> None:
    snapshot = {"_meta": {"generated_at": "2026-11-02T00:00:00"}, "p": {"models": {"m": {"cost": {"input": 1}}}}}

    catalog = read_catalog_document(snapshot)

    assert set(catalog.providers) == {"p"}


@pytest.mark.parametrize(
    "document",
    [
        {"_meta": {"format": CATALOG_FORMAT + 1}, "providers": {}},
        {"_meta": {"format": CATALOG_FORMAT}, "providers": {"p": {"id": "p", "models": {"m": {"id": "m", "x": 1}}}}},
        {"_meta": {"format": CATALOG_FORMAT}, "providers": {"p": {"id": "p", "models": []}}},
        {"_meta": {"format": CATALOG_FORMAT}},
        [],
    ],
    ids=["a newer format", "an unknown field", "a damaged provider", "no providers", "not an object"],
)
def test_a_stored_catalog_this_gateway_cannot_read_is_refused(document: object) -> None:
    with pytest.raises(ValueError, match="stored models.dev catalog"):
        read_catalog_document(document)
