"""Drift gate between the gateway's maker vocabulary and the dashboard's marks.

A model's vendor is voted from ``_ORG_VENDORS`` and ``_KEY_VENDORS`` in
``services/model_identity.py`` and from nowhere else, so that pair is the whole
maker vocabulary. The dashboard draws each maker's mark from a table keyed on
``vendor_slug`` of that string (``web/src/shared/helpers/brandMarks.ts``).

The two live in different languages and neither imports the other, so a vendor
added on this side reaches the dashboard as a lettermark tile with nothing
failing. The dashboard's own suite cannot catch it either: a TypeScript test can
only compare the table against a list somebody typed there by hand, which drifts
the moment this vocabulary grows. So the assertion belongs here, where the
vocabulary is defined and where the change that breaks it is made.

Total both ways: every vendor is classified, and every key still names a vendor
that exists.
"""

from __future__ import annotations

import re
from pathlib import Path

from gateway.services.model_identity import _KEY_VENDORS, _ORG_VENDORS, vendor_slug

MARKS = Path(__file__).resolve().parents[2] / "web" / "src" / "shared" / "helpers" / "brandMarks.ts"

# Makers deliberately drawn as a lettermark tile instead, with the reason.
TILED = {
    # The only Amazon asset in the source set is the lowercase "aws" wordmark,
    # which is not legible at the 14px and 16px steps a mark is drawn at. The
    # `bedrock` and `sagemaker` provider rows are left out for the same reason,
    # so a card and a rail row agree about Amazon.
    "amazon",
    # No mark for it in either source set.
    "openbmb",
}


def _maker_keys() -> set[str]:
    """The keys of ``MAKER_MARKS``, read off the module rather than imported."""
    source = MARKS.read_text(encoding="utf-8")
    start = source.index("const MAKER_MARKS")
    body = source[start : source.index("\n}\n", start)]
    return set(re.findall(r'^\s+"?([A-Za-z0-9_-]+)"?:', body, re.M))


def _vendor_slugs() -> set[str]:
    vendors = set(_ORG_VENDORS.values()) | {vendor for _, vendor in _KEY_VENDORS}
    return {vendor_slug(vendor) for vendor in vendors}


def test_every_maker_is_marked_or_deliberately_tiled() -> None:
    missing = _vendor_slugs() - _maker_keys() - TILED
    assert not missing, (
        f"{sorted(missing)} can be a model's vendor but has no mark in {MARKS.name} "
        "and is not in TILED. Add the mark, or add it to TILED with the reason."
    )


def test_no_mark_names_a_vendor_that_is_gone() -> None:
    # The other direction, which is what keeps TILED honest as the vocabulary
    # changes: a key or an exemption for a vendor nobody can be assigned any
    # more is dead weight that reads as coverage.
    slugs = _vendor_slugs()
    stale_keys = _maker_keys() - slugs
    stale_tiles = TILED - slugs
    assert not stale_keys, f"{sorted(stale_keys)} is keyed in {MARKS.name} but is no longer a vendor"
    assert not stale_tiles, f"{sorted(stale_tiles)} is exempted here but is no longer a vendor"
