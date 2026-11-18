"""Drift gate between the gateway's maker vocabulary and the dashboard's marks.

A vendor written into ``_ORG_VENDORS`` or ``_KEY_VENDORS`` in
``services/model_identity.py`` is one somebody added in this repository. The
dashboard draws each maker's mark from a table keyed on ``vendor_slug`` of that
string (``web/src/shared/helpers/brandMarks.ts``).

The two live in different languages and neither imports the other, so a vendor
added on this side reaches the dashboard as a lettermark tile with nothing
failing. The dashboard's own suite cannot catch it either: a TypeScript test can
only compare the table against a list somebody typed there by hand, which drifts
the moment this vocabulary grows. So the assertion belongs here, where the
vocabulary is defined and where the change that breaks it is made.

A vendor can also be any org a models.dev canonical id names (``baai``,
``audio8``), and the table keys those as the hosted catalog shows them. They are
not gated: they arrive with data rather than a change here, and some are a
person's account with no mark to draw.

Providers are not gated either. A provider is any id any-llm supports, so a new
one arrives with an any-llm bump, often with no published mark. The dashboard
tiles it until someone draws one, and a dependency bump should not fail over a
missing logo.
"""

from __future__ import annotations

import re
from pathlib import Path

from gateway.services.model_identity import _KEY_VENDORS, _ORG_VENDORS, vendor_slug

MARKS = Path(__file__).resolve().parents[2] / "web" / "src" / "shared" / "helpers" / "brandMarks.ts"

# Makers deliberately drawn as a lettermark tile instead, by reason.
TILED = {
    # Only a wordmark or a product's mark exists, and the row names the company.
    # Amazon has the aws wordmark and the Nova and Bedrock product marks;
    # Meituan's is its name in Chinese; inclusionAI's is the Ling family's.
    "amazon",
    "inclusionai",
    "meituan",
    # Drawn through a gradient mask the shape table cannot hold.
    "poolside",
    # A mark exists, but the hosted catalog does not list the vendor yet.
    "arceeai",
    "sakanaai",
    "upstage",
    # No vector mark in LobeHub, Simple Icons or on the vendor's own site.
    "aisingapore",
    "mixedbread",
    "sarvamai",
    "sdaia",
    "swissai",
    "thinkingmachines",
    "trendyol",
    "writer",
}


def _table_keys(table: str) -> set[str]:
    """The keys of one mark table, read off the module rather than imported."""
    source = MARKS.read_text(encoding="utf-8")
    start = source.index(f"const {table}")
    body = source[start : source.index("\n}\n", start)]
    return set(re.findall(r'^\s+"?([A-Za-z0-9_-]+)"?:', body, re.M))


def _maker_keys() -> set[str]:
    return _table_keys("MAKER_MARKS")


def _vendor_slugs() -> set[str]:
    vendors = set(_ORG_VENDORS.values()) | {vendor for _, vendor in _KEY_VENDORS}
    return {vendor_slug(vendor) for vendor in vendors}


def test_every_maker_is_marked_or_deliberately_tiled() -> None:
    missing = _vendor_slugs() - _maker_keys() - TILED
    assert not missing, (
        f"{sorted(missing)} can be a model's vendor but has no mark in {MARKS.name} "
        "and is not in TILED. Add the mark, or add it to TILED with the reason."
    )


def test_no_exemption_outlives_its_reason() -> None:
    # The other direction, which is what keeps TILED honest: an exemption for a
    # vendor nobody can be assigned any more, or one that has since been given a
    # mark, is dead weight that reads as a gap.
    stale = TILED - _vendor_slugs()
    marked = TILED & _maker_keys()
    assert not stale, f"{sorted(stale)} is exempted here but is no longer a vendor"
    assert not marked, f"{sorted(marked)} has a mark in {MARKS.name} and is also in TILED"
