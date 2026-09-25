"""Gate on the header naming rule in AGENTS.md: Otari's own headers carry no ``X-`` prefix.

RFC 6648 (BCP 178) retired that convention. Nothing else reaches the rule: it is
prose, no lint rule covers it, and no other test names a header it does not
itself send, so a reintroduced prefix would ship silently.

Every ``X-`` literal under ``src/gateway`` is classified here rather than only the
ones spelled ``X-Otari-``, because a scan for that spelling alone passes a header
of ours named anything else, which is the wider gap. The scan stops at
``src/gateway`` because that is where a header reaches the wire.

Both buckets only shrink. An entry whose literal has left the source fails until
it leaves the bucket too, so a retired name cannot sit here pre-authorizing a
future one.
"""

from __future__ import annotations

import re
from pathlib import Path

GATEWAY_ROOT = Path(__file__).resolve().parents[2] / "src" / "gateway"
X_LITERAL = re.compile(r"""["']([Xx]-[A-Za-z0-9-]+)["']""")

# Header names Otari sends or reads but did not define, so it cannot rename them
# alone. Each reason says what owns the name.
INHERITED_HEADERS = {
    "x-api-key": "Anthropic's credential header, which its SDKs send.",
    "X-Content-Type-Options": "Browser security header, honored under this name only.",
    "X-Frame-Options": "Browser security header, honored under this name only.",
    "X-Forwarded-Proto": "Proxy convention, read and never sent.",
    "X-RateLimit-Limit": "Rate-limit convention clients already parse.",
    "X-RateLimit-Remaining": "Rate-limit convention clients already parse.",
    "X-RateLimit-Reset": "Rate-limit convention clients already parse.",
    "X-Subscription-Token": "The Brave Search API's own header.",
    "X-Gateway-Token": "Platform wire contract; the platform must accept a new name first.",
    "X-User-Token": "Platform wire contract; the platform must accept a new name first.",
}

# Literals of the same shape that name no header at all.
NOT_HEADERS = {
    "x-ai": "The xAI provider id.",
    "x-gzip": "A Content-Encoding token.",
}

CLASSIFIED = INHERITED_HEADERS | NOT_HEADERS


def _x_prefixed_literals() -> dict[str, str]:
    """Each distinct ``X-`` literal under the gateway, mapped to where it first appears."""
    found: dict[str, str] = {}
    for path in sorted(GATEWAY_ROOT.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for name in X_LITERAL.findall(line):
                found.setdefault(name, f"{path.relative_to(GATEWAY_ROOT)}:{number}")
    return found


def test_every_x_prefixed_literal_is_classified() -> None:
    unclassified = {name: where for name, where in _x_prefixed_literals().items() if name not in CLASSIFIED}
    assert not unclassified, (
        "An X- prefixed literal must either be a header Otari defines, and then be renamed to "
        "Otari-* per RFC 6648 and the rule in AGENTS.md, or be classified in this test with the "
        f"reason it keeps the prefix. Unclassified: {unclassified}"
    )


def test_no_classification_outlives_its_literal() -> None:
    found = _x_prefixed_literals()
    stale = sorted(name for name in CLASSIFIED if name not in found)
    assert not stale, f"No longer present under src/gateway, so drop from this test: {stale}"
