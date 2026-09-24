"""Gate on the header naming rule in AGENTS.md: Otari's own headers carry no ``X-`` prefix.

RFC 6648 (BCP 178) retired that convention. Nothing else reaches the rule: it is
prose, no lint rule covers it, and no other test names a header it does not
itself send, so a reintroduced prefix would ship silently.

The scan covers ``src/gateway`` alone, because that is where a header reaches
the wire. It matches case-sensitively, which is what keeps an OpenAPI
specification extension such as ``x-otari-optional`` out of scope: that spec
requires the ``x-`` prefix, and lowercase is how it is written.
"""

from __future__ import annotations

from pathlib import Path

GATEWAY_ROOT = Path(__file__).resolve().parents[2] / "src" / "gateway"
FORBIDDEN_PREFIX = "X-Otari-"


def test_no_gateway_source_names_an_x_prefixed_otari_header() -> None:
    offenders = [
        f"{path.relative_to(GATEWAY_ROOT)}:{number}"
        for path in sorted(GATEWAY_ROOT.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FORBIDDEN_PREFIX in line
    ]
    assert not offenders, (
        f"Otari's own headers are named Otari-*, never {FORBIDDEN_PREFIX}*. See RFC 6648 "
        f"and the rule in AGENTS.md. Found: {', '.join(offenders)}"
    )
