"""Structural checks for the dashboard operator guide (docs/run-the-dashboard.md)."""

from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
_OPERATOR_DOC = _DOCS_DIR / "run-the-dashboard.md"


def test_operator_doc_explains_the_two_keys() -> None:
    # The operator page keeps the sign-in key and the encryption key distinct.
    # Both env var names must appear so it stays accurate if either is renamed
    # in code.
    operator_text = _OPERATOR_DOC.read_text(encoding="utf-8")
    for heading in ("## Sign-in and secrets", "## First-run walkthrough"):
        assert heading in operator_text, f"missing required section: {heading!r}"
    for token in ("OTARI_MASTER_KEY", "OTARI_SECRET_KEY", "master key", "gen-secret-key"):
        assert token in operator_text, f"two-key model text should mention {token!r}"
