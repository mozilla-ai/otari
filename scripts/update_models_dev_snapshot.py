"""Regenerate the bundled models.dev price snapshot.

Run at release: ``python scripts/update_models_dev_snapshot.py`` fetches
https://models.dev/api.json, or ``--input FILE`` trims a saved copy. Standard
library only; the catalog's typed document lives in the gateway so a stored
snapshot and the bundled one are cut the same way.
"""

import argparse
import importlib.util
import json
import sys
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_MODULE = ROOT / "src" / "gateway" / "types" / "models_dev.py"
OUTPUT = ROOT / "src" / "gateway" / "data" / "models_dev_pricing.json"
URL = "https://models.dev/api.json"


def _load_trim() -> Callable[[dict[str, Any]], dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("models_dev", DOCUMENT_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def trim(raw: dict[str, Any]) -> dict[str, Any]:
        document: dict[str, Any] = module.snapshot_document(module.parse_catalog(raw))
        return document

    return trim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="a saved api.json instead of fetching")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--date", help="generated_at (YYYY-MM-DD); defaults to today")
    args = parser.parse_args()

    if args.input:
        raw = json.loads(args.input.read_text("utf-8"))
    else:
        request = urllib.request.Request(URL, headers={"User-Agent": "otari-gateway"})
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            raw = json.loads(response.read())
    if not isinstance(raw, dict):
        print("models.dev did not return a JSON object", file=sys.stderr)
        return 1

    trim = _load_trim()
    document = {"_meta": {"source": URL, "generated_at": args.date or datetime.now(UTC).date().isoformat()}}
    document.update(trim(raw))
    models = sum(len(provider["models"]) for key, provider in document.items() if key != "_meta")
    args.output.write_text(json.dumps(document, separators=(",", ":"), ensure_ascii=False) + "\n", "utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes, {models} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
