"""Regenerate the bundled models.dev price snapshot.

Run at release: ``python scripts/update_models_dev_snapshot.py`` fetches the
mozilla-ai fork of models.dev, ``--url`` fetches another copy serving the same
api.json (upstream is https://models.dev/api.json), and ``--input FILE`` trims a
saved copy. The v2 catalog, which carries the per-image prices of image models,
is fetched from ``--v2-url`` or read from ``--v2-input FILE`` the same way.
Standard library only; the trimming rules live in the gateway so a stored
snapshot and the bundled one are cut the same way.
"""

import argparse
import importlib.util
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRIM_MODULE = ROOT / "src" / "gateway" / "services" / "pricing" / "catalog_trim.py"
OUTPUT = ROOT / "src" / "gateway" / "data" / "models_dev_pricing.json"
URL = "https://mozilla-ai.github.io/models.dev/api.json"
V2_URL = "https://mozilla-ai.github.io/models.dev/experimental_v2.0_api.json"


def _load_trim() -> ModuleType:
    spec = importlib.util.spec_from_file_location("catalog_trim", TRIM_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path | None, url: str) -> Any:
    if path:
        return json.loads(path.read_text("utf-8"))
    request = urllib.request.Request(url, headers={"User-Agent": "otari-gateway"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL, help=f"where to fetch api.json from (default: {URL})")
    parser.add_argument("--input", type=Path, help="a saved api.json instead of fetching")
    parser.add_argument("--v2-url", default=V2_URL, help=f"where to fetch the v2 api.json from (default: {V2_URL})")
    parser.add_argument("--v2-input", type=Path, help="a saved v2 api.json instead of fetching")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--date", help="generated_at (YYYY-MM-DD); defaults to today")
    args = parser.parse_args()

    raw = _read(args.input, args.url)
    v2 = _read(args.v2_input, args.v2_url)
    if not isinstance(raw, dict) or not isinstance(v2, dict):
        print("models.dev did not return a JSON object", file=sys.stderr)
        return 1

    trim = _load_trim()
    meta = {
        "source": args.url,
        "v2_source": args.v2_url,
        "generated_at": args.date or datetime.now(UTC).date().isoformat(),
    }
    document: dict[str, Any] = {"_meta": meta}
    document.update(trim.trim_catalog(trim.merge_image_prices(raw, v2)))
    models = sum(len(provider["models"]) for key, provider in document.items() if key != "_meta")
    args.output.write_text(json.dumps(document, separators=(",", ":"), ensure_ascii=False) + "\n", "utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes, {models} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
