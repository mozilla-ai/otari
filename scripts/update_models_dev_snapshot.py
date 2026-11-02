"""Regenerate the bundled models.dev catalog: the offline prices and model facts.

Run at release: ``python scripts/update_models_dev_snapshot.py`` fetches the
models.dev v2 ``api.json``, or the v1 one when v2 fails, and ``--input FILE``
reads a saved copy (``--input-version v1`` for a v1 one). Standard library
only; the catalog's typed document lives in the gateway so the bundled catalog
is written as the gateway stores a downloaded one, and a document the gateway
would refuse is refused here too.
"""

import argparse
import http.client
import importlib.util
import json
import sys
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_MODULE = ROOT / "src" / "gateway" / "types" / "models_dev.py"
OUTPUT = ROOT / "src" / "gateway" / "data" / "models_dev_catalog.json"


def _load_document_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("models_dev", DOCUMENT_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fetch(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "otari-gateway"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.loads(response.read())


def _read(source: Any, load: Callable[[], object]) -> Any:
    """The catalog one document reads into, or ``None`` after saying on stderr why it does not."""
    name = f"models.dev {source.name} catalog"
    try:
        raw = load()
    except (OSError, ValueError, http.client.HTTPException) as exc:
        print(f"{name} fetch failed: {exc}", file=sys.stderr)
        return None
    if not isinstance(raw, dict):
        print(f"{name} refused: it is not a JSON object", file=sys.stderr)
        return None
    catalog = source.parse(raw)
    if (reason := catalog.refusal()) is not None:
        print(f"{name} refused: {reason}", file=sys.stderr)
        return None
    if catalog.issues.count:
        print(f"{name} read with {catalog.issues.describe()}", file=sys.stderr)
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="a saved api.json instead of fetching")
    parser.add_argument("--input-version", choices=("v2", "v1"), default="v2", help="which document --input holds")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--date", help="generated_at (YYYY-MM-DD); defaults to today")
    args = parser.parse_args()

    module = _load_document_module()
    if args.input:
        (source,) = (source for source in module.MODELS_DEV_SOURCES if source.name == args.input_version)
        attempts = [(source, lambda: json.loads(args.input.read_text("utf-8")))]
    else:
        attempts = [(source, partial(_fetch, source.url)) for source in module.MODELS_DEV_SOURCES]
    for source, load in attempts:
        if (catalog := _read(source, load)) is not None:
            break
    else:
        return 1

    generated_at = args.date or datetime.now(UTC).date().isoformat()
    document = module.catalog_document(catalog, source=source.url, generated_at=generated_at)
    models = sum(1 for _ in catalog.models())
    args.output.write_text(json.dumps(document, separators=(",", ":"), ensure_ascii=False) + "\n", "utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes, {models} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
