#!/usr/bin/env python3
"""Apply local eval artifacts to a weighted routing policy."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gateway.core.config import API_KEY_HEADER  # noqa: E402
from gateway.services.eval_score_pipeline import (
    EvalScorePipelineError,
    build_eval_scores_payload,
    load_eval_score_rows,
)  # noqa: E402


def _post_eval_scores(
    *,
    gateway_url: str,
    policy_id: str,
    master_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[int, dict[str, Any]]:
    encoded_policy_id = urllib.parse.quote(policy_id, safe="")
    url = f"{gateway_url.rstrip('/')}/v1/routing-policies/{encoded_policy_id}/eval-scores"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            API_KEY_HEADER: f"Bearer {master_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        body = response.read().decode("utf-8")
        parsed = json.loads(body) if body else {}
        if not isinstance(parsed, dict):
            raise EvalScorePipelineError("Gateway returned a non-object JSON response")
        return response.status, parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize JSON, JSONL, or CSV eval artifacts and apply them to a weighted routing policy.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to JSON, JSONL/NDJSON, or CSV eval scores")
    parser.add_argument("--policy-id", required=True, help="Routing policy id to update")
    parser.add_argument("--gateway-url", default="http://localhost:8000", help="Gateway base URL")
    parser.add_argument(
        "--master-key",
        default=os.getenv("GATEWAY_MASTER_KEY") or os.getenv("OTARI_MASTER_KEY"),
        help="Gateway master key. Defaults to GATEWAY_MASTER_KEY or OTARI_MASTER_KEY.",
    )
    parser.add_argument("--metric", help="Default metric name for rows that do not include one")
    parser.add_argument("--change-note", help="Revision note for the routing policy update")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="Gateway request timeout")
    parser.add_argument("--dry-run", action="store_true", help="Print the normalized payload without posting")
    args = parser.parse_args()

    try:
        rows = load_eval_score_rows(args.input)
        payload = build_eval_scores_payload(
            rows,
            default_metric=args.metric,
            change_note=args.change_note,
            source=str(args.input),
        )
    except (OSError, json.JSONDecodeError, EvalScorePipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not args.master_key:
        print(
            "Error: --master-key or GATEWAY_MASTER_KEY/OTARI_MASTER_KEY is required unless --dry-run is set",
            file=sys.stderr,
        )
        return 1

    try:
        status, response = _post_eval_scores(
            gateway_url=args.gateway_url,
            policy_id=args.policy_id,
            master_key=args.master_key,
            payload=payload,
            timeout_seconds=args.timeout_seconds,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        print(f"Gateway rejected eval score import with HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, EvalScorePipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(response, indent=2, sort_keys=True))
    return 0 if 200 <= status < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
