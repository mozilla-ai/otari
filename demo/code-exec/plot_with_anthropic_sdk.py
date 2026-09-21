#!/usr/bin/env python3
"""Anthropic's own SDK, pointed at Otari, asks for a bar plot and saves it.

The point is the model id. Nothing else in the script changes between:

    uv run python demo/code-exec/plot_with_anthropic_sdk.py anthropic:claude-sonnet-4-6
    uv run python demo/code-exec/plot_with_anthropic_sdk.py nebius:openai/gpt-oss-120b

The first runs on Anthropic's own sandbox, because the `auto` executor leaves a
declaration with a provider that serves it natively. The second has no native
sandbox, so Otari runs the code. Either way the reply carries the same
`code_execution_tool_result` blocks, and the chart downloads from Otari's files
API with the same call: a file Otari's sandbox produced it stored, and one
Anthropic's produced it streams back from Anthropic under Anthropic's own id.

Environment:
    OTARI_URL   default http://localhost:8000
    OTARI_KEY   the gateway's master key or an API key (default demo-master-key)
    QA_USER     the user a master-key request bills (default plot-roundtrip)
    QA_OUT_DIR  where to write the plots (default demo/code-exec/plots)
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

from anthropic import Anthropic

OTARI_URL = os.environ.get("OTARI_URL", "http://localhost:8000").rstrip("/")
OTARI_KEY = os.environ.get("OTARI_KEY", "demo-master-key")
USER = os.environ.get("QA_USER", "plot-roundtrip")
OUT_DIR = pathlib.Path(os.environ.get("QA_OUT_DIR", pathlib.Path(__file__).parent / "plots"))

PROMPT = (
    "Use the code execution tool to draw a bar chart of these sales figures with "
    "matplotlib: Jan 120, Feb 95, Mar 160, Apr 210, May 175. Save it as 'bar_plot.png' "
    "in the working directory, then reply with the file id of the png you saved."
)
CODE_TOOL: Any = {"type": "code_execution_20250825", "name": "code_execution"}

# The SDK appends /v1 to its base URL, and Otari serves its API under /api/v1.
# `default_query` is only for a master key: the files routes want to be told
# whose files to read, which an API key answers by itself.
otari = Anthropic(base_url=f"{OTARI_URL}/api", api_key=OTARI_KEY, default_query={"user": USER})


def produced_file_ids(message: Any) -> list[str]:
    """Every file id the run's tool results announce, in the order they appear.

    The python and the bash variants of the tool each name their outputs in
    their own block type, and Otari answers in whichever the run used, so this
    looks for the shape rather than for one block name.
    """
    ids = []
    for block in message.content:
        result = getattr(block, "content", None)
        for output in getattr(result, "content", None) or []:
            file_id = getattr(output, "file_id", None)
            if file_id and file_id not in ids:
                ids.append(file_id)
    return ids


def download(file_id: str, model: str) -> pathlib.Path:
    """Fetch a produced file from Otari, wherever its bytes actually live."""
    meta = otari.beta.files.retrieve_metadata(file_id)
    body = otari.beta.files.download(file_id)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{model.replace(':', '-').replace('/', '-')}-{meta.filename}"
    body.write_to_file(path)
    return path


def main(model: str) -> int:
    # No `betas`: they are Anthropic's own, and a provider with no Messages API
    # of its own cannot honor one, so Otari drops them rather than refusing.
    message = otari.beta.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": PROMPT}],
        tools=[CODE_TOOL],
        # A master-key request must say who it bills; an API key ignores this
        # in favor of its own user.
        metadata={"user_id": USER},
    )

    ran_natively = message.container is not None
    print(f"{model}: code ran on {'the provider' if ran_natively else "otari's sandbox"}")

    file_ids = produced_file_ids(message)
    if not file_ids:
        print("no file id in the tool results:")
        print(message.model_dump_json(indent=2)[:2000])
        return 1

    for file_id in file_ids:
        path = download(file_id, model)
        print(f"  {file_id} -> {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "anthropic:claude-sonnet-4-6"))
