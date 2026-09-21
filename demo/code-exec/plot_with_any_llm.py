#!/usr/bin/env python3
"""any-llm, pointed at Otari, asks for a bar plot and saves it.

The third way to write the other two scripts. any-llm talks to Otari through
its `otari` provider, so one call shape reaches every model Otari serves and
the code-execution tool is declared in Otari's own vocabulary rather than a
provider's:

    uv run python demo/code-exec/plot_with_any_llm.py nebius:openai/gpt-oss-120b
    uv run python demo/code-exec/plot_with_any_llm.py anthropic:claude-sonnet-4-6

`otari_code_execution` always runs on Otari's sandbox, whatever the model and
whatever the executor says, which is the trade for one vocabulary: an
`anthropic:` model here runs the code here rather than natively. Ask for the
provider's own sandbox with that provider's declaration instead, which is what
`plot_with_anthropic_sdk.py` does.

any-llm covers the inference half only. It has no files API, so the chart is
downloaded with a plain HTTP GET against the same gateway.

Environment:
    OTARI_URL   default http://localhost:8000
    OTARI_KEY   the gateway's master key or an API key (default demo-master-key)
    QA_USER     the user a master-key request bills (default plot-roundtrip)
    QA_OUT_DIR  where to write the plots (default demo/code-exec/plots)
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import re
import sys
from typing import Any

import httpx
from any_llm import acompletion

OTARI_URL = os.environ.get("OTARI_URL", "http://localhost:8000").rstrip("/")
OTARI_KEY = os.environ.get("OTARI_KEY", "demo-master-key")
USER = os.environ.get("QA_USER", "plot-roundtrip")
OUT_DIR = pathlib.Path(os.environ.get("QA_OUT_DIR", pathlib.Path(__file__).parent / "plots"))

PROMPT = (
    "Use the code execution tool to draw a bar chart of these sales figures with "
    "matplotlib: Jan 120, Feb 95, Mar 160, Apr 210, May 175. Save it as 'bar_plot.png' "
    "in the working directory, then reply with the file id of the png you saved."
)
# Otari's own declaration, the one shape that means the same thing to every model.
CODE_TOOL: Any = {"type": "otari_code_execution"}
# A stored file's id as the tool result gives it to the model.
FILE_ID = re.compile(r"file-[0-9a-f]{32}")

files = httpx.Client(base_url=f"{OTARI_URL}/api/v1", headers={"Otari-Key": OTARI_KEY}, timeout=60.0)


def save(file_id: str, model: str) -> pathlib.Path:
    """Download one stored file and write it beside the other plots."""
    meta = files.get(f"/files/{file_id}", params={"user": USER})
    meta.raise_for_status()
    body = files.get(f"/files/{file_id}/content", params={"user": USER})
    body.raise_for_status()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{model.replace(':', '-').replace('/', '-')}-{meta.json()['filename']}"
    path.write_bytes(body.content)
    return path


async def run(model: str) -> int:
    response = await acompletion(
        # The gateway is the provider; the selector it routes on travels as the model.
        provider="otari",
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        tools=[CODE_TOOL],
        api_base=f"{OTARI_URL}/api/v1",
        api_key=OTARI_KEY,
        # A master-key request must say who it bills; an API key ignores this
        # in favor of its own user.
        user=USER,
    )

    # Chat Completions has no shape for a server-run tool, so the loop resolves
    # inside the gateway and only the final message comes back. The model was
    # given each produced file as `name (file_id: file-...)` and asked to pass
    # the id on, which is what is read back here.
    reply = response.choices[0].message.content or ""
    file_ids = list(dict.fromkeys(FILE_ID.findall(reply)))
    if not file_ids:
        print(f"{model}: no stored file id in the reply: {reply[:300]}")
        return 1

    print(f"{model}: code ran on otari's sandbox")
    for file_id in file_ids:
        path = save(file_id, model)
        print(f"  {file_id} -> {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else "nebius:openai/gpt-oss-120b")))
