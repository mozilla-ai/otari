#!/usr/bin/env python3
"""OpenAI's own SDK, pointed at Otari, asks for a bar plot and saves it.

The counterpart to `plot_with_anthropic_sdk.py`, on the Responses API:

    uv run python demo/code-exec/plot_with_openai_sdk.py openai:gpt-4o-mini
    QA_EXECUTOR=otari uv run python demo/code-exec/plot_with_openai_sdk.py openai:gpt-4o-mini

The first runs `code_interpreter` on OpenAI's own container, because the `auto`
executor leaves a declaration with a provider that serves it natively. The
second sends `X-Otari-Code-Execution: otari`, so the same request runs on
Otari's sandbox instead. Both come back as a `code_interpreter_call` item, and
either way the chart downloads from Otari's files API: one Otari's sandbox
produced it stored, and one OpenAI's produced it streams back from OpenAI's
container under OpenAI's own id.

Swapping in an open model works the same way, but only for a provider whose
Responses API Otari supports (groq, fireworks, openrouter, huggingface, gmi).
Nebius is not one of those: `nebius:...` on this endpoint is refused with
"Provider 'nebius' does not support the Responses API", so use the Anthropic
script, or Chat Completions, for a nebius model.

Environment:
    OTARI_URL        default http://localhost:8000
    OTARI_KEY        the gateway's master key or an API key (default demo-master-key)
    QA_EXECUTOR      `auto` (default), `otari` or `provider`, sent as X-Otari-Code-Execution
    QA_USER          the user a master-key request bills (default plot-roundtrip)
    QA_OUT_DIR       where to write the plots (default demo/code-exec/plots)
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

from openai import OpenAI

OTARI_URL = os.environ.get("OTARI_URL", "http://localhost:8000").rstrip("/")
OTARI_KEY = os.environ.get("OTARI_KEY", "demo-master-key")
EXECUTOR = os.environ.get("QA_EXECUTOR", "auto")
USER = os.environ.get("QA_USER", "plot-roundtrip")
OUT_DIR = pathlib.Path(os.environ.get("QA_OUT_DIR", pathlib.Path(__file__).parent / "plots"))

PROMPT = (
    "Use the code interpreter to draw a bar chart of these sales figures with matplotlib: "
    "Jan 120, Feb 95, Mar 160, Apr 210, May 175. Save it as 'bar_plot.png' in the working "
    "directory, then reply with the file id of the png you saved."
)
CODE_TOOL: Any = {"type": "code_interpreter", "container": {"type": "auto"}}

# Otari serves its API under /api/v1, which is the whole base URL for this SDK.
# `default_query` is only for a master key: the files routes want to be told
# whose files to read, which an API key answers by itself.
otari = OpenAI(base_url=f"{OTARI_URL}/api/v1", api_key=OTARI_KEY, default_query={"user": USER})


def container_citations(response: Any) -> list[tuple[str, str]]:
    """`(file_id, filename)` for each file the provider's container cited.

    OpenAI announces a produced file as a `container_file_citation` annotation
    on the message it wrote. Otari records those ids as it answers, so the
    download below reaches them through the gateway like any other file.
    """
    return [
        (note.file_id, note.filename)
        for item in response.output
        for part in getattr(item, "content", None) or []
        for note in getattr(part, "annotations", None) or []
        if getattr(note, "type", "") == "container_file_citation"
    ]


def produced_images(response: Any) -> list[str]:
    """The file ids in the `image` outputs of a gateway-run `code_interpreter_call`.

    Otari announces a produced image as the URL it serves the file from, which
    is the only shape the Responses API has for one, so the id is the last path
    segment before `/content`.
    """
    return [
        output.url.rsplit("/", 2)[-2]
        for item in response.output
        if item.type == "code_interpreter_call"
        for output in item.outputs or []
        if getattr(output, "type", "") == "image"
    ]


def save(file_id: str, name: str, model: str) -> pathlib.Path:
    body = otari.files.content(file_id).read()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{model.replace(':', '-').replace('/', '-')}-{name}"
    path.write_bytes(body)
    return path


def main(model: str) -> int:
    response = otari.responses.create(
        model=model,
        input=PROMPT,
        tools=[CODE_TOOL],
        # A master-key request must say who it bills; an API key ignores this
        # in favor of its own user.
        user=USER,
        extra_headers={"X-Otari-Code-Execution": EXECUTOR},
    )

    # Otari's own ids mark a gateway-run execution; the provider's are its own.
    calls = [item for item in response.output if item.type == "code_interpreter_call"]
    ran_natively = bool(calls) and not calls[0].id.startswith("otari_ci_")
    print(f"{model}: code ran on {'the provider' if ran_natively else "otari's sandbox"} (executor {EXECUTOR})")

    produced = container_citations(response) or [
        (file_id, otari.files.retrieve(file_id).filename) for file_id in produced_images(response)
    ]
    if not produced:
        print(f"nothing produced: {response.model_dump_json(indent=2)[:2000]}")
        return 1

    for file_id, name in produced:
        path = save(file_id, name, model)
        print(f"  {file_id} -> {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "openai:gpt-4o-mini"))
