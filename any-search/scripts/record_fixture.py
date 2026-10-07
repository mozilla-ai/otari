"""Record a provider's live answer as a test fixture, with the API key redacted.

    uv run python any-search/scripts/record_fixture.py exa normal "latest stable python release" -o type=auto

Runs one search against the live API, with the key from the environment
variable the provider's metadata names, and writes the last HTTP response to
``tests/fixtures/<provider>/<case>.json`` as ``{"status": ..., "body": ...}``.
A provider's unit tests replay it through ``httpx.MockTransport``. Every
occurrence of the key is replaced before anything is written.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from any_search import AnySearch, AnySearchError

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
REDACTED = "<redacted>"


class _Recorder:
    """A response hook that keeps the last response the provider received."""

    def __init__(self) -> None:
        self.last: tuple[int, bytes] | None = None

    async def __call__(self, response: httpx.Response) -> None:
        self.last = (response.status_code, await response.aread())


def _option(text: str) -> tuple[str, Any]:
    name, _, value = text.partition("=")
    try:
        return name, json.loads(value)
    except json.JSONDecodeError:
        return name, value


async def _record(args: argparse.Namespace) -> Path:
    metadata = AnySearch.get_provider_metadata(args.provider)
    api_key = os.environ.get(metadata.env_key) if metadata.env_key else None
    recorder = _Recorder()
    async with httpx.AsyncClient(event_hooks={"response": [recorder]}) as client:
        async with AnySearch.create(args.provider, api_key=api_key, client=client) as engine:
            try:
                await engine.search(args.query, max_results=args.max_results, **dict(args.option))
            except AnySearchError as exc:
                # An error case is recorded as well; the provider's answer is what matters.
                print(f"note: {exc}", file=sys.stderr)
    if recorder.last is None:
        raise SystemExit(f"{args.provider} sent no request, so there is nothing to record")
    status, raw_body = recorder.last
    text = raw_body.decode("utf-8", errors="replace")
    if api_key:
        text = text.replace(api_key, REDACTED)
    try:
        body: Any = json.loads(text)
    except json.JSONDecodeError:
        body = text
    path = FIXTURES / str(args.provider) / f"{args.case}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": status, "body": body}, indent=2, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a provider's live answer as a test fixture.")
    parser.add_argument("provider", choices=AnySearch.get_supported_providers())
    parser.add_argument("case", help="the fixture's name, such as normal, empty, error or in_body_error")
    parser.add_argument("query")
    parser.add_argument("--max-results", type=int)
    parser.add_argument("-o", "--option", type=_option, action="append", default=[], metavar="NAME=VALUE")
    print(asyncio.run(_record(parser.parse_args())))


if __name__ == "__main__":
    main()
