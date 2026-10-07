"""``any-search``: run one search from the command line, keys from the environment.

any-search fake "latest stable python release" --max-results 2
any-search exa "latest stable python release" --json
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any, get_args

from any_search._api import AnySearch, asearch
from any_search._errors import AnySearchError
from any_search._logging import install as install_log_filter
from any_search._types import SearchResult, TimeRange


def _option(text: str) -> tuple[str, Any]:
    name, separator, value = text.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("an option is NAME=VALUE")
    try:
        return name, json.loads(value)
    except json.JSONDecodeError:
        return name, value


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="any-search", description="Run one web search. Keys come from the environment."
    )
    parser.add_argument("provider", choices=AnySearch.get_supported_providers())
    parser.add_argument("query")
    parser.add_argument("--max-results", type=_positive, metavar="N")
    parser.add_argument("--time-range", choices=get_args(TimeRange))
    parser.add_argument(
        "-o",
        "--option",
        type=_option,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a native option of the provider; VALUE is read as JSON when it parses",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--raw", action="store_true", help="print the provider's own response (with --json, keep it)")
    return parser


def _render(result: SearchResult, *, as_json: bool, raw: bool) -> str:
    if as_json:
        exclude: Any = None if raw else {"raw": True, "hits": {"__all__": {"raw"}}}
        return json.dumps(result.model_dump(mode="json", exclude=exclude), indent=2)
    if raw:
        return json.dumps(result.raw, indent=2, default=str)
    lines: list[str] = []
    for number, hit in enumerate(result.hits, start=1):
        lines += [f"{number}. {hit.title or hit.url}", f"   {hit.url}"]
        if hit.snippet:
            lines.append(f"   {hit.snippet}")
    if not result.hits:
        lines.append("No results.")
    if result.cost is not None:
        lines.append(f"Cost: {result.cost} USD")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; return its exit status."""
    install_log_filter()
    parser = _parser()
    args = parser.parse_args(argv)
    options = dict(args.option)
    if shared := {"max_results", "time_range"} & options.keys():
        parser.error(f"use --{sorted(shared)[0].replace('_', '-')} for a shared parameter")
    try:
        result = asyncio.run(
            asearch(
                args.provider,
                args.query,
                max_results=args.max_results,
                time_range=args.time_range,
                **options,
            )
        )
    except AnySearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if result.error is not None:
        status = "" if result.error.status is None else f", HTTP {result.error.status}"
        print(f"error: {args.provider} reported {result.error.tag}{status}", file=sys.stderr)
        return 1
    print(_render(result, as_json=args.json, raw=args.raw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
