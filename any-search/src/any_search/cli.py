"""``any-search``: run one search from the command line, keys from the environment.

any-search fake "latest stable python release" --max-results 2
any-search fake "latest stable python release" --json
any-search --describe exa
"""

import argparse
import asyncio
import json
import sys
import textwrap
from collections.abc import Sequence
from typing import Any, get_args

from any_search._api import AnySearch, asearch
from any_search._errors import AnySearchError
from any_search._logging import install as install_log_filter
from any_search._types import OptionSpec, ProviderMetadata, SearchResult, TimeRange


def _option(text: str) -> tuple[str, Any]:
    name, separator, value = text.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("an option is NAME=VALUE")
    try:
        return name, json.loads(value)
    except json.JSONDecodeError:
        return name, value


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _source(required: bool, env: str | None, default: str | None) -> str:
    if default:
        return f"{default}, or {env} when set" if env else default
    need = "required" if required else "optional" if env else "none"
    return f"{need}, from {env}" if env else need


def _describe_option(option: OptionSpec) -> list[str]:
    kind = f"{option.type}, operator only" if option.operator_only else option.type
    head = f"  {option.name} ({kind})"
    if option.default is not None:
        head += f", default {json.dumps(option.default)}"
    lines = [head]
    if option.enum:
        lines.append(f"      One of: {', '.join(option.enum)}.")
    if option.description:
        lines += textwrap.wrap(option.description, width=100, initial_indent=" " * 6, subsequent_indent=" " * 6)
    return lines


def _describe(metadata: ProviderMetadata) -> str:
    lines = [
        f"{metadata.name}: {metadata.doc_url}",
        f"API key: {_source(metadata.requires_api_key, metadata.env_key, None)}",
        f"API base: {_source(metadata.requires_api_base, metadata.env_api_base, metadata.default_api_base)}",
        f"Results: at most {metadata.max_results} per call",
    ]
    if metadata.key_in_url:
        lines.append("The key travels in the request URL.")
    if metadata.query_in_url:
        lines.append("The query travels in the request URL.")
    if metadata.tier == "test":
        lines.append("For tests: it answers from canned data and its options, and calls no service.")
    lines += ["Shared parameters: --max-results, --time-range", ""]
    if not metadata.options:
        lines.append("Native options: none.")
        return "\n".join(lines)
    lines.append("Native options, passed with -o NAME=VALUE (VALUE read as JSON when it parses):")
    for option in metadata.options:
        lines += _describe_option(option)
    if any(option.operator_only for option in metadata.options):
        lines += ["", "Operator only: a host lets only its operator set it, never a tenant or a request."]
    return "\n".join(lines)


class _Describe(argparse.Action):
    """Print a provider's metadata and exit while parsing, as --help does, so no query is needed."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        print(_describe(AnySearch.get_provider_metadata(str(values))))
        parser.exit()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="any-search", description="Run one web search. Keys come from the environment."
    )
    parser.add_argument(
        "--describe",
        action=_Describe,
        choices=AnySearch.get_supported_providers(),
        metavar="PROVIDER",
        help="print the provider's metadata and native options, then exit",
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
    except Exception as exc:
        # A provider bug, not a reported failure: its text may carry the query, so only its type is shown.
        print(f"error: {args.provider} failed unexpectedly ({type(exc).__name__})", file=sys.stderr)
        return 1
    if result.error is not None:
        status = "" if result.error.status is None else f", HTTP {result.error.status}"
        print(f"error: {args.provider} reported {result.error.tag}{status}", file=sys.stderr)
        return 1
    print(_render(result, as_json=args.json, raw=args.raw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
