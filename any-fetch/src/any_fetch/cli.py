"""``any-fetch``: fetch one page from the command line, keys from the environment.

any-fetch fake https://www.python.org/downloads/
any-fetch fake https://www.python.org/downloads/ --max-chars 2000 --json
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any

from any_fetch._api import AnyFetch, afetch
from any_fetch._errors import AnyFetchError
from any_fetch._logging import install as install_log_filter
from any_fetch._types import FetchedPage


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="any-fetch", description="Fetch one page. Keys come from the environment.")
    parser.add_argument("provider", choices=AnyFetch.get_supported_providers())
    parser.add_argument("url")
    parser.add_argument("--max-chars", type=_positive, metavar="N")
    parser.add_argument(
        "-o",
        "--option",
        type=_option,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a native option of the provider; VALUE is read as JSON when it parses",
    )
    parser.add_argument("--json", action="store_true", help="print the page as JSON")
    parser.add_argument("--raw", action="store_true", help="print the provider's own response (with --json, keep it)")
    return parser


def _render(page: FetchedPage, *, as_json: bool, raw: bool) -> str:
    if as_json:
        return json.dumps(page.model_dump(mode="json", exclude=None if raw else {"raw"}), indent=2)
    if raw:
        return json.dumps(page.raw, indent=2, default=str)
    lines = [page.title or page.final_url, page.final_url, page.content_type, "", page.text]
    if page.source_truncated or page.text_truncated:
        lines += ["", "(truncated)"]
    if page.cost is not None:
        lines += ["", f"Cost: {page.cost} USD"]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; return its exit status."""
    install_log_filter()
    parser = _parser()
    args = parser.parse_args(argv)
    options = dict(args.option)
    if "max_chars" in options:
        parser.error("use --max-chars for a shared parameter")
    try:
        page = asyncio.run(afetch(args.provider, args.url, max_chars=args.max_chars, **options))
    except AnyFetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # A provider bug, not a reported failure: its text may carry the URL, so only its type is shown.
        print(f"error: {args.provider} failed unexpectedly ({type(exc).__name__})", file=sys.stderr)
        return 1
    if page.error is not None:
        status = "" if page.error.status is None else f", HTTP {page.error.status}"
        print(f"error: {args.provider} reported {page.error.tag}{status}", file=sys.stderr)
        return 1
    print(_render(page, as_json=args.json, raw=args.raw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
