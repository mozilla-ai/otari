"""The fake provider: a canned page, with its behavior taken from native options.

It is a provider like any other, in the ``test`` tier, so the library's own
tests and a host's can drive every path a real provider takes without a
network or a monkeypatch.
"""

import asyncio
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from any_fetch._api import AnyFetch
from any_fetch._errors import ProviderError
from any_fetch._types import FetchedPage, FetchError, OptionSpec, ProviderMetadata

CANNED_TITLE = "Fake page"
CANNED_TEXT = (
    "The canned text of the fake fetch provider. It stands in for the readable text of a web page, "
    "long enough for a test to cut it short with max_chars."
)


class FakeOptions(BaseModel):
    """The native options, parsed. Their names match the metadata's option list."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    title: str | None = None
    content_type: str = "text/html"
    final_url: str | None = None
    source_truncated: bool = False
    cost: Decimal | None = None
    delay: float = 0.0
    error: str | None = None
    error_status: int | None = None
    in_body_error: str | None = None
    leak_url: bool = False
    account: str | None = None


class FakeProvider(AnyFetch):
    """Answers with a canned page; see the option descriptions for what each one does."""

    METADATA = ProviderMetadata(
        name="fake",
        doc_url="https://github.com/mozilla-ai/otari/tree/main/any-fetch#the-fake-provider",
        env_key=None,
        env_api_base=None,
        requires_api_key=False,
        requires_api_base=False,
        default_api_base=None,
        tier="test",
        max_urls_per_call=1,
        renders_javascript=False,
        formats=["text"],
        options=[
            OptionSpec(name="text", type="string", description="The page text. Canned text when unset."),
            OptionSpec(name="title", type="string", description="The page title."),
            OptionSpec(name="content_type", type="string", default="text/html", description="The content type."),
            OptionSpec(name="final_url", type="string", description="Where the page was found. The URL when unset."),
            OptionSpec(
                name="source_truncated",
                type="boolean",
                default=False,
                description="Report that the response body hit the fetcher's size limit.",
            ),
            OptionSpec(name="cost", type="number", description="A cost in USD to report."),
            OptionSpec(name="delay", type="number", default=0.0, description="Seconds to wait before answering."),
            OptionSpec(name="error", type="string", description="Fail the call with a ProviderError of this tag."),
            OptionSpec(name="error_status", type="integer", description="The HTTP status of either error."),
            OptionSpec(
                name="in_body_error",
                type="string",
                description="Answer with no text and this tag as an error signaled inside a successful response.",
            ),
            OptionSpec(
                name="leak_url",
                type="boolean",
                default=False,
                description=(
                    "Raise a RuntimeError whose message carries the URL, as a careless adapter might, so a "
                    "host can prove it never passes exception text on."
                ),
            ),
            OptionSpec(
                name="account",
                type="string",
                operator_only=True,
                description="An account name, echoed in raw. Operator-only, so a host can exercise that rule.",
            ),
        ],
    )

    async def _fetch(self, url: str, *, max_chars: int | None, options: dict[str, Any]) -> FetchedPage:
        # A known option with a value of the wrong type, such as delay=abc from the command line, is refused
        # the way a provider refuses a bad value: as a library error, never pydantic's, which quotes the value.
        try:
            parsed = FakeOptions.model_validate(options)
        except ValidationError:
            raise ProviderError(self.METADATA.name, None, "invalid_option") from None
        if parsed.delay > 0:
            await asyncio.sleep(parsed.delay)
        if parsed.leak_url:
            raise RuntimeError(f"fake fetch failed for {url!r}")
        if parsed.error is not None:
            raise ProviderError(self.METADATA.name, parsed.error_status, parsed.error)
        # The request is echoed, minus the URL, so a host's test can see what reached the provider.
        raw: dict[str, Any] = {"request": {"max_chars": max_chars, "options": options}}
        if parsed.in_body_error is not None:
            return FetchedPage(
                url=url,
                final_url=url,
                text="",
                content_type="",
                cost_source="none",
                error=FetchError(tag=parsed.in_body_error, status=parsed.error_status),
                raw=raw,
            )
        text = CANNED_TEXT if parsed.text is None else parsed.text
        cut = max_chars is not None and len(text) > max_chars
        return FetchedPage(
            url=url,
            final_url=parsed.final_url or url,
            title=CANNED_TITLE if parsed.title is None else parsed.title,
            text=text[:max_chars] if cut else text,
            content_type=parsed.content_type,
            cost=parsed.cost,
            cost_source="none" if parsed.cost is None else "reported",
            source_truncated=parsed.source_truncated,
            text_truncated=cut,
            raw=raw,
        )
