# any-fetch

One interface to web fetch providers, built the way
[any-llm](https://github.com/mozilla-ai/any-llm) is: import it, give it a key,
call it. otari is its first consumer, not its owner, so it runs without otari.

It depends on `httpx` and `pydantic` and nothing else. Providers are called over
HTTP directly, with no vendor SDKs.

## Calling a provider

```python
from any_fetch import AnyFetch, afetch

page = await afetch("fake", "https://www.python.org/downloads/", max_chars=20_000)

async with AnyFetch.create("fake", timeout=15.0) as fetcher:
    page = await fetcher.fetch("https://www.python.org/downloads/")

AnyFetch.get_supported_providers()
AnyFetch.get_provider_metadata("fake")
```

`create` takes:

- `api_key`: the provider's key. When it is not passed, the environment variable the provider's
  metadata names (`env_key`) is read. A key passed as an empty string counts as passed and never
  falls back to the environment.
- `api_base`: overrides the provider's endpoint, which is how a test or a proxy redirects it. When
  it is not passed, the variable `env_api_base` names is read, then the provider's default.
- `timeout`: seconds per request, 15 by default.
- `client`: an `httpx.AsyncClient` you own. The provider sends every request through it with its own
  timeout, and never closes it or changes its settings. Without one, the provider opens a client
  with httpx's defaults and closes it in `aclose()`, which `async with` calls.

Every request a provider sends carries `User-Agent: any-fetch/<version>`, the library's own
version (`any_fetch.__version__`, which a change to the library bumps in the same PR), so a
provider, or a proxy that allowlists outbound traffic, can tell the library's requests apart. A
`client` that sets a `User-Agent` of its own keeps it: httpx's default is replaced, and a client
that carries none is filled in. `builtin` is the host's own fetcher, and sends what the host sends.

`fetch` takes one shared parameter, which every provider maps onto its own: `max_chars`. Every
other keyword is a native option, passed to the provider under its own name. An option the
provider's metadata does not list raises `UnsupportedParameterError` rather than being dropped.
`SHARED_PARAMETERS` lists the shared parameter, so a host can tell it from native options, and
`shared_parameter_problems` says which values `fetch` refuses, so a host can check them before the call.

## What comes back

A `FetchedPage`: the `url` asked for, the `final_url` after redirects, the `title`, the `text`, the
`content_type`, `published`, the `cost` in USD the provider reported with `cost_source`
(`reported` or `none`), an `error` when the provider signaled one inside a successful response,
`source_truncated` when the response body hit the fetcher's size limit, `text_truncated` when the
text hit its own, and the provider's own response as `raw`, which is left out of `repr`.

A failed call raises `ProviderError` with the provider's name, a `tag` and the HTTP `status` when
there is one. A successful answer that lacks a field the provider documents, or carries one of
the wrong type, raises it with the tag `unexpected_response`, so a change on the provider's side
fails the call rather than degrading it. A missing key raises `MissingCredentialError`. All errors derive from
`AnyFetchError`, and none carries a response body, a URL or a key.

`ProviderMetadata` says what a host needs before calling: where the key comes from, the provider's
`tier` (`production` or `test`), `max_urls_per_call`, whether it `renders_javascript`, whether it
`reports_final_url` (when false, `final_url` is the URL as asked, because the provider follows
redirects without saying where they led, so a caller cannot check a redirect it followed), the
`formats` it returns text in, whether it `reports_cost`, and its native options as `OptionSpec`s
(`name`, `type`, `enum`, `default`, `description`, `operator_only`, for an option only a
deployment's operator should set, and `cost_tier`, for the option whose values are the provider's
cost tiers, each billed at a price per call of its own; the library accepts both from anyone, and
acting on them is the host's job). A cost tier option's `enum` lists the tiers and its `default` is
the one a call is billed at when the option is not passed: the provider's own, unless the adapter
sends another. For example `youdotcom`'s any-fetch adapter sends `lite` by default, even though the API default
is `standard`. The metadata records no amount.

## The builtin provider

`builtin` is the plain fetcher of `http` and `https` pages. This package declares its id and
metadata; the implementation is the host's, which registers a factory that builds one:

```python
AnyFetch.register_builtin(factory)
fetcher = AnyFetch.create("builtin", client=client, allow_private_hosts=False)
```

`create("builtin", **kwargs)` passes its keyword arguments, whatever they are, to the factory, and
wraps the fetcher it builds, which has `async fetch(url, *, max_chars)` returning a `FetchedPage`
and `async aclose()`. The wrapper runs each fetch in the same logging context as any other
provider's. Creating `builtin` before a factory is registered raises `BuiltinNotRegisteredError`.
otari will register one that wraps the gateway's hardened fetcher.

## Logging

The URL a fetch is asked for is the user's content, and httpx logs every request's full URL at
INFO. Every provider call, `builtin` included, runs with a context variable set, and a filter on the
`httpx` logger replaces the URL with `<redacted>` in records emitted during one. Requests made by
other code in the process are left alone.

Importing the package does not install the filter, so it never changes a process's logging on its
own. A host calls `install_log_filter()` once at startup; the command line installs it itself.

```python
from any_fetch import install_log_filter

install_log_filter()
```

## The command line

```bash
any-fetch fake https://www.python.org/downloads/ --max-chars 200
any-fetch fake https://www.python.org/downloads/ --json
any-fetch fake https://www.python.org/downloads/ -o error=http_status -o error_status=404
any-fetch --describe exa
```

Keys come from the environment. `-o NAME=VALUE` passes a native option, its value read as JSON
when it parses. `--json` prints the page as JSON without `raw`, `--raw` prints the provider's own
response, and both together print the page with `raw` kept. `--describe PROVIDER` prints the
provider's metadata and each native option with its type, allowed values, default and
description, then exits without fetching. `builtin` needs a host, so it fails from the command
line, though `--describe builtin` works.

## DeepSeek

`deepseek` calls DeepSeek's [Fetch API](https://api-docs.deepseek.com/api/fetch),
`POST /v1/tools/fetch`, with the key in a bearer header, read from `DEEPSEEK_API_KEY` when not
passed. It fetches one page per call and returns its title and its text as markdown. It takes the
URL and nothing else, so there are no native options. DeepSeek has no length cap, so `max_chars`
cuts the text it returns and sets `text_truncated`, and without `max_chars` the whole page comes
back. `final_url` is the URL asked for, since DeepSeek follows a redirect without saying where it
ended, so `reports_final_url` is false: a caller cannot check where a redirect DeepSeek followed
led. `content_type` is empty. DeepSeek reports no cost and no date.

A failed call raises `ProviderError` with DeepSeek's HTTP status, tagged with the `error.type` of
its OpenAI-shaped body, or its `error.code` when there is no type; the message is never read. A
page DeepSeek cannot turn into markdown fails the call with a 404 tagged `not_found_error`, and the
answer does not say why. That was seen for a page that does not exist, a site's 500, a private
address, a PDF, a zero-byte file and HTML with an empty body, so DeepSeek never returns an empty
page, and a missing page never comes back as the site's own error page.

Every field read from a successful answer is checked: a `title` or a `content` that is missing or
not a string raises `ProviderError` with the tag `unexpected_response`. DeepSeek documents both as
always present. Fields the adapter does not read are ignored.

## Exa

`exa` calls Exa's [`POST /contents`](https://exa.ai/docs/reference/get-contents) for the URL,
with the key in the `x-api-key` header, read from `EXA_API_KEY` when not passed.

It always asks for the page text, up to `max_chars`, or up to 10,000 characters, what Exa's own
Python SDK asks for, when `max_chars` is not passed. A longer page is cut there and marked
`text_truncated`. The `text` option shapes the extraction (`verbosity`, `includeHtmlTags`,
`includeSections`, `excludeSections`), and `true` keeps Exa's defaults; its length is `max_chars`
only, so `text.maxCharacters` is refused. The other native options are `highlights`, `summary` and `extras`, whose answers stay in
`raw`, and `maxAgeHours`, `livecrawlTimeout` and `snapshotAsOf`.

`final_url` is the URL Exa answered for, which is the URL as asked even after a redirect, so
`reports_final_url` is false: a caller cannot check where a redirect Exa followed led.
`published` is its `publishedDate`, and the cost `costDollars.total`. `content_type` is empty: Exa does not say what the page was served as.

A page Exa could not fetch comes back as `FetchedPage.error`, with Exa's tag, such as
`CRAWL_NOT_FOUND`, and the page's HTTP status. A page with no text, which Exa reports as
`CRAWL_EMPTY_CONTENT`, comes back empty rather than as an error. A failed call raises
`ProviderError` with Exa's HTTP status and its `tag`, such as `INVALID_API_KEY`.

Every field read from a successful answer is checked, so a change on Exa's side fails the call
rather than degrading it. An answer without `results`, a `statuses` list without an entry for the
URL whose `status` is `success` or `error`, a success without its page's `url` and `title`, any of these,
`text` or `costDollars.total` of the wrong type, or a `publishedDate` that does not parse, raises
`ProviderError` with the tag `unexpected_response`. So does an answer with no page and no reason.
Exa documents `statuses` and a page's `text` as optional: without `statuses` the page in
`results` is the answer, and a success without `text` is an empty page.
Exa documents its cost as optional, so an answer without one reports none; the live tests check
that it reports one.
Fields the adapter does not read are ignored, so a field Exa adds breaks nothing.

## Tabstack

`tabstack` calls the [`POST /extract/markdown`](https://docs.tabstack.ai/api/resources/extract/methods/markdown)
endpoint of Tabstack, Mozilla's web API for agents, with the key in a bearer header, read from
`TABSTACK_API_KEY` when not passed. Its base URL is `https://api.tabstack.ai/v1`, or
`TABSTACK_BASE_URL` when set, the variable Tabstack's own SDKs read.

It fetches one page per call and returns its text as markdown. It always asks for the page's
metadata as a field of its own, which is where `title` comes from, so the text carries no YAML
front matter; `metadata` is therefore not an option. Tabstack has no length cap, so `max_chars`
cuts the text it returns and sets `text_truncated`, and without `max_chars` the whole page comes
back. The native options are `content` (`main`, the default, or `full`), `effort` (`min`,
`standard`, the default, or `max`), `geo_target` (`{"country": "GB"}`) and `nocache`. Tabstack
documents up to 15 seconds for a fetch at `standard` effort, the default `timeout`, and 15 to 60
at `max`, which renders the page in a browser, so pass a `timeout` to match.

`final_url` is the `url` Tabstack answered for, which is the URL as asked even after a redirect, so
`reports_final_url` is false: a caller cannot check where a redirect Tabstack followed led, for
example against its own allow or deny lists.
`content_type` is empty and `published` is `None`: Tabstack does not say what the page was
served as, and its `created_at` is not a date to rely on, since for a PDF it carries the PDF's
own date format rather than the ISO 8601 it documents. The metadata stays in `raw`. Tabstack
reports no cost, so `cost_source` is `none`.

A failed call raises `ProviderError` with Tabstack's HTTP status and the tag `http_error`: its
error body is a sentence that can quote the URL, so it is not read. Tabstack reports no error
inside a successful answer. A page it cannot fetch fails the call, as does a URL it refuses, one
whose host does not resolve (422) or that points at a private address (403). A page whose site
answered with an error status, such as a 404, comes back as the text of that error page, since
Tabstack does not say what status the site answered with.

Every field read from a successful answer is checked. An answer whose `content` is not a string,
or whose `url` is not a non-empty string, raises `ProviderError` with the tag
`unexpected_response`, and so does a `metadata` that is not an object or a `metadata.title` that
is not a string. Tabstack documents those two as optional, and without them the title is empty.
Fields the adapter does not read are ignored, so a field Tabstack adds breaks nothing.

## You.com

`youdotcom` calls You.com's [Extract API](https://you.com/docs/api-reference/extract),
`POST /v1/extract`, with the key in the `X-API-Key` header, read from `YDC_API_KEY` when not
passed. One web page or PDF per call, returned as Markdown. `parse_effort` is `lite` unless
passed: You.com's default, `standard`, bills each PDF page on top of the URL, and the option is
the cost tier.

You.com has no length cap, so `max_chars` cuts the text and sets `text_truncated`. Native options:
`full_page` (Markdown is always requested by the adapter; an HTML copy, when additionaly requested, stays in `raw`), 
`extraction_source` (`blend`, `cache`, `fetch`), `timeout` (2 to 100 seconds) and `parse_effort` (`lite`, `standard`).
`None` counts as not passed. You.com's `timeout` is longer than the fetcher's own 15 seconds, and
`afetch`'s `timeout` argument is the fetcher's, so pass You.com's through `fetch` and give the
fetcher a longer one. `blend` can serve a cached copy that differs from the live page, so pass
`extraction_source="fetch"` where the page as it is now matters.

`final_url` is the URL as asked, even after a redirect, so `reports_final_url` is false.
`content_type` is empty, `published` is `None`, and You.com reports no cost.

A body without `contents` is `FetchedPage.error`: `not_cached` when the fetch asked for the cache
only, else `nothing_extracted`. A site's own error page, such as a 404, comes back as that page's
text, which can be empty.

A failed call raises `ProviderError`. 401 is `unauthorized` and a refused key's 403 is
`forbidden`, by status, since those bodies are the gateway's. Any other status is tagged with
`error.code` when it carries one (`invalid_url`, `site_blocked`, `site_unreachable`,
`unsupported_content`, `timeout`); the message and details are not read. A 402 without one takes
the code in its flat `error`, else `payment_required`. Anything else is `http_error`.

A `url` or `title` that is not a string, or `contents` that does not hold `full_page.markdown` as
a string, raises `ProviderError` with the tag `unexpected_response`. Both are optional: without
them the URL is the one asked for and the title is empty. An empty `markdown` is an empty page.
Unread fields are ignored.

## The fake provider

`fake` is a provider like any other, in the `test` tier. It answers with a canned page and takes
its behavior from native options, so tests can drive every path a real provider takes without a
network or a monkeypatch:

| Option | Effect |
|---|---|
| `text` | The page text. Canned text when unset; `max_chars` cuts it and sets `text_truncated`. |
| `title`, `content_type`, `final_url` | The page's title, content type (`text/html` by default) and final URL (the URL asked for by default). |
| `source_truncated` | Report that the response body hit the fetcher's size limit. |
| `cost` | A cost in USD to report. |
| `delay` | Seconds to wait before answering, to hold a call open. |
| `error` | Fail the call with a `ProviderError` of this tag. |
| `error_status` | The HTTP status of either error. |
| `in_body_error` | Answer with no text and this tag as `FetchedPage.error`. |
| `leak_url` | Raise a `RuntimeError` whose message carries the URL, as a careless adapter might, so a host can prove it never passes exception text on. |
| `account` | Echoed in `raw`. Operator-only, so a host can exercise that rule. |
| `depth` | `basic` or `advanced`, echoed in `raw`. A cost tier, so a host can exercise its rule for cost tiers. |

`raw` echoes the request, the URL left out, so a test can see what reached the provider.

## Tests

From the repository root, `make test-libs` runs this package's tests and any-search's.
`tests/conformance/` holds the checks every provider must pass; a provider joins them by adding
its scenarios there, and `builtin` is skipped while no factory is registered.
`scripts/record_fixture.py` records a provider's live answer as a fixture under
`tests/fixtures/<provider>/`, with the key redacted.

`tests/test_live.py` calls every production provider whose key is in the environment, and is
skipped unless `ANY_FETCH_LIVE=1`:

```bash
ANY_FETCH_LIVE=1 uv run pytest any-fetch/tests/test_live.py
```

It checks the data as well as the call, against what its `EXPECTED` table says each provider must
return, which a production provider joins as it joins the conformance scenarios. CI runs it weekly,
on demand, and on a pull request that changes either library
(`.github/workflows/otari-libs-live.yml`), with the keys from repository secrets.
