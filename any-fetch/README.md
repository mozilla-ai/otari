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
`tier` (`production` or `test`), `max_urls_per_call`, whether it `renders_javascript`, the
`formats` it returns text in, and its native options as `OptionSpec`s (`name`, `type`, `enum`,
`default`, `description` and `operator_only`, for an option only a deployment's operator should
set; the library accepts it from anyone, and enforcing that is the host's job).

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

## Exa

`exa` calls Exa's [`POST /contents`](https://exa.ai/docs/reference/get-contents) for the URL,
with the key in the `x-api-key` header, read from `EXA_API_KEY` when not passed.

It always asks for the page text, up to `max_chars`, or up to 10,000 characters, what Exa's own
Python SDK asks for, when `max_chars` is not passed. A longer page is cut there and marked
`text_truncated`. The `text` option shapes the extraction (`verbosity`, `includeHtmlTags`,
`includeSections`, `excludeSections`), and `true` keeps Exa's defaults; its length is `max_chars`
only, so `text.maxCharacters` is refused. The other native options are `highlights`, `summary` and `extras`, whose answers stay in
`raw`, and `maxAgeHours`, `livecrawlTimeout` and `snapshotAsOf`.

`final_url` is the URL Exa answered for, `published` its `publishedDate`, and the cost
`costDollars.total`. `content_type` is empty: Exa does not say what the page was served as.

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

`final_url` is the `url` Tabstack answered for, which is the URL as asked even after a redirect.
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
