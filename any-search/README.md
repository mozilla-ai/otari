# any-search

One interface to web search providers, built the way
[any-llm](https://github.com/mozilla-ai/any-llm) is: import it, give it a key,
call it. otari is its first consumer, not its owner, so it runs without otari.

It depends on `httpx` and `pydantic` and nothing else. Providers are called over
HTTP directly, with no vendor SDKs.

## Calling a provider

```python
from any_search import AnySearch, asearch

result = await asearch("fake", "latest stable python release", max_results=5)

async with AnySearch.create("fake", timeout=15.0) as engine:
    result = await engine.search("latest stable python release", max_results=5, time_range="month")

AnySearch.get_supported_providers()
AnySearch.get_provider_metadata("fake")
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

`search` takes two shared parameters, which every provider maps onto its own: `max_results` and
`time_range` (`day`, `week`, `month` or `year`). Every other keyword is a native option, passed to
the provider under its own name. An option the provider's metadata does not list raises
`UnsupportedParameterError` rather than being dropped.

## What comes back

A `SearchResult`: the provider's name, a list of `SearchHit`s (`url`, `title`, `snippet`, the page
`text` when the provider returned it, `published`, and the provider's own item as `raw`), the
`cost` in USD the provider reported with `cost_source` (`reported` or `none`), an `error` when the
provider signaled one inside a successful response, and the provider's whole response as `raw`.
`raw` is left out of `repr`.

A failed call raises `ProviderError` with the provider's name, a `tag` and the HTTP `status` when
there is one. A missing key raises `MissingCredentialError`. All errors derive from
`AnySearchError`, and none carries a response body, a URL, the query or a key.

`ProviderMetadata` says what a host needs before calling: where the key comes from, whether the
query or the key travels in the URL, the provider's `tier` (`production` or `test`), its
`max_results`, and its native options as `OptionSpec`s (`name`, `type`, `enum`, `default`,
`description` and `operator_only`, for an option only a deployment's operator should set; the
library accepts it from anyone, and enforcing that is the host's job).

## Logging

The query is the user's content, and some providers put it, or the key, in the request URL, which
httpx logs at INFO. Every provider call runs with a context variable set, and a filter on the
`httpx` logger replaces the URL with `<redacted>` in records emitted during one. Requests made by
other code in the process are left alone.

Importing the package does not install the filter, so it never changes a process's logging on its
own. A host calls `install_log_filter()` once at startup; the command line installs it itself.

```python
from any_search import install_log_filter

install_log_filter()
```

## The command line

```bash
any-search fake "latest stable python release" --max-results 2
any-search fake "latest stable python release" --json
any-search fake "latest stable python release" -o cost=0.007 -o error=rate_limit
```

Keys come from the environment. `-o NAME=VALUE` passes a native option, its value read as JSON
when it parses. `--json` prints the result as JSON without `raw`, `--raw` prints the provider's
own response, and both together print the result with `raw` kept.

## Exa

`exa` calls Exa's [`POST /search`](https://exa.ai/docs/reference/search), with the key in the
`x-api-key` header, read from `EXA_API_KEY` when not passed.

It sends what Exa's own Python SDK sends by default, so moving from the SDK changes nothing:
`type` `auto`, and each page's text up to 10,000 characters when you pass no `contents`. That can
be a large answer, about 100,000 characters for ten hits at the cap. To get less, or something
else, pass `contents`, which is sent as given:

```python
await engine.search(query, contents={"text": {"maxCharacters": 2000}})  # less text per page
await engine.search(query, contents={"highlights": True})  # excerpts as snippets, no page text
await engine.search(query, contents=False)  # titles and URLs only
```

```bash
any-search exa "latest stable python release" -o 'contents={"highlights": true}'
```

| `SearchHit` | From Exa |
|---|---|
| `url`, `title` | `url`, `title` |
| `snippet` | `highlights`, joined; empty unless `contents` asks for highlights |
| `text` | `text` |
| `published` | `publishedDate`, a date or a date-time; UTC when it has no zone |

The cost is `costDollars.total`. `max_results` becomes `numResults`, at most 100; the `numResults`
option, capped the same way, applies only when `max_results` is not passed. `time_range` becomes
`startPublishedDate`, at the start of that day in UTC, since Exa often dates a page by its day
alone, and wins over that option. The other native options are `type`, `category`, `includeDomains`,
`excludeDomains`, `endPublishedDate`, `userLocation`, `moderation` and `additionalQueries`.

A failure raises `ProviderError` with Exa's HTTP status and its `tag`, such as `INVALID_API_KEY`
or `RATE_LIMIT_EXCEEDED`. Exa signals no error inside a successful search.

## The fake provider

`fake` is a provider like any other, in the `test` tier. It answers from canned data and takes its
behavior from native options, so tests can drive every path a real provider takes without a
network or a monkeypatch:

| Option | Effect |
|---|---|
| `hits` | The hits to answer with, each an object with `url` and optionally `title`, `snippet`, `text` and `published`. An item without `url` is dropped. Three canned hits when unset. |
| `cost` | A cost in USD to report. |
| `delay` | Seconds to wait before answering, to hold a call open. |
| `error` | Fail the call with a `ProviderError` of this tag. |
| `error_status` | The HTTP status of either error. |
| `in_body_error` | Answer with no hits and this tag as `SearchResult.error`. |
| `leak_query` | Raise a `RuntimeError` whose message carries the query, as a careless adapter might, so a host can prove it never passes exception text on. |
| `account` | Echoed in `raw`. Operator-only, so a host can exercise that rule. |

`raw` echoes the request, the query left out, so a test can see what reached the provider.

## Tests

From the repository root, `make test-libs` runs this package's tests and any-fetch's.
`tests/conformance/` holds the checks every provider must pass; a provider joins them by adding
its scenarios there. `scripts/record_fixture.py` records a provider's live answer as a fixture
under `tests/fixtures/<provider>/`, with the key redacted.

`tests/test_live.py` calls every production provider whose key is in the environment, and is
skipped unless `ANY_SEARCH_LIVE=1`:

```bash
ANY_SEARCH_LIVE=1 uv run pytest any-search/tests/test_live.py
```
