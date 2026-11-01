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
`UnsupportedParameterError` rather than being dropped. `SHARED_PARAMETERS` lists the shared
parameters, so a host can tell them from native options, and `shared_parameter_problems` says which
values `search` refuses, so a host can check them before the call.

## What comes back

A `SearchResult`: the provider's name, a list of `SearchHit`s (`url`, `title`, `snippet`, the page
`text` when the provider returned it, `published`, and the provider's own item as `raw`), the
`cost` in USD the provider reported with `cost_source` (`reported` or `none`), an `error` when the
provider signaled one inside a successful response, and the provider's whole response as `raw`.
`raw` is left out of `repr`.

A failed call raises `ProviderError` with the provider's name, a `tag` and the HTTP `status` when
there is one. A successful answer that lacks a field the provider documents, or carries one of
the wrong type, raises it with the tag `unexpected_response`, so a change on the provider's side
fails the call rather than degrading it. A missing key raises `MissingCredentialError`. All errors derive from
`AnySearchError`, and none carries a response body, a URL, the query or a key.

`ProviderMetadata` says what a host needs before calling: where the key comes from, whether the
query or the key travels in the URL, the provider's `tier` (`production` or `test`), its
`max_results`, and its native options as `OptionSpec`s (`name`, `type`, `enum`, `default`,
`description` and `operator_only`, for an option only a deployment's operator should set; the
library accepts it from anyone, and enforcing that is the host's job).

## Logging

The query is the user's content, and some providers put it, or the key, in the request URL, which
httpx logs at INFO. Every provider call runs with a context variable set, and a filter on the
`httpx` logger replaces the URL with `<redacted>` in records emitted during one. Over HTTP/2, hpack
logs each request header at DEBUG, the path with its query included, so the filter also redacts
the headers in hpack's records. Requests made by other code in the process are left alone.

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
any-search --describe exa
```

Keys come from the environment. `-o NAME=VALUE` passes a native option, its value read as JSON
when it parses. `--json` prints the result as JSON without `raw`, `--raw` prints the provider's
own response, and both together print the result with `raw` kept. `--describe PROVIDER` prints the
provider's metadata and each native option with its type, allowed values, default and
description, then exits without searching.

## DeepSeek

`deepseek` calls DeepSeek's [Search API](https://api-docs.deepseek.com/api/search),
`POST /v1/tools/search`, a search engine of DeepSeek's own, apart from its models, with the key in
a bearer header, read from `DEEPSEEK_API_KEY` when not passed. DeepSeek publishes no SDK for it, so
the adapter sends the query, and `count` when `max_results` is passed, and nothing else unless you
pass a native option.

| `SearchHit` | From DeepSeek |
|---|---|
| `url`, `title` | `url`, `title` |
| `snippet` | `snippet`, DeepSeek's short text; its longer `summary`, written for a model, stays in `raw` |
| `text` | `content`, which DeepSeek returns only with `need_content=True`; empty counts as none |
| `published` | `date`, RFC 3339 in UTC, or `null` when DeepSeek does not know it |

DeepSeek reports no cost. `max_results` becomes `count`, at most 50; without it DeepSeek returns
its default of 10. DeepSeek has no recency parameter, so `time_range` becomes its `after:` query
operator, the day the range starts on as `YYYY-MM-DD`, appended to the query. DeepSeek counts that
day in from midnight China Standard Time (UTC+8), so the range covers at least its span. The
operator counts toward DeepSeek's 128-character limit on the query, so a query within 17 characters of that limit is
refused with a 400 when a `time_range` is passed. The other native options are `ori_question`,
`localization` (an IP address) and `need_content`; an option set to `None` is not sent.

A failed call raises `ProviderError` with DeepSeek's HTTP status, tagged with the `error.type` of
its OpenAI-shaped body: `authentication_error` for a refused key, `invalid_request_error` for a bad
request. Its `error.code` is `invalid_request_error` for every failure, so it is the tag only when
there is no type. The message is never read, since for a refused key it quotes the key's last four
characters. A successful search carries no error of its own.

Every field read from a successful answer is checked: an answer without `results`, or an item
without `url`, `title`, `snippet`, `content` or `date`, all of which DeepSeek documents on every
result, or with any of them of the wrong type or a `date` that does not parse, raises
`ProviderError` with the tag `unexpected_response`. Fields the adapter does not read are ignored.

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

Every field read from a successful search is checked, so a change on Exa's side fails the call
rather than degrading it: an answer without `results` or an item's `url` or `title`, or with any
of these, `text`, `highlights`, `publishedDate` or `costDollars.total` of the wrong type, or a date
that does not parse, raises `ProviderError` with the tag `unexpected_response`. Exa documents its
cost as optional, so an answer without one reports none; the live tests check that it reports
one. Fields the adapter does not read are ignored, so a field Exa adds breaks nothing.

## Mojeek

`mojeek` calls Mojeek's [Web Search API](https://www.mojeek.com/support/api/search/request_parameters.html),
`GET /search`, with the key read from `MOJEEK_API_KEY` when not passed. Mojeek takes everything in
the URL, the key and the query included, so the adapter never hands the URL to a logger, and the
log filter above keeps it out of httpx's log.

It always sends `fmt=json`, since Mojeek answers in XML otherwise, and `date=1`, unless you pass
`date=False`, so each hit is dated by the page's last-modified time.

| `SearchHit` | From Mojeek |
|---|---|
| `url`, `title` | `url`, `title` |
| `snippet` | `desc`; Mojeek returns no page text, so `text` is always `None` |
| `published` | `pdate`, else `timestamp`, both Unix times; 0, which Mojeek sends for a time it does not know, counts as none |

Mojeek reports no cost. `max_results` becomes `t`, at most 10, the most Mojeek's entry plan
returns; its higher plans return up to 40 or 100, but the adapter asks for no more than 10.
`time_range` becomes `since`: Mojeek's own `day`, `month` or `year`, and for a week the date it
starts on, as `YYYYMMDD`. It wins over the `since` option. The other native options are Mojeek's
request parameters under their own names, except those of its OrgSearch product; `--describe
mojeek` lists them. Booleans are sent as Mojeek's 0 and 1, and an option set to `None` is not sent.

Mojeek signals a refused key or a spent daily quota inside a 200, in `response.status`, so any
status but `OK` is returned as `SearchResult.error`, tagged `access_denied`, `daily_limit_reached`
or, for any other, `error`. A malformed parameter gets an HTTP 503 with a plain body; any status
from 400 raises `ProviderError` with the tag `http_error`.

Every field read from a successful answer is checked: an answer without `response` or its
`status`, or, when the status is `OK`, without `results` or an item's `url`, `title` or `desc`, or
with any of these of the wrong type, raises `ProviderError` with the tag `unexpected_response`.
Mojeek's documentation lists `pdate` and `timestamp` on every result, but they come only on
request (`timestamp` with `date=1`), so neither is required: the date is read from `pdate` when it
is present and not 0, else from `timestamp`, and the field read must be a Unix time. A `timestamp`
behind a usable `pdate` is not read. Fields the adapter does not read are ignored.

## Serply

`serply` calls Serply's [Google Search API](https://serply.io/docs/resources/google-search),
`GET /v1/search`, with the key read from `SERPLY_API_KEY` when not passed. The key travels in the
`X-Api-Key` header; the query travels in the URL, which the log filter above keeps out of httpx's
log. Each request names the library in its `User-Agent`.

| `SearchHit` | From Serply |
|---|---|
| `url`, `title` | `link`, `title` |
| `snippet` | `description`; Serply returns no page text, so `text` is always `None` |
| `published` | always `None`: Serply's results carry no date |

Serply reports no cost. `max_results` becomes `num`, at most 10, one page of Google's results;
a page can hold one more than asked for, so the hits are cut to the count as well. `time_range`
becomes Google's `tbs` window, `qdr:d`, `qdr:w`, `qdr:m` or `qdr:y`, and wins over the `tbs`
option. The other native options are `gl`, `start`, `tbs` and `tbm` (`nws` for Google News),
sent as query parameters, and `X-Proxy-Location` and `X-User-Agent`, sent as headers under those
names; `--describe serply` lists them. An option set to `None` is not sent.

Serply signals every failure with an HTTP status and a `detail` sentence in the body. Any status
from 400 raises `ProviderError`, tagged `invalid_api_key` or `missing_api_key` from how the detail
starts, else `http_error`; the detail itself never reaches the error.

Every field read from a successful answer is checked: an answer without `results`, or with an item
lacking `link`, `title` or `description`, or with any of these of the wrong type, raises
`ProviderError` with the tag `unexpected_response`. A `link` that is not an http or https URL,
such as an opaque Google redirect token, which Serply sometimes returns in its place, is dropped
rather than failing the search, and the count is filled from the items after it. Ads, the
knowledge graph and Serply's other SERP features come in fields of their own, which the adapter
does not read, so a field Serply adds breaks nothing.

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

It checks the data as well as the call, against what its `EXPECTED` table says each provider must
return, which a production provider joins as it joins the conformance scenarios. CI runs it weekly,
on demand, and on a pull request that changes either library
(`.github/workflows/otari-libs-live.yml`), with the keys from repository secrets.
