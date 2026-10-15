---
name: any-provider
description: Add a web search provider to any-search or a web fetch provider to any-fetch, the two libraries under any-search/ and any-fetch/. Use when asked to add, change or re-record a provider in either library, or to check one against the procedure.
---

# Adding a provider to any-search or any-fetch

The two libraries call web search and web fetch providers over HTTP, with no vendor SDKs, and
answer in one envelope per library. Each library's README says how a caller uses it; read it
first. This skill is how a provider gets in. It has a shared procedure, one checklist per library,
and the learnings each provider added, newest last.

A provider is one module, `<package>/src/<import name>/providers/<name>.py`, holding a class with
`METADATA` and `_search` (any-search) or `_fetch` (any-fetch), listed in the package's
`_registry.py`. The base class checks option names and the shared parameters, sets the logging
context and calls the adapter.

## The procedure

1. **Read the provider's documentation**, its OpenAPI spec when it has one, since that is usually
   the source of truth. Fill `METADATA`: the endpoint and method, where the key travels, whether
   a key or a base URL is required, the default base URL and the variable the provider's SDK
   reads it from, if any, the limits, and the native options under the provider's own names and
   enums. Note the pricing, with the date you read it, in the module's docstring.
2. **Write the adapter.** Build the request from the shared parameters and the native options,
   read the response into the envelope, and write the date parser, the error reader and the cost
   reader. The rules:
   - Send with `self._http.request`, which uses the host's client and the provider's timeout and
     turns a transport failure into a `ProviderError` without its text.
   - By default, send what the provider's own SDK sends, so moving from the SDK changes nothing.
     A caller who wants less passes the native option. What the envelope needs is always asked
     for, and no option turns it off: Exa's page text, Tabstack's metadata, which holds the title.
   - A shared parameter wins over the native option it maps to.
   - Refuse a reserved key: the query, the key, and anything a shared parameter sets.
   - An HTTP status of 400 or more raises `ProviderError(provider, status, tag)`. Take the tag
     from the body only when it looks like a code, and never put body text, the URL, the query or
     the key in an error.
   - An error the provider reports inside a 200 is a value, `SearchError` or `FetchError`.
   - Check every field the adapter reads in a successful answer, so a change on the provider's
     side fails the call rather than degrading it. A field the provider documents as always
     present and missing, a field of the wrong type, or a date that does not parse raises
     `ProviderError(provider, status, "unexpected_response")`, with no body text. Ignore the
     fields the adapter does not read, so a field the provider adds breaks nothing. A body that
     is not the provider's answer at all, not JSON or not an object, is `invalid_response`.
   - A date without a zone is UTC.
   - The cost is the provider's reported figure only, as a `Decimal`, with `cost_source` set.
   - `api_base` replaces the endpoint for every provider; strip its trailing slash.
3. **Record fixtures and test.** Record one fixture per case (below) with
   `uv run python <package>/scripts/record_fixture.py`. Keep them small: ask for few results and
   little text. Check the key is not in them. Then add:
   - the provider's scenarios to `tests/conformance/test_conformance.py`, replaying the fixtures;
   - `tests/test_<name>.py`, for the request it sends, how it reads each fixture, and each field
     it checks missing, of the wrong type and, for a date, unparsable;
   - the provider to `tests/test_registry.py`;
   - the provider to `EXPECTED` in `tests/test_live.py`, saying what its live answers must hold,
     with a new field when the table cannot say it yet, and its key to
     `.github/workflows/otari-libs-live.yml` from a repository secret, which a maintainer adds.

   Run `make test-libs`, `make lint-python` and `make typecheck-python`, then the live tests once.
   CI runs them weekly and on every pull request that changes either library.
4. **Try it from the CLI** against the real account, and note anything surprising. Check that
   `--describe <name>` prints the metadata and options as the documentation gives them.
5. **In the gateway**, nothing in code, since the catalog routes and the settings validation read
   the libraries' metadata. Add the provider's pricing note to `docs/tools.md`, under "Search
   providers" or "Fetch providers", and a compose or `.env.example` line where a key is expected.
6. **Update the learnings** below with what the provider taught, and this procedure if a step
   changed.

The four cases a fixture can record:

| Case | What it is |
|---|---|
| `normal` | A successful answer with content. |
| `empty` | A successful answer with nothing in it: no hits, or a page with no text. |
| `error` | A failed call, with an HTTP status. A key the provider refuses is the easiest to record, unless it refuses keys inside a 200; then try a malformed parameter. |
| `in_body_error` | An error inside a 200. Leave it out when the provider has none. |

## Search checklist (any-search)

- `METADATA`: `max_results` is the provider's own limit; `query_in_url` and `key_in_url` say
  whether either travels in the URL. When one does, pass them as the request's `params`, never
  format the URL yourself or hand it to a logger, and add a test over a real socket (a local
  server) that no log record and no transport error carries the key or the query:
  `httpx.MockTransport` skips httpcore, which logs below httpx.
- Map `max_results`, and also cut the hits to it, since a replayed fixture does not honor it. A
  native count option that stands in for it gets the same cap. Map `time_range` (`day`, `week`,
  `month`, `year`) to the provider's recency filter, rounded to the precision of its dates.
- A hit needs a URL. Drop an item without one only when the provider's documentation allows such
  an item; otherwise it is `unexpected_response`. `title` and `snippet` are strings, never `None`.
  `snippet` is the provider's short text; `text` is page text, when it came back.
- List the fields the adapter checks in the module's docstring and the README's section on the
  provider.
- `raw` holds the provider's item and its whole response, untouched.

## Fetch checklist (any-fetch)

- `METADATA`: `max_urls_per_call`, `renders_javascript` (`False` unless the provider documents
  it, even under an option only, as Tabstack's `effort: max`) and the `formats` its text comes in.
- `max_chars` caps `text`. When the provider takes a length cap, ask for one character more than
  the cap, cut the text, and set `text_truncated` when it was longer; when it has none, cut the
  text it returns. Without `max_chars`, return what the SDK would. `source_truncated` is for a
  fetcher that stops reading the response.
- `url` is the URL asked for, `final_url` the one the provider answered for. `content_type` is
  the type the page was served as, or empty when the provider does not say.
- A page the provider fetched with no text is empty: `text` is `""` and `error` is `None`. An
  answer with neither a page nor a reason raises `unexpected_response`; it is not an empty page.
- A provider with no error inside a 200 either fails the call for a page it could not fetch or
  returns the text of the site's own error page. Find out which live, with a page that does not
  exist, and say so in the docstring, the README and `EXPECTED`'s `page_errors`.
- List the fields the adapter checks in the module's docstring and the README's section on the
  provider.
- `builtin` is not added here: the host registers it (`AnyFetch.register_builtin`).

## Learnings

### 2026-10-08, Exa (iteration one, both libraries)

What each adapter does is in its library's README; these are the lessons for the procedure.

- A provider's API and its SDK can differ in what they send by default. Exa's API returns only
  titles and URLs unless asked for more, while its SDKs ask for page text. Hence step 2's rule:
  follow the SDK.
- Not every provider has an in-body error: Exa's `/search` has none, so it has three fixtures.
- A provider can report an empty page as a page error (Exa's `CRAWL_EMPTY_CONTENT`); read that
  one as an empty page. A page error can also come with no tag at all.
- Check whether the provider's length cap is exact before relying on it. Exa's is, which is what
  makes asking for one character more work.
- Exa dates many pages by their day alone, as midnight, which is why a `time_range` start is
  rounded down to the day.
- A provider can report a cost of 0: Exa did for a search with no results and for a page it could
  not fetch. The host bills the reported cost first.
- Leave out of the options what the envelope cannot carry or the provider has deprecated. For
  Exa: the synthesis fields (`outputSchema`, `systemPrompt`, and `stream`, whose server-sent
  events the adapter cannot read), the enterprise `compliance` mode, and the deprecated fields.
- Fixture recipes: an empty search with `includeDomains` set to a domain that does not exist; an
  in-body fetch error with a page that does not exist; an empty page with a zero-byte file.
- The recorder reads the key from its own environment. When the key is set only in an
  interactive shell's startup file, run the recorder through that shell (`zsh -ic '...'`).
- `make lint-python` runs pre-commit over tracked files only, so stage new files first
  (`git add -N`), or they go unchecked.
- Step 5 for Exa waits for the gateway units that wire the libraries in, since none existed yet.

### 2026-10-13, field checks and the live workflow (both libraries)

- A provider's OpenAPI spec can mark as optional what every recorded answer carries: Exa marks
  `costDollars` optional on both endpoints, and on `/contents` also `results` and `statuses`.
  Follow the spec rather than the recordings: require only what the spec requires or the adapter
  cannot answer without (`results`), take the rest as optional but type-checked, and let the live
  tests' `EXPECTED` require what the host relies on, such as a reported cost, so its absence is
  noticed without failing every call.

### 2026-10-13, Mojeek (iteration two, any-search)

The first search provider with the key in the URL and errors inside a 200.

- A key in the URL: pass the key and the query to `self._http.request` as `params`, never format
  the URL yourself, and set `query_in_url` and `key_in_url`. httpx logs the URL at INFO, which the
  log filter redacts during a provider call. httpcore, below it, logs at DEBUG only the method, the
  host and the response headers, so a redirect's `Location` would be the one way the key reached
  it; Mojeek sent none in any case tried. Only a test over a real socket shows this. Over HTTP/2,
  which a host's client may use, hpack logs every request header at DEBUG, the path included, so
  the filter covers hpack's loggers too.
- Errors inside a 200 can be the only way a provider refuses a key: Mojeek answers a bad key with
  HTTP 200 and `response.status` `Access Denied: invalid key/password`, not the `ERROR: ...` its
  documentation shows. That key records the `in_body_error` fixture; the `error` one came from a
  malformed parameter (`-o s=abc`), which Mojeek answers with a 503.
- Turn a status text into a tag of the adapter's own, by how it starts, with a fallback, so none of
  the provider's text reaches an error.
- A documentation can list fields a default answer never carries. Mojeek's lists `pdate`,
  `timestamp`, `date`, `size` and `cfs` on every result, but its request parameters turn `date` and
  `size` on, and `pdate` never came. Check what a default answer carries, live, before deciding
  which fields are required, and send what the envelope needs (`date=1` for a date).
- A provider can mean "unknown" by a value that parses: Mojeek sends a `timestamp` of 0, the Unix
  epoch, for a page it has no time for. Read the recorded fixtures for such values.
- A provider can cap or refuse quietly: Mojeek returns 10 results for `t=50` on its entry plan,
  and no results with status `OK` for a `since` it does not take (`week`) or for `t=0`. Map the
  shared parameters to values the provider takes, since nothing else will say they were wrong.
- Mojeek reports no cost, so the live tests' `EXPECTED` says so, and says what the host relies on
  instead: dated hits and a snippet, since a hit carries no page text.

### 2026-10-13, Tabstack (the first fetch-only provider)

- The error shape a provider documents may not be what it sends. Tabstack documents
  `{"error": "..."}` for every error, but answers a refused key in plain text, and its sentences
  can quote the URL (`domain resolution failed for <url>`). Record the error fixture to see the
  real shape, and read no tag from a body that holds sentences: the tag is `http_error`.
- A documented format can be wrong for some pages. Tabstack documents `created_at` as ISO 8601,
  but a PDF's comes in the PDF's own format (`D:20240410211143Z`). Before reading a date, fetch a
  PDF and an article live; a field whose format the provider does not hold stays in `raw`, since
  checking it would fail every such call.
- Not every fetch provider has an error inside a 200. Tabstack returns a 404 page's text with a
  200 and does not say what status the site answered, hence `page_errors` in the live tests.
- A provider can be slower than the default 15-second timeout: Tabstack documents up to 15 s at
  its default effort and 15 to 60 s at `max`, and a zero-byte file took 29 s. Say so in the
  option's description, the README and the pricing note, which tells an operator to set the
  instance's `timeout`. The live tests keep the default, which is what hosts use. A slow fixture
  can be recorded with a faster option (`effort: min` for the empty page).
- A provider may report no cost in any answer: Tabstack shows spend only in its console. The
  pricing note in `docs/tools.md` then says what it charges, and no more.
- Fixture recipes: `https://example.com/` for a normal page small enough to keep whole, and the
  key variable set to a made-up key for the error.
