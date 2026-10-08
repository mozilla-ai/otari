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
   a key or a base URL is required, the default base URL, the limits, and the native options
   under the provider's own names and enums. Note the pricing, with the date you read it, in the
   module's docstring.
2. **Write the adapter.** Build the request from the shared parameters and the native options,
   read the response into the envelope, and write the date parser, the error reader and the cost
   reader. The rules:
   - Send with `self._http.request`, which uses the host's client and the provider's timeout and
     turns a transport failure into a `ProviderError` without its text.
   - By default, send what the provider's own SDK sends, so moving from the SDK changes nothing.
     A caller who wants less passes the native option.
   - A shared parameter wins over the native option it maps to.
   - Refuse a reserved key: the query, the key, and anything a shared parameter sets.
   - An HTTP status of 400 or more raises `ProviderError(provider, status, tag)`. Take the tag
     from the body only when it looks like a code, and never put body text, the URL, the query or
     the key in an error.
   - An error the provider reports inside a 200 is a value, `SearchError` or `FetchError`.
   - A date without a zone is UTC; one that does not parse is `None`.
   - The cost is the provider's reported figure only, as a `Decimal`, with `cost_source` set.
   - `api_base` replaces the endpoint for every provider; strip its trailing slash.
3. **Record fixtures and test.** Record one fixture per case (below) with
   `uv run python <package>/scripts/record_fixture.py`. Keep them small: ask for few results and
   little text. Check the key is not in them. Then add:
   - the provider's scenarios to `tests/conformance/test_conformance.py`, replaying the fixtures;
   - `tests/test_<name>.py`, for the request it sends and how it reads each fixture;
   - the provider to `tests/test_registry.py`.

   Run `make test-libs`, `make lint-python` and `make typecheck-python`, then the live tests once.
4. **Try it from the CLI** against the real account, and note anything surprising.
5. **In the gateway**, nothing in code, since the catalog routes and the settings validation read
   the libraries' metadata. Add the provider's pricing note to `docs/tools.md`, and a compose or
   `.env.example` line where a key is expected.
6. **Update the learnings** below with what the provider taught, and this procedure if a step
   changed.

The four cases a fixture can record:

| Case | What it is |
|---|---|
| `normal` | A successful answer with content. |
| `empty` | A successful answer with nothing in it: no hits, or a page with no text. |
| `error` | A failed call. A key the provider refuses is the easiest to record. |
| `in_body_error` | An error inside a 200. Leave it out when the provider has none. |

## Search checklist (any-search)

- `METADATA`: `max_results` is the provider's own limit; `query_in_url` and `key_in_url` say
  whether either travels in the URL. When one does, build the URL yourself and never pass it to a
  logger.
- Map `max_results`, and also cut the hits to it, since a replayed fixture does not honor it.
  Map `time_range` (`day`, `week`, `month`, `year`) to the provider's recency filter.
- A hit needs a URL; drop items without one. `title` and `snippet` are strings, never `None`.
  `snippet` is the provider's short text; `text` is page text, when it came back.
- `raw` holds the provider's item and its whole response, untouched.

## Fetch checklist (any-fetch)

- `METADATA`: `max_urls_per_call`, `renders_javascript` (`False` unless the provider documents
  it) and the `formats` its text comes in.
- `max_chars` caps `text`. Ask the provider for one character more than the cap, cut the text,
  and set `text_truncated` when it was longer. `source_truncated` is for a fetcher that stops
  reading the response.
- `url` is the URL asked for, `final_url` the one the provider answered for. `content_type` is
  the type the page was served as, or empty when the provider does not say.
- A page the provider fetched with no text is empty: `text` is `""` and `error` is `None`.
- `builtin` is not added here: the host registers it (`AnyFetch.register_builtin`).

## Learnings

### 2026-10-08, Exa (iteration one, both libraries)

- Exa's API and its SDK differ. With no `contents`, `POST /search` returns only `id`, `title` and
  `url`, while the SDKs ask for page text up to 10,000 characters. The adapters follow the SDK,
  hence step 2's rule. Highlights fill `snippet` only when a caller asks for them;
  `highlights: true` gave up to about 3,400 characters per hit.
- `/search` reports no error inside a 200 (Exa's error-codes page says so), so any-search has
  three Exa fixtures and any-fetch four.
- `/contents` reports a page with no text as a page error, `CRAWL_EMPTY_CONTENT`, with
  `httpStatusCode` 500. The adapter reads it as an empty page. A page error can come with an
  empty `error` object; it becomes the tag `fetch_error`.
- Exa's `text.maxCharacters` is exact, which is what makes asking for one more character work.
- Exa reported a cost of 0 for a search with no results and for a page it could not fetch.
- Exa's `type` enum is `instant`, `fast`, `auto`, `deep-lite`, `deep` and `deep-reasoning`;
  `neural` and `keyword` are gone. Left out of the options: the deprecated fields, the synthesis
  fields (`outputSchema`, `systemPrompt`, and `stream`, whose server-sent events the adapter
  cannot read) and the enterprise `compliance` mode.
- Exa has no environment variable for its base URL, so `env_api_base` is `None`.
- Fixture recipes: an empty search with `includeDomains` set to a domain that does not exist; an
  in-body fetch error with a page that does not exist; an empty page with a zero-byte file.
- The recorder reads the key from its own environment. When the key is set only in an
  interactive shell's startup file, run the recorder through that shell (`zsh -ic '...'`).
- `make lint-python` runs pre-commit over tracked files only, so stage new files first
  (`git add -N`), or they go unchecked.
- Step 5 for Exa waits for the gateway units that wire the libraries in, since none existed yet.
