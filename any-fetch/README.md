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

## What comes back

A `FetchedPage`: the `url` asked for, the `final_url` after redirects, the `title`, the `text`, the
`content_type`, `published`, the `cost` in USD the provider reported with `cost_source`
(`reported` or `none`), an `error` when the provider signaled one inside a successful response,
`source_truncated` when the response body hit the fetcher's size limit, `text_truncated` when the
text hit its own, and the provider's own response as `raw`, which is left out of `repr`.

A failed call raises `ProviderError` with the provider's name, a `tag` and the HTTP `status` when
there is one. A missing key raises `MissingCredentialError`. All errors derive from
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
INFO. Every provider call, `builtin` included, runs with a context variable set, and a filter
installed on the `httpx` logger when the package is imported replaces the URL with `<redacted>` in
records emitted during one. Requests made by other code in the process are left alone.

## The command line

```bash
any-fetch fake https://www.python.org/downloads/ --max-chars 200
any-fetch fake https://www.python.org/downloads/ --json
any-fetch fake https://www.python.org/downloads/ -o error=http_status -o error_status=404
```

Keys come from the environment. `-o NAME=VALUE` passes a native option, its value read as JSON
when it parses. `--json` prints the page as JSON without `raw`, `--raw` prints the provider's own
response, and both together print the page with `raw` kept. `builtin` needs a host, so it fails
from the command line.

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
