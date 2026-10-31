# Models

Otari routes provider calls through
[any-llm](https://github.com/mozilla-ai/any-llm). Provider support changes with
that dependency, so the running gateway's `GET /api/v1/models` response is more
reliable than a copied provider table.

## Model format

Use `provider:model`:

```text
openai:gpt-5
anthropic:claude-sonnet-4-6
ollama:llama3
```

The prefix selects a provider or named provider instance. Everything after the
first colon is sent as the provider's model ID.

A request may also name the model the way the catalog does. `vendor/model` (`deepseek/deepseek-v4.1-flash`) is the model's catalog id, and Otari picks the provider: among the priced offerings the caller can reach, the organization's own provider key where it offers the model, then the vendor's own provider where it serves the model, then the cheapest. `provider:vendor/model` (`nebius:deepseek/deepseek-v4.1-flash`) pins the provider and lets Otari pick the model id that provider spells it under. See [Catalog spellings](#catalog-spellings) for the rules.

The legacy `provider/model` spelling is still honored where it names an
offering the deployment serves (`openai/gpt-4o` while an `openai` instance
serves `gpt-4o`); read as a catalog id otherwise. Where it is also the catalog
id of a model the catalog also offers elsewhere (an OpenAI model served by
Azure OpenAI too, say), the named offering serves first and the others are
its fallback (see [Catalog spellings](#catalog-spellings)). Prefer the colon form in
configuration, pricing, aliases, and routing policies.

## Configuring a provider

Declare a provider under `providers` in `config.yml`, or add it on the
Hosted providers page. Either way it serves requests and its models are
discovered. Supply the key from the environment with a `${VAR}` reference:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
```

A key reaches a provider only from where a declaration names it: an entry's own
`api_key` (a `${VAR}` reference included), a hosted provider's key, or
an organization's own provider key, which declares the provider for that
organization's requests. A provider's native variable (such as `OPENAI_API_KEY`)
is never read for it otherwise, so an entry with no `api_key`, or one carrying
only `client_args`, has no credential. A request it cannot serve is refused with
a `424` and the `provider_not_configured` code. Otherwise a variable present on
the server would serve every caller that has no credential of its own.

Keyless local backends (`ollama`, `llamacpp`, `llamafile`, `lmstudio`, `vllm`)
and providers authenticating from ambient cloud credentials (`bedrock`,
`sagemaker`, `vertexai`) need a declaration too, but no key: a bare entry
declares one. Undeclared, they are refused like any other provider, because the
local server and the cloud role belong to the deployment.

```yaml
providers:
  ollama:
  bedrock:
```

The one exception is a build whose extension serves the request on a credential
the deployment owns (`ModelProviderPort`, asked only after every other source
missed); a plain deployment has none. The gateway logs a warning at startup
naming each `<PROVIDER>_API_KEY` variable that is set and that no entry's
`api_key` reads, and ignores an entry for a keyed provider that names no key.

Provider support is endpoint-specific. A provider that supports chat may not
support Responses, images, audio, rerank, or batches. Unsupported combinations
are rejected before dispatch when Otari can identify them.

## Named provider instances

Use a named instance when several backends share one provider implementation or
when a self-hosted server speaks a compatible protocol:

```yaml
providers:
  home_lab:
    provider_type: openai
    api_base: "https://models.example.com/v1"
    api_key: ${HOME_LAB_TOKEN}
    models:
      - qwen3-32b
```

Call this model as `home_lab:qwen3-32b`. Pricing and usage also use the instance
name. `provider_type: openai-compatible` and `openai_compatible` are accepted
aliases for the OpenAI implementation.

The optional `models` list supplies discovery for a backend that has no
model-listing endpoint. It does not restrict direct dispatch.

A workspace or a single user can also bring its own server, reached the same
way (`my-vllm:qwen3`) but visible only to its owner. See
[Provider endpoints](provider-endpoints.md).

## Local providers

Ollama, llama.cpp, and llamafile can run without an API key. Add a provider entry
when you want Otari to discover their models:

```yaml
providers:
  ollama:
    api_base: "http://gpu-box:11434"
```

Without the entry, a direct selector can still work if the backend is reachable
at the provider's default address. Otari does not probe local servers merely
because their provider exists in any-llm.

## Model discovery

`GET /api/v1/models` combines discoverable provider models, configured prices,
aliases, and routing-policy names, and lists each model once however many
providers serve it (see [Listing available models](#listing-available-models)). A provider's embedding and rerank models are
offered alongside its chat models. A provider that pages its listing, as Cohere
does, is read to the last page.
Discovery is cached and bounded; an unreachable provider does not block the
catalog indefinitely.

A [hosted provider](hosted-providers.md#hosted-providers-and-byo-credentials) lists its
roster, the models switched on under it, rather than everything its upstream
lists. The operator's `GET /api/v1/models/discoverable` and provider health
dial it on its stored key beside the configured instances. A hosted provider
serves only its roster unless it serves models not on its roster
(`serves_unlisted_models`, off by default): it then also serves any model a
request names, the way a `providers:` instance does, switches on every model
it offers whether or not something prices it, and still refuses a model its
roster switches off. A provider moved from the retired Deployment providers page
starts with it off and every model it lists switched on.

Each listed model carries the rate a request from the caller would be billed at,
and `pricing_source` says which price list it came from: `organization` for the
caller's organization's own override, `configured` for the deployment's stored
rate, `default` for the models.dev fallback, `dynamic` for a routing policy
that picks its model per request, and `none` for an unpriced model. An alias is
priced from its target.

The two surfaces differ on a value the gateway does not know.
`GET /api/v1/models` and `GET /api/v1/models/{model_id}` leave the field out
altogether, including a nested one such as a per-tier rate override. The grouped
catalog under `GET /api/v1/catalog/models` sends `null` instead. Both model
routes always send `pricing_source`, where `none` says nobody has priced the
model. The grouped catalog's offering carries `price_source`, which is `null`
when nothing prices it.

Each model carries its token limits, `context_window` and `max_output_tokens`,
so a client does not have to guess them. Both come from models.dev's entry for
the model under the provider serving it. Where that entry has no context window,
the bundled or accepted models.dev pricing snapshot supplies one; the snapshot
keeps no output limit, so a model the live catalog does not describe reports no
`max_output_tokens`. The same is true of any limit neither source knows. An alias reports its target's
limits, and the grouped catalog reports the same values for each offering. An
instance declared `openai-compatible` is looked up under OpenAI, so a model it
serves under another name usually has no models.dev entry. With
`models_dev_metadata: false`, the context window comes from the pricing snapshot alone.

Each model also says what it accepts, in the fields OpenRouter's model listing
uses, so a client built for that shape reads them unchanged. `name` is models.dev's
display name. `supported_parameters` lists `tools` and `tool_choice` for a model
that calls tools, `response_format` for one with structured output, and
`reasoning` for a reasoning model; it names only the parameters models.dev records
a flag for, not every parameter the model takes. `architecture` carries
`input_modalities` and `output_modalities`, such as `["text", "image"]`. All three
are unset for a model models.dev does not describe, which means unknown rather
than unsupported. An alias reports its target's capabilities but no `name`,
because the target's name would say what the alias points at.

Set `model_discovery: false` to publish a curated catalog made from aliases and
explicitly priced models. For a backend with no listing API, use the instance's
`models` list.

### Who is shown which models

An API key is shown the models its allow-list permits, so the catalog never
advertises a model that would be refused at inference. A dashboard session is
answered the same way, from its membership: a caller who operates the deployment
sees the whole catalog, and anyone else sees every `providers:` instance, which
is deployment-wide, plus the models their own organization's provider keys
reach, narrowed by any workspace model restriction. A deployment whose providers
all come from `config.yml` therefore shows every tenant the same catalog it
always did.

The models an organization's own key reaches are the ones offered on it, which
is what the Providers page records when it pulls a key's catalog. Each carries a
Serving switch, and a model switched off leaves the catalog and is refused at
dispatch, because both read one allow-list. A key nobody has refreshed offers no
rows at all, and that is not the same as offering none: it means the key is
unnarrowed and reaches whatever its provider serves. Offering none, which is
every model switched off, serves nothing.

A model nothing prices is offered switched off while `require_pricing` is on,
and the switch refuses to turn it on until it has a rate, so a model the
pricing data has not caught up with cannot be billed at nothing. With
`require_pricing: false` the deployment serves unpriced traffic by choice, so
such a model is offered switched on and served at no cost, like an unpriced
model of a `providers:` instance (see
[Default pricing](configuration.md#default-pricing)).

A refusal on that path says which narrowing turned the model away. A model the
key offers with its switch off is a 403 with the code `model_not_serving`, naming
the key and pointing at the switch. A model no key of the provider offers is a
404 with `model_not_found`. An undated name whose dated builds the key offers is
answered for those builds: not serving where every one is switched off, and not
found, naming them, where there is more than one to choose from (see
[Catalog spellings](#catalog-spellings)). A model the key serves that a workspace restriction
or the API key's own allow-list excludes keeps the 403 `model_not_allowed`. The
Providers page shows each key's serving count beside it, so a key offering models
and serving none is visible before the first request. See
[Error codes](api-reference.md#error-codes).

Aliases and stored routing policies are workspace-scoped rows, and the catalog
reads them for a workspace rather than filtering them by target, so a name alone
would cross a tenant boundary that the allow-list cannot see. A session is
therefore shown that layer only where the workspace it comes from is one the
caller may see, which on a single-tenant deployment is everyone in it. An API key
is unaffected: it names its own workspace.

## Capabilities

`model_capabilities` can correct image and PDF support when a compatible
provider reports capability at the protocol level rather than for each model:

```yaml
model_capabilities:
  "ollama:qwen2-vl":
    supports_image: true
    supports_pdf: false
```

File handling uses these values to decide whether to pass an attachment through
or normalize it for a text-only model. See [Files](files.md).

## Prompt caching

Mark the prompt prefix to cache the way Anthropic does, with `cache_control`
(`{"type": "ephemeral"}`, optionally `"ttl": "1h"`): on a system block or content
block on `/api/v1/messages`, or on a message or content part on
`/api/v1/chat/completions`. Anthropic models read it natively. For Gemini, any-llm
turns the last breakpoint into a Gemini context cache holding everything up to
it, reuses that cache while it lives (5 minutes, or 1 hour with `"ttl": "1h"`),
and falls back to an uncached request when Gemini refuses a prefix below its
minimum size. Gemini 2.5 and later models also cache repeated prefixes on their
own, with nothing marked.

Either way, the usage reports the tokens written to and read from the cache, and
Otari bills them at the model's `cache_write_price_per_million` and
`cache_read_price_per_million` when the pricing sets them, and at the input price
otherwise. On either route, a Claude model whose cache writes are priced bills a
1-hour write at twice its input price, as Anthropic does, on a `cache_write_1h`
charge line. The time a Gemini cache is stored is not billed separately.

## Model aliases

An alias gives callers a stable name for one real selector:

```yaml
aliases:
  fast: openai:gpt-5-mini
  local: home_lab:qwen3-32b
```

Callers send the alias in `model`. Completion responses keep the alias while
pricing, budgets, and usage use the resolved target, which the gateway
names in the `Otari-Provider` and `Otari-Model` response headers. Configure the
target's price, not a price under the alias.

An alias also withholds its target offering from `GET /api/v1/models` in that
workspace, where the alias is listed in its place; the model stays listed while
another provider serves it. This does not apply to routing-policy
targets. The grouped catalog (`GET /api/v1/catalog/models` and the Models page)
lists real models only and never an alias, so an aliased model stays in it.

Aliases are useful for a curated catalog. A routing policy is the broader form
when a name needs conditions, failover, weighting, learned selection, or
mandatory guardrails. See [Routing](routing.md).

Alias names cannot contain `:` or `/`, collide with provider instances, or
point to another alias.

### Runtime aliases

Operators can manage aliases from Routing or `/api/v1/aliases` without
restarting. A stored alias belongs to a workspace and can optionally be narrowed
to one user. Resolution prefers the most specific applicable alias.

The authenticating API key determines the workspace. A user-scoped alias applies
to all of that user's keys in the workspace. Config-file aliases are
deployment-wide and apply in every workspace.

Use the generated OpenAPI specification for management request shapes and
scoping parameters.

## Routing-policy model names

A static one-target policy resolves anywhere an alias does. A conditional,
weighted, or learned policy applies only to Chat Completions, Messages, and
Responses because other endpoints do not provide the request context needed to
choose a candidate.

Policies appear as model names, but dynamic policies have no single price.
Price their concrete candidates. Unlike aliases, policies do not hide candidate
models from the catalog.

## The catalog, grouped by model

`GET /api/v1/catalog/models` reads the same merged catalog as `GET /api/v1/models` and
folds it by model the same way, so `nebius:zai-org/GLM-5.3` and
`fireworks:accounts/fireworks/models/glm-5p3` are two offerings of one entry.
`?at_context=<tokens>` on the list takes each model's minimum from the pricing
tier a request of that size would settle at, so tiered offerings compare at
the size that matters rather than at their base rate.
`GET /api/v1/catalog/models/{id}` lists every offering of one model the caller may
use, cheapest first, with each provider's context and output limits and the
price the caller's organization would be charged, labeled by which price list
it came from: the organization's own override, the deployment's stored rate, or
the models.dev default. Both routes accept the same credentials as
`GET /api/v1/models`, and a model the caller may not use answers 404.

Grouping keys on the models.dev display name where the dataset knows the
model, and on the provider's id with its path prefixes, org segment and version
pins removed where it does not; a vendor's name written in front of its own
model (`NVIDIA Nemotron 3 Ultra`, `openai-gpt-oss-120b`) is dropped from both.
Bedrock's cross-region inference profiles of one model, such as `us.moonshotai.kimi-k3` and `global.moonshotai.kimi-k3`, are offerings of that one model: the profile prefix is dropped from the id, and the region models.dev puts in the profile's name, as in `Kimi K3 (US)`, is dropped from the name. A
model's id is its vendor and its name, `z-ai/glm-5.3`, or the bare name where
nobody could say the vendor. The vendor is the org of models.dev's
`canonical_model_id` where the dataset has one, else the org segment of the
provider's id, else the model family (`claude`, `nemotron`). A dated build, a size or tier, and a mode a
reseller exposes as its own id stay separate models. models.dev's description,
capabilities and modalities are served to every catalog reader here, where
`GET /api/v1/models/metadata` stays operator-only.

Each offering also carries the provider's list price from the live models.dev
catalog, where it has one. The default is billed from the accepted models.dev
snapshot, which can lag that catalog, so a difference between the two means a
price update is waiting for review. Each offering also carries, for a signed-in
caller, the organization's last thirty days on that offering: requests, cache
hit rate, and the effective price per million tokens after cache reads and
tiers.

### Browsing the grouped catalog

The Models page requests one page at a time from `GET /api/v1/catalog/models`.
Filters and sorting apply to the caller's full authorized catalog before `skip`
and `limit` select a page. `count` is the number of matching models before paging;
`limit` defaults to 100 and is capped at 1,000.

Repeat `provider` or `vendor` to match any selected value. An empty `vendor`
selects models whose vendor is unknown. Repeat `input_modality`,
`output_modality`, or `capability` to require every selected value. The endpoint
also accepts `search`, `min_context`, `max_input`, `pricing`, `source`, and
`released_within_days`. Release windows end today in UTC and exclude unknown
and future dates. A price ceiling excludes models with no known input rate.

`sort` accepts `name`, `released`, `input`, `output`, `context`, or `providers`,
with `direction=asc` or `desc`. Unknown values sort last in either direction;
names and catalog ids break ties so pages have a stable order.

Request `include_facets=true` to receive the provider and vendor choices from
the complete authorized catalog, whatever the filters match, so a filter that
matches nothing can still be undone. `facets.total_count` counts authorized
models before filtering. `facets` is null unless requested, and never names a
model or provider the caller cannot access.

### Catalog spellings

A provider's own id can be long, so the gateway also accepts the two spellings
the catalog shows. Each is relabeled like an alias, so a response's `model` is
what was sent, and pricing, budgets and usage key on the offering reached.

- The model's catalog id (`z-ai/glm-5.3`, `deepseek/deepseek-v4.1-flash`) resolves to the first of the model's offerings in this order. Priced offerings come before unpriced ones, because a request an unpriced offering serves costs nothing and so escapes budgets. Among the priced ones, the organization's own provider key comes first, even where another offering is cheaper. Next is the vendor's own provider, where the id's vendor is also a provider (`openai/gpt-4o`), because the caller who names OpenAI's model while OpenAI is configured means OpenAI's price. Then come the other offerings, cheapest first at the caller's rates, with ties in A-Z order. To make an unpriced offering lead, such as an organization's own key or a local model, give it a price, even 0.
- `instance:<catalog id>` (`nebius:deepseek/deepseek-v4.1-flash`) pins the instance and resolves to the model's first offering there by the same order, never elsewhere. This is the spelling the catalog shows as an offering's `short_selector`.
- On an organization's own provider key, a provider's undated name (`anthropic:claude-haiku-4-5`) resolves to the dated build the key serves under it (`claude-haiku-4-5-20251001`), where exactly one serving model is that name plus a release date (`-YYYYMMDD` or `-YYYY-MM-DD`). The dated build's serving switch, price and allow-lists apply as if it had been sent. A name with two dated builds, at least one of them serving, is refused as not found (404 `model_not_found`), and the detail names both; a name whose dated builds are all switched off is refused as not serving (403 `model_not_serving`). Neither detail names the dated builds to a caller who reached the model through an alias. A selector whose prefix names a `providers:` instance is never rewritten this way, even for a caller whose organization also holds a key for that provider: the instance sends any id to the provider as written.

A selector that already names an offering is never rewritten, so a provider's
own id keeps working verbatim.

The index behind this is rebuilt every minute and on
`POST /api/v1/catalog/selectors/refresh`, an operator call. It holds the
deployment's catalog view, priced from the deployment's list and the defaults,
and one view per organization for the models it offers on its own provider
keys, priced at that organization's rates. An organization's view answers only
its own callers: a model one tenant reaches through its key is never where
another tenant's selector lands, and a model nobody offers to the caller
resolves as the deployment's view says. A model's `selector` in the catalog is
null where the caller has no such spelling for it.

A request that names a catalog id with more than one offering falls back across them. When the first offering fails before it sends a response, the next one in the same order serves the request, as a routing policy's `on_failure` list does. A catalog id spelled like the legacy `provider/model` of one of its own offerings (`openai/gpt-4o` while the `openai` instance serves `gpt-4o`) tries that offering first and the others after it in the usual order. The colon spelling of an offering (`openai:gpt-4o`) never falls back, so it is the way to pin one provider. The offerings a key may not use are left out first, so a key whose allow-list names one offering reaches it by the catalog id. A plan tries at most five offerings. Usage rows record the catalog id as `policy_name`, `catalog` as the first offering's `selection_reason`, and `on_failure` for the ones after it.

The Routing page's "Automatic fallback" section, read from
`GET /api/v1/organizations/me/automatic-fallbacks`, lists every model your
organization sent requests to in the last 30 days, most requested first, with
its offerings in the order a request sent by catalog id tries them for your
organization. Beside each it counts the requests (an absorbed attempt is not one), the
successful ones a later offering served, and the ones that named one offering
or a policy rather than the catalog id and so had no automatic fallback. A model
most of whose requests named one provider shows that provider alone, marked "No
backup · pinned", because those requests get no fallback until they send the
catalog id. Under each provider it says who runs the built-in tools: "Own
tools" where the provider runs at least one itself, "Otari's tools" where Otari
runs them all; the response's `own_tools` and `otari_tools` name which tool is
whose. That assumes a caller who declares
the tools in the provider's own words and sends no `Otari-*` executor header,
so the deployment's code execution executor and web search interception decide;
a workspace's code execution pin is not reflected. The order is the catalog's,
before a key's allow-list or a missing credential leaves an offering out of a
request's plan. An owner or admin reads every workspace in the organization,
anyone else the workspaces they belong to.

The dashboard's Models page is this catalog: one card per model, with a rail
of filters beside it, and a page per model with its facts and every offering
compared in a table. "Use this model" opens a drawer beside the table with
the request to copy, sent to the gateway's pick or to a provider pinned by
its selector. It is read-only; an organization's own rate is set on Providers,
which an offering on one of that organization's keys links to. An offering the
deployment supplies the credential for says so instead, because its rate is the
deployment price list's rather than a tenant's. A metered rate that differs from the provider's
list price is marked with the list price.

With `public_catalog: true` (see [Configuration](configuration.md)), the same
two routes and the same page are served to a visitor with no session, at the
deployment's rates and for the models the deployment itself serves, its
hosted providers included. A visitor's
"Use this model" opens account creation instead of the drawer (sign-in where
signup is closed), and the dashboard reopens that model on their first sign-in.

## Listing available models

```bash
curl http://localhost:8000/api/v1/models \
  -H "Authorization: Bearer $OTARI_API_KEY"
```

The listing has one entry per model, and its `id` is the model's catalog id
(`z-ai/glm-5.3`, `openai/gpt-4o`), not a `provider:model` selector. Send that id
as `model` and Otari chooses the provider: it tries the model's offerings best
first (priced before unpriced, then your organization's own key, then the
model's own vendor, then the cheapest) and falls back to the next when one
fails. A catalog id spelled like the legacy `provider/model` of one of its own
offerings (`openai/gpt-4o` while the `openai` instance serves `gpt-4o`) tries that
offering first (see [Catalog spellings](#catalog-spellings)). An entry's price, limits, capabilities and `deployment_managed` are those
of the offering tried first, and `owned_by` is the model's vendor.

To pin a request to one provider, send its `provider:model` selector, which
still works everywhere it did. `?provider=<instance>` lists only the models that
instance serves, described as it serves them, and
`GET /api/v1/catalog/models/{id}` lists every offering of one model.
`GET /api/v1/models/{id}` answers for a catalog id, an alias, or a selector.

On a deployment that hosts at least one provider, an organization API key is
shown a priced model only where something would
serve it from the key's workspace: a configured instance, the organization's own
provider key, a hosted provider the deployment serves the organization, or the
provider's own environment variable. A hosted provider the operator switched
off or deleted leaves its rates on the price list, and its models leave this
listing. The master key and an operator session still list the whole price
list.

## HuggingFace provider routing

A HuggingFace model may be served by several inference providers. Pin the backend
when deterministic routing and pricing matter:

```text
huggingface:zai-org/GLM-4.6:together
huggingface:zai-org/GLM-4.6:novita
```

Otari splits only the first colon, so the backend suffix reaches HuggingFace
unchanged. Auto, cheapest, or fastest routing cannot be assigned one reliable
backend price; configure pinned selectors when enforcing budgets.
