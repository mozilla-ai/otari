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

A request may also name the model the way the catalog does. `vendor/model`
(`deepseek/deepseek-v4.1-flash`) is the model's catalog id, and Otari picks the
provider: the vendor's own where it serves the model, otherwise the cheapest
offering the caller can reach, including one on the organization's own
provider key. `provider:vendor/model` (`nebius:deepseek/deepseek-v4.1-flash`)
pins the provider and lets Otari pick the model id that provider spells it
under. See [Catalog spellings](#catalog-spellings) for the rules.

The legacy `provider/model` spelling is still honored where it names an
offering the deployment serves (`openai/gpt-4o` while an `openai` instance
serves `gpt-4o`); read as a catalog id otherwise. Prefer the colon form in
standalone configuration, pricing, aliases, and routing policies.

## Configuring a provider

A provider can come from `config.yml`, its native credential environment
variable, or the standalone Providers page:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
```

A native variable such as `OPENAI_API_KEY` can be enough to dispatch a direct
request. Add the provider to `providers` when you also want model discovery or
explicit client settings.

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

Named instances are local configuration and do not apply in hybrid mode, where
the control plane resolves each attempt.

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
aliases, and routing-policy names. Discovery is cached and bounded; an
unreachable provider does not block the catalog indefinitely.

Set `model_discovery: false` to publish a curated catalog made from aliases and
explicitly priced models. For a backend with no listing API, use the instance's
`models` list.

Hosted mode keeps `GET /api/v1/models` for control-plane discovery. Hybrid mode
does not serve the local catalog.

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

## Model aliases

An alias gives callers a stable name for one real selector:

```yaml
aliases:
  fast: openai:gpt-5-mini
  local: home_lab:qwen3-32b
```

Callers send the alias in `model`. Completion responses keep the alias while
pricing, budgets, and usage use the resolved target. Configure the target's
price, not a price under the alias.

An alias also withholds its target from model listings in that workspace. This
does not apply to routing-policy targets.

Aliases are useful for a curated catalog. A routing policy is the broader form
when a name needs conditions, failover, weighting, learned selection, or
mandatory guardrails. See [Routing](routing.md).

Alias names cannot contain `:` or `/`, collide with provider instances, or
point to another alias.

### Runtime aliases

Standalone operators can manage aliases from Routing or `/api/v1/aliases` without
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
folds it by model, so `nebius:zai-org/GLM-5.3` and
`fireworks:accounts/fireworks/models/glm-5p3` are two offerings of one entry.
`?at_context=<tokens>` on the list takes each model's minimum from the pricing
tier a request of that size would settle at, so tiered offerings compare at
the size that matters rather than at their base rate.
`GET /api/v1/catalog/models/{id}` lists every offering of one model the caller may
use, cheapest first, with each provider's context and output limits and the
price the caller's organization would be charged, labeled by which price list
it came from: the organization's own override, the deployment's stored rate, or
the genai-prices default. Both routes accept the same credentials as
`GET /api/v1/models`, and a model the caller may not use answers 404.

Grouping keys on the models.dev display name where the dataset knows the
model, and on the provider's id with its path prefixes, org segment and version
pins removed where it does not. A model's id is its vendor and its name,
`z-ai/glm-5.3`, or the bare name where nobody could say the vendor. A dated build, a size or tier, and a mode a
reseller exposes as its own id stay separate models. models.dev's description,
capabilities and modalities are served to every catalog reader here, where
`GET /api/v1/models/metadata` stays operator-only.

Each offering also carries the provider's own list price from models.dev,
where it has one, and for a signed-in caller the organization's last thirty
days on that offering: requests, cache hit rate, and the effective price per
million tokens after cache reads and tiers.

### Catalog spellings

A provider's own id can be long, so the gateway also accepts the two spellings
the catalog shows. Each is relabeled like an alias, so a response's `model` is
what was sent, and pricing, budgets and usage key on the offering reached.

- The model's catalog id (`z-ai/glm-5.3`, `deepseek/deepseek-v4.1-flash`)
  resolves to the model's cheapest priced offering, at the caller's rates.
  Where the id's vendor is also a provider (`openai/gpt-4o`), that provider's
  own offerings win while it serves the model, because the caller who names
  OpenAI's model while OpenAI is configured means OpenAI's price; where it
  serves nothing, the model is reached through whoever resells it.
- `instance:<catalog id>` (`nebius:deepseek/deepseek-v4.1-flash`) pins the
  instance and resolves to the model's cheapest offering there, never
  elsewhere. This is the spelling the catalog shows as an offering's
  `short_selector`.

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
null where the caller has no such spelling for it. A key whose allow-list names
some instances only should send one of those instances, a pinned spelling or
an alias, since a bare catalog id resolves before the allow-list is consulted.

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
deployment's rates and for the models the deployment itself serves: the
configured providers, and on a managed platform its hosted models. A visitor's
"Use this model" opens account creation instead of the drawer (sign-in where
signup is closed), and the dashboard reopens that model on their first sign-in.

## Listing available models

```bash
curl http://localhost:8000/api/v1/models \
  -H "Authorization: Bearer $OTARI_API_KEY"
```

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
