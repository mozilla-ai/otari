# Models

Otari routes provider calls through
[any-llm](https://github.com/mozilla-ai/any-llm). Provider support changes with
that dependency, so the running gateway's `GET /v1/models` response is more
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

`provider/model` is also accepted on completion routes for compatibility with
otari.ai. Prefer the colon form in standalone configuration, pricing, aliases,
and routing policies.

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
`/v1/models` endpoint. It does not restrict direct dispatch.

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

`GET /v1/models` combines discoverable provider models, configured prices,
aliases, and routing-policy names. Discovery is cached and bounded; an
unreachable provider does not block the catalog indefinitely.

Set `model_discovery: false` to publish a curated catalog made from aliases and
explicitly priced models. For a backend with no listing API, use the instance's
`models` list.

Hosted mode keeps `GET /v1/models` for control-plane discovery. Hybrid mode
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

Standalone operators can manage aliases from Routing or `/v1/aliases` without
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

## Listing available models

```bash
curl http://localhost:8000/v1/models \
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
