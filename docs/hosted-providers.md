# Hosted providers and BYO credentials

Every provider call is made on one credential. This page describes where that
credential comes from: the deployment's own configuration, an organization's
own provider key (bring your own key, BYO), or one of the deployment's
**hosted providers**, a credential the deployment owns and pays for. For the
modes these apply in, see [Runtime modes](modes.md).

## The credential ladder

A request is served on the first credential the ladder finds, checked per
candidate model in this order:

| Rung | Credential | Reached by | Managed by |
| --- | --- | --- | --- |
| 1 | An [owned provider endpoint](provider-endpoints.md) of the caller's workspace or user | `<name>:<model>`, on the completion routes only | The workspace or user |
| 2 | A `providers:` instance in `config.yml` | `instance:model`, and a bare `provider:model` whose provider spells the instance's name | The deployment, in config |
| 3 | The organization's own provider key | A bare `provider:model` that no instance answers | Organization owners and admins, on the Providers page |
| 4 | A hosted provider | A bare `provider:model` once rungs 2 and 3 miss, or a hosted provider's own name | Deployment operators, on the Hosted providers page |

A request no rung serves is refused with `424` and the code
`provider_not_configured`. A provider's native environment variable (such as
`OPENAI_API_KEY`) is never a rung; see
[Configuring a provider](models.md#configuring-a-provider).

### Organization provider keys

Where an organization holds several keys for one provider, a workspace uses
the key it set to "Always use this key", otherwise the organization's default
for the provider (the star on the Providers page; clicking a filled star
removes it), otherwise the organization's oldest key. A key a workspace set to
"Never use this key" is passed over at every step.

The models a key serves, its serving switches and what a refusal on that path
returns are described in
[Who is shown which models](models.md#who-is-shown-which-models).

### Hosted provider names

A hosted provider is named after its provider by default. A name of its own,
such as `openai-eu`, lets two endpoints of one provider be served side by side.

- A hosted provider named after its provider is reached by a bare
  `provider:model` request once neither a `providers:` instance nor the
  organization's own key answers.
- One with a name of its own is reached only by that name (`openai-eu:gpt-4o`).
  An organization's key for the provider never serves it, because that key
  belongs to another endpoint.
- For a provider that needs no key, only the deployment's hosted provider of
  the same name can answer.
- A hosted provider restricted to one organization does not exist for every
  other organization. A request whose workspace belongs to no organization is
  served only on a hosted provider restricted to no organization.

The name is what pricing, budgets and usage are keyed on. A name may not
contain `:` or `/`, may not be `otari` or `hosted`, and may not be another
implementation's name.

## Who manages hosted providers

Only a deployment operator manages hosted providers: the master key, the
bootstrap operator's session, or the session of an account granted operator
access. That covers every route under `/api/v1/hosted-providers`, reads
included. Any other signed-in account, whatever its role in whichever
organization, gets the same `404` that `/api/v1/admin` gives, and an API key
gets `401`. Organization owners and admins manage their own organization's
provider keys, never these.

A standalone or hosted deployment configures hosted providers in three places:

- the Hosted providers page (see [The Hosted providers page](#the-hosted-providers-page)),
- `/api/v1/hosted-providers`,
- the `hosted_providers:` block of `config.yml` (see
  [The `hosted_providers:` block](#the-hosted_providers-block)).

A provider stored through the page or the API wins over a `config.yml` entry of
the same name. A hybrid gateway ignores the block and holds no hosted providers
of its own.

Storing a hosted provider's key requires `OTARI_SECRET_KEY`; see
[Runtime provider management](configuration.md#runtime-provider-management)
for generating and rotating it. The API never returns a hosted provider's key,
only its last four characters, and it masks credentials in `client_args` and in
the base URL.

## Roster and serving

A hosted provider carries a roster: the models offered on it, each with a
serving switch and the rate it is charged at on the deployment price list.

| Situation | Result |
| --- | --- |
| Model on the roster, switched on | Served on the hosted credential |
| Model on the roster, switched off | Refused with `424` |
| Model not on the roster, `serves_unlisted_models` off (default) | Refused with `424` |
| Model not on the roster, `serves_unlisted_models` on | Served, the way a `providers:` instance serves any model a request names |

With `serves_unlisted_models` on, every model the provider offers is switched
on whether or not something prices it, and a model the roster switches off is
still refused.

Adding a provider offers every model its key lists:

- A model the deployment price list already prices is switched on at that rate.
- Otherwise, where `default_pricing` is on, the model is seeded with its
  models.dev rate and switched on.
- Where `default_pricing` is off, or models.dev does not price the model, it is
  offered switched off until the operator prices it.

Refreshing a provider's models moves a seeded rate nobody has changed to the
current default, and leaves a rate the operator set alone.

## Pricing

Which price list meters a request depends on whose credential served it:

| Credential | Rate used |
| --- | --- |
| The organization's own key | The organization's own rate, then the deployment price list, then the models.dev defaults (the first that answers) |
| A `providers:` instance | The deployment price list (`/api/v1/pricing`) |
| A hosted provider | The deployment price list, or a rate the operator set for this organization |

An organization may set its own rate only for a model it supplies the provider
key for. On a deployment-owned credential the deployment settles the upstream
bill, so a request served there is never priced from a rate the organization
wrote itself, even one written while its own key still covered the model. A
rate the operator set for an organization is listed beside the organization's
own on the Providers page, and only the operator may change or remove it. The
deployment price list has no dashboard page of its own; the models.dev defaults
are kept current through `/api/v1/pricing/refresh` (see
[Keeping the defaults current](configuration.md#keeping-the-defaults-current)).

On the Providers page a model nothing prices is offered and left unserved, a
key's row counts how many of its offered models are serving, and a key serving
none of them is marked, because every request through it is refused until a
switch is on.

## Catalog and discovery

What a hosted provider contributes to the catalog (`GET /api/v1/models`) is its
roster: the models switched on under it, answered for the caller's
organization, rather than everything its upstream lists.

It also takes part in the operator's views of the deployment's credentials,
beside the configured instances:

| View | Shows |
| --- | --- |
| `GET /api/v1/models/discoverable` | What each hosted provider's stored key can reach |
| `GET /api/v1/providers/health` | Whether each hosted provider answers |

Both list each hosted provider under its name, and both are the deployment's
view: a provider restricted to one organization is included, one defined in
`config.yml` is too, and one switched off is not. A hosted provider whose stored
key cannot be decrypted with the configured `OTARI_SECRET_KEY` is reported
unhealthy with that reason rather than dialed. A configured instance named like
a hosted provider shadows it here as it does at dispatch, so only the instance
is listed.

A hosted provider's model listing is dialed upstream on the deployment's stored
key, at its `api_base` when one is set:

- The background discovery refresher dials as the gateway starts and then on
  every tick, every `model_cache_ttl_seconds`, or every
  `model_discovery_negative_ttl_seconds` after a round with a failure, for as
  long as `model_discovery` is on and `model_cache_ttl_seconds` is above 0.
- A `refresh=true` read of either view dials at once.
- Changing or deleting a hosted provider drops its cached listing, so the next
  read dials the credential as it now stands.
- A provider switched off, and one whose key cannot be decrypted, are never
  dialed.

On a deployment that hosts at least one provider, an organization API key is
shown a priced model only where something would serve it; see
[Listing available models](models.md#listing-available-models).

## The `hosted_providers:` block

A standalone or hosted deployment can declare hosted providers in
`config.yml`, keyed by name, with the fields the `/api/v1/hosted-providers`
body takes:

```yaml
hosted_providers:
  openai:
    api_key: ${OPENAI_API_KEY}
  openai-eu:
    provider: openai
    api_key: ${OPENAI_EU_API_KEY}
    api_base: https://eu.api.openai.com/v1
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    restricted_to_organization_id: 4f1c2a7e-0b9d-4c55-9a0e-2d6f8b3c1e90
  vllm:
    api_base: http://vllm.internal:8000/v1
    models: [qwen3-32b]
```

| Field | Meaning |
|---|---|
| `provider` | The any-llm implementation serving it. Defaults to the name. |
| `api_key` | The key the provider is served on. Required unless the provider needs none (listed below) or has an `api_base` of its own. |
| `api_base` | An endpoint other than the provider's default. |
| `client_args` | What the provider's SDK client needs beyond the key, such as Bedrock's region. |
| `enabled` | `false` keeps the entry without serving it. Defaults to `true`. |
| `restricted_to_organization_id` | Serve one organization only, by its id. Checked at boot rather than at load, because the organizations are in the database: an id that names none is logged, and the provider then serves no organization. |
| `models` | Model ids a refresh offers when the provider cannot list its own. |

A key may be left out for the self-hosted backends `vllm`, `ollama`,
`llamacpp`, `llamafile`, `lmstudio`, `cascadia` and `otari`; for `vertexai` and
`vertexaianthropic`, which authenticate through the Google Cloud credentials in
the gateway's own environment; and for any provider given an `api_base`, which
is called with a placeholder key, as a self-hosted OpenAI-compatible server is.
Every other provider, Bedrock included, needs `api_key`. The Hosted providers
page and API follow the same rule.

Reference the key as `${VAR}` rather than writing it into the file, so it stays
in your secret store. The whole block can instead be set as JSON through
`OTARI_HOSTED_PROVIDERS`, where `${VAR}` resolves the same way:

```bash
OTARI_HOSTED_PROVIDERS='{"openai": {"api_key": "${OPENAI_API_KEY}"}}'
```

When it is set, it replaces the `hosted_providers:` block of the file (and of
`OTARI_CONFIG_YAML`) whole: a provider only the file names is not served, and
the two are never merged per provider. An empty value leaves the file's block
in place.

The block is validated at load. An unknown implementation, a `provider_type`
key (the implementation goes in `provider`), a name a hosted provider cannot
take, an unknown field, or a missing `api_key` for a provider that needs one
fails the start. With `OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS=false`, an `api_base`
that resolves to a private address fails it too, as it is refused for a
provider stored through the page; that check runs at boot, because it resolves
the host. The block is never shown by `/api/v1/settings`.

A configured entry:

- is never written to the database or encrypted;
- is listed on the Hosted providers page with a `config.yml` badge, and marked
  `source: "config"` with no `id` by the API; changing or deleting it through the
  API answers `409`, so change it in the configuration and restart;
- has its models, their serving switches and their prices managed on the page
  as any hosted provider's are;
- starts with no models and serves only the ones switched on under it, so
  refresh its models or add them before it serves a request;
- is shadowed by a provider stored through the page or the API under the same
  name, and the start logs each entry it shadows; deleting the stored provider
  serves the entry again on the same models.

Taking an entry out of the configuration leaves its models in the database,
serving nothing and offered in no catalog; they come back if the entry does.
The next catalog sweep (`POST /api/v1/hosted-providers/refresh`, Refresh and
purge on the page) deletes them and their prices, naming the provider in
`orphaned_rosters`. Where no hosted provider offers a model at all, the sweep
deletes those models and leaves every price in place.

## Copies of `providers:` entries

The `providers:` block of `config.yml` is deprecated in favor of hosted
providers. At startup, each `providers:` entry a hosted provider can hold is
copied onto a hosted provider of the same name, once:

- Its `${VAR}` values are read at that start.
- The start asks the provider for its models and switches on every one it
  lists, priced or not. An entry whose provider lists no models is not copied,
  keeps serving from `config.yml`, and is tried again at the next start.
- From then on the hosted provider serves the name, and the entry is no longer
  read; the start logs that it can be taken out of `config.yml`.
- An entry copied by an earlier start is not copied again, so removing the
  copied provider on the Hosted providers page sticks, and the entry does not
  serve either.
- Once copied, the provider sits on rung 4 of the ladder, so an organization's
  own key for the provider is asked first.

An entry a hosted provider cannot hold keeps serving from `config.yml`, and the
start logs why. That is an entry that:

- sets `models`, `max_retries` or `messages_api`, or Vertex AI's
  `credentials`, `project` or `location`;
- is a keyless Bedrock or SageMaker entry served on the server's own cloud role;
- has a name that names another provider, or that a hosted provider already
  holds;
- carries a key with no `OTARI_SECRET_KEY` to encrypt it.

`import_configured_providers: false` (`OTARI_IMPORT_CONFIGURED_PROVIDERS`)
turns the copy off, so every entry keeps serving from `config.yml`. For the
steps of moving a deployment, see
[Move config.yml providers to hosted providers](migrate-providers.md).

## The Hosted providers page

The Hosted providers page is in the deployment view, opened from the account
menu, and is listed only for a deployment operator.

- **Adding a provider** takes an optional name, which defaults to its
  implementation. The key is optional for a provider that needs none and for
  one pointed at an endpoint of its own. Adding is disabled, with a banner
  saying why, while the server has no `OTARI_SECRET_KEY`. Adding a provider of
  the same name as a `config.yml` one overrides it, which the form warns about.
- **Test connection**, in the add and edit dialogs, asks the provider what it
  lists on the dialog's values without saving them, the stored key standing in
  for a blank key box. **Test** on a provider's row asks the same of what it
  stores.
- **Serve models not on its list**, in both dialogs, sets
  `serves_unlisted_models`.
- **Models** carry a serving switch and a price, set per million tokens or per
  1,000 requests or images; a model nothing prices arrives switched off.
- **Refresh models** asks one provider again. **Refresh and purge** asks every
  provider, then removes priced models nothing on the deployment serves, along
  with the model lists of providers nothing configures any more, listing them
  before anything is removed.
- A provider kept to one organization shows which. That restriction is set
  through `/api/v1/hosted-providers`.
- A provider's name opens Models filtered to it. Above the table, a summary
  counts the providers reachable, with **Re-check all** to ask them again.
- The table also lists the `providers:` instances of `config.yml`, marked
  config.yml and read-only, with their reachability. A `hosted_providers:` entry
  carries a `config.yml` badge and cannot be edited, switched or removed here,
  while its models can.
- With no provider yet, the page walks through adding the first one, and offers
  to enable default pricing when pricing is required and nothing sets it.
- Providers that could not be moved from the retired Deployment providers page
  are listed in a banner with the reason; see
  [Move config.yml providers to hosted providers](migrate-providers.md#providers-from-the-deployment-providers-page).

## Deployment-paid credentials in standalone

Once an organization's own key and `config.yml` have both missed, the core
serves a model on one of the deployment's hosted providers, and a build may
rebind `ModelProviderPort` through `OTARI_BOOTSTRAP` to decide differently. The
deployment pays that upstream bill, so the request also goes through
`BillingPort`:

- Before the provider call, the organization is granted any credit it is owed,
  and refused with `402` (`insufficient_funds`) when it has no funds paid in.
- When the request settles, its cost, including the cost an abandoned or failed
  request still owes, is charged to the organization.
- No funds are held while the request runs, so concurrent requests can take an
  organization below zero.

A request on any other credential never reaches either port. The core's billing
adapter does nothing, so with no overlay a hosted-provider request is served and
recorded as deployment-paid on its usage row, and no organization is charged.

This covers every path that calls a provider: Chat Completions, Messages,
Responses and the Playground, each candidate of a routing policy with
fallbacks, the pass-through endpoints (embeddings, images, audio, rerank,
moderations) and batches. A routing policy asks for each candidate as it
reaches it, so the one that serves is the one that settles, and one the
organization cannot fund is skipped. A batch is admitted when it is created and
charged when its results are accounted.

## In hybrid mode

A hybrid gateway holds no hosted providers or organization keys of its own: the
control plane resolves each attempt's credential. Which kinds of credential it
can receive, and who is billed for each, is described in
[Hosted providers and BYO credentials](modes.md#hosted-providers-and-byo-credentials)
on the Runtime modes page.
