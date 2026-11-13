# Hosted providers and BYO credentials

Every provider call is made on one credential: from the deployment's
configuration, an organization's own provider key (bring your own key, BYO), or
one of the deployment's **hosted providers**, which the deployment owns and
pays for. The upstream bill follows the credential, so a hosted model's rate
belongs to the deployment price list.

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

Where an organization holds several keys for one provider, a workspace uses,
in order, skipping keys it turned off:

1. the key it set to "Always use this key"
2. the organization's default for the provider (the star on the Providers
   page; click a filled star to remove it)
3. the organization's oldest key

A key with its own API base URL is not in this list: requests name it by its
own prefix (see
[Add a key for your own endpoint](model-providers.md#add-a-key-for-your-own-endpoint)),
and it serves every workspace that has not turned it off.

The models a key serves and its serving switches are in
[Who is shown which models](models.md#who-is-shown-which-models).

### Zero data retention

A hosted provider or an organization's provider key can be declared zero data
retention (`zero_data_retention`): its account keeps no prompts or completions.
Otari cannot verify this; it records the declaration. The Models page shows it
beside each offering and filters on it (see
[Browsing the grouped catalog](models.md#browsing-the-grouped-catalog)). A
workspace can require it, refusing every undeclared source; see
[Control workspace access](model-providers.md#control-workspace-access).

Whose declaration counts depends on whose key serves the offering:

- A `hosted` offering takes the hosted provider's declaration.
- An offering on the organization's own key counts only while every usable
  key the organization holds for that provider is declared, since any of them
  may serve a request.
- A `providers:` instance from `config.yml` is never declared.

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
implementation's name. An `otari` provider therefore always takes a name of its
own.

## Who manages hosted providers

Only a deployment operator manages hosted providers: the master key, the
bootstrap operator's session, or an account granted operator access. That
covers every route under `/api/v1/hosted-providers`, reads included. Any other
signed-in account gets `404` (as on `/api/v1/admin`), and an API key gets
`401`.

A deployment configures hosted providers in three places:

- the Hosted providers page (see [The Hosted providers page](#the-hosted-providers-page)),
- `/api/v1/hosted-providers`,
- the `hosted_providers:` block of `config.yml` (see
  [The `hosted_providers:` block](#the-hosted_providers-block)).

A provider stored through the page or the API wins over a `config.yml` entry of
the same name.

Storing a hosted provider's key requires an encryption key: `OTARI_SECRET_KEY`,
or on SQLite the key Otari generates beside the database (see
[Runtime provider management](configuration.md#runtime-provider-management)).
The API returns only the key's last four characters, and masks credentials in
`client_args` and the base URL.

## Connecting otari.ai

[otari.ai](https://otari.ai) is a hosted Otari: one key reaches models from many
providers. It is a paid service, and a request served on it leaves this gateway
for otari.ai. A gateway connects to it as a hosted provider named `otari-ai`, of
the `otari` implementation.

The first time a deployment operator opens the dashboard in a browser, on a
gateway not yet connected, a **Welcome to Otari** dialog offers otari.ai, with a
link to its plans. It is remembered per browser, not per account: skip it or
close it and that browser does not show it again, while a new browser, or an
existing operator after upgrading, sees it once. It is not shown where
`platform_url` is empty or the gateway cannot encrypt a provider key.

On the Hosted providers page, **Connect otari.ai** is offered beside adding a
provider of your own:

1. The gateway asks otari.ai for a code and opens its approval page in a new
   tab. The page here shows the code and the link while it waits.
2. Sign in to otari.ai (or create an account) and approve the code.
3. The gateway stores the key otari.ai issues, lists its models and prices
   them. With no other provider yet, the dashboard moves on to the setup guide.

What a connection stores:

- A hosted provider `otari-ai`, restricted to the organization of the operator
  who connected. Lift or change the restriction through
  `/api/v1/hosted-providers`.
- Its models, under the names otari.ai gives them: `<vendor>/<model>`, such as
  `openai/gpt-4o-mini`. A request names one as
  `otari-ai:<vendor>/<model>` (`otari-ai:openai/gpt-4o-mini`), which always
  reaches otari.ai. The catalog (`GET /api/v1/models`) and the setup guide list
  the same model as `openai/gpt-4o-mini`, which a request may send too.
- A rate for each model, so it arrives switched on and a request can be sent
  at once:
  - the price otari.ai lists for the model, which is what it charges. The page
    marks it **Listed**, and a refresh moves it when otari.ai changes it;
  - where otari.ai lists none, the rate of the vendor's models.dev entry, an
    estimate of what otari.ai charges, marked **Estimate**.

  Both are stored whether `default_pricing` is on or off, for a provider of the
  `otari` implementation only. Set your own rate to replace either. A model
  neither prices is offered switched off, as on any hosted provider.

Connecting again replaces the key of the existing `otari-ai` provider. The
earlier key stays valid on otari.ai until you revoke it on its
[API keys page](https://otari.ai/dashboard/#/keys). A code
that expires or is declined leaves nothing behind; choose **Try again**.

To disconnect, remove the `otari-ai` provider. That deletes this gateway's copy
of the key only, so also revoke the key on otari.ai's
[API keys page](https://otari.ai/dashboard/#/keys).

**Without the button.** Create a key on otari.ai's API keys page and add it as
any hosted provider: choose the `otari` implementation, name it `otari-ai`,
paste the key, and set the base URL to `https://api.otari.ai` (the origin,
without `/api/v1`). The result is the same provider.

otari.ai shows the gateway, on its approval page and on the key, under the host
of `public_base_url`, or under the machine's hostname where that is not set.

When trying this against an otari.ai running on the same machine, open one of
the two dashboards on `127.0.0.1` and the other on `localhost`. Browsers keep
one cookie jar per host whatever the port, so two dashboards on `localhost`
share a session cookie and sign each other out.

Nothing is sent to otari.ai until an operator presses **Connect otari.ai**. The
address it asks is the `platform_url` setting (`OTARI_PLATFORM_URL`,
`https://api.otari.ai` by default); set it to an empty string and the option,
every mention of it and its routes (`/api/v1/platform/connect`) are gone. The
exchange itself is specified in the
[Platform connect protocol](platform-connect-protocol.md).

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
on, priced or not; a model the roster switches off is still refused.

A provider can stop serving a model it still lists (Together does this when it
moves a model to dedicated endpoints). The first request or
[health check](health-checks.md) the provider refuses that way on the hosted
credential switches the model off with `disabled_reason: unavailable` and
emails the [alert list](health-checks.md#email-alerts). Switch it back on if
the refusal was wrong. Only a refusal about the model itself, on the hosted
credential, does this.

Adding a provider offers every model its key lists:

- A model the deployment price list already prices is switched on at that rate.
- Otherwise, where `default_pricing` is on, the model is seeded with its
  models.dev rate and switched on. A provider of the `otari` implementation is
  priced whatever `default_pricing` says, at the price its own listing names
  where it names one; see [Connecting otari.ai](#connecting-otariai).
- Where `default_pricing` is off, or models.dev does not price the model, it is
  offered switched off until it is priced, by the operator or by a later
  refresh once models.dev prices it.

Refreshing a provider's models:

- moves an unchanged seeded rate to the current default, and leaves an
  operator-set rate alone
- never switches on a model the operator switched off
- keeps on a model switched on without a price on purpose (one a moved or
  copied provider lists, one an operator switched on unpriced, and every model
  of a provider that serves unlisted models)

### Workspace access

On a workspace's **Providers** tab (**Organization**, then **Workspaces**),
an organization or workspace owner or admin can switch a hosted provider off
for that workspace or narrow it to some of its models. Workspace members can
read the setting.

- A workspace can only take away, never add a provider or model.
- A refused request gets `403`, as for a model an API key may not use, and the
  workspace's catalog leaves the model out.
- Where the workspace uses the organization's own key for the provider, the
  hosted provider is never asked, so the setting has no effect.
- A change applies at once on the server that saved it and within 30 seconds
  on the others. The API is
  `/api/v1/workspaces/{workspace_id}/hosted-providers`.

## Pricing

Which price list meters a request depends on whose credential served it:

| Credential | Rate used |
| --- | --- |
| The organization's own key | The organization's own rate, then the deployment price list, then the models.dev defaults (the first that answers) |
| A `providers:` instance | The deployment price list (`/api/v1/pricing`) |
| A hosted provider | The deployment price list, or a rate the operator set for this organization |

- An organization may set its own rate only for a model it supplies the key
  for. A request served on a deployment credential is never priced from an
  organization's own rate.
- A rate the operator set for an organization is listed beside the
  organization's own on the Providers page; only the operator can change it.
- The deployment price list has no dashboard page. The models.dev defaults are
  kept current by **Update catalog** on the Hosted providers page, or on a
  schedule under `pricing_refresh` (see
  [Keeping the defaults current](configuration.md#keeping-the-defaults-current)).

On the Providers page, an unpriced model is offered but not served, each key's
row counts its serving models, and a key serving none is marked.

### Entering a rate

The rate form asks for the unit first: tokens, requests, images, megapixels,
characters, minutes of audio, seconds of video, or web searches. Enter each rate
the way providers publish it:

| Unit | Entered per |
| --- | --- |
| Tokens, characters | 1,000,000 |
| Requests, images, searches | 1,000 |
| Megapixels, minutes, seconds | 1 |

A minute of audio has two rates, one for audio the model hears and one for audio
it speaks. The model lists show each rate in its unit, for example "$0.002 per
request" or "$0.06 per second". How each unit is stored and billed is in
[Each endpoint in its unit](configuration.md#each-endpoint-in-its-unit).

Some limits of the form:

- An override that prices seconds of both audio and video shows its audio rates;
  change its video price through the API.
- An edit in the same unit keeps the override's context tiers and rules, which
  the form does not show.
- The hosted model form stores per-million fields only, so it does not offer
  seconds of video and cannot reprice a per-second model.

**Price this model** on a request in the Requests tab opens the same form, in the
unit the request's endpoint bills. It is offered on a served request that carried
no cost.

## Catalog and discovery

A hosted provider contributes its roster to the catalog (`GET /api/v1/models`):
the models switched on under it for the caller's organization, not everything
its upstream lists.

It also appears in the operator's credential views, beside configured
instances:

| View | Shows |
| --- | --- |
| `GET /api/v1/models/discoverable` | What each hosted provider's stored key can reach |
| `GET /api/v1/providers/health` | Whether each hosted provider answers |

Both list each hosted provider by name, including organization-restricted and
`config.yml` ones, but not switched-off ones. A provider whose key the
configured `OTARI_SECRET_KEY` cannot decrypt is reported unhealthy, not dialed.
A configured instance with a hosted provider's name shadows it, so only the
instance is listed.

The model listing is fetched on the stored key, at its `api_base` when set:

- The discovery refresher fetches at startup and then every
  `model_cache_ttl_seconds` (or `model_discovery_negative_ttl_seconds` after a
  failed round), while `model_discovery` is on and `model_cache_ttl_seconds` is
  above 0.
- A `refresh=true` read of either view fetches at once.
- Changing or deleting a hosted provider drops its cached listing.
- Switched-off providers and undecryptable keys are never fetched.

On a deployment that hosts at least one provider, an organization API key is
shown a priced model only where something would serve it; see
[Listing available models](models.md#listing-available-models).

## The `hosted_providers:` block

Declare hosted providers in `config.yml`, keyed by name, with the fields the
`/api/v1/hosted-providers` body takes:

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
| `supported_countries` | The two-letter country codes whose requests it serves. Unset serves every country. See [Supported countries](configuration.md#supported-countries). |
| `zero_data_retention` | `true` declares that the provider keeps no prompts or completions. Defaults to `false`. See [Zero data retention](#zero-data-retention). |

`api_key` may be left out for:

- the self-hosted backends `vllm`, `ollama`, `llamacpp`, `llamafile`,
  `lmstudio`, `cascadia` and `otari`
- `vertexai` and `vertexaianthropic`, which use the Google Cloud credentials in
  the gateway's environment
- any provider given an `api_base` (called with a placeholder key)

Every other provider, Bedrock included, needs it. The page and API follow the
same rule.

Reference the key as `${VAR}` so it stays in your secret store. The whole block
can instead be set as JSON in `OTARI_HOSTED_PROVIDERS`, where `${VAR}` resolves
the same way:

```bash
OTARI_HOSTED_PROVIDERS='{"openai": {"api_key": "${OPENAI_API_KEY}"}}'
```

When set, it replaces the file's (and `OTARI_CONFIG_YAML`'s) block whole; the
two are never merged per provider. An empty value leaves the file's block in
place.

The block is validated at load. These fail the start: an unknown
implementation, a `provider_type` key (use `provider`), a name a hosted
provider cannot take, an unknown field, or a missing required `api_key`. With
`OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS=false`, an `api_base` resolving to a
private address also fails it (checked at boot). `/api/v1/settings` never shows
the block.

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

Removing an entry from the configuration leaves its models in the database,
unserved; they come back if the entry does. A catalog update with cleanup
(**Update catalog** on the page, or
`POST /api/v1/hosted-providers/catalog-update` with `cleanup`) deletes them and
their prices, naming the provider in `orphaned_rosters`. Models no hosted
provider offers at all are deleted, but their prices are kept.

## Copies of `providers:` entries

The `providers:` block of `config.yml` is deprecated in favor of hosted
providers. At startup, each `providers:` entry a hosted provider can hold is
copied onto a hosted provider of the same name, once:

- Its `${VAR}` values are read at that start.
- The start switches on every model the provider lists, priced or not, and
  later refreshes keep them on. An entry whose provider lists no models is not
  copied, keeps serving from `config.yml`, and is retried at the next start.
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
steps of moving a deployment, see [Upgrading](#upgrading).

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
- **Zero data retention**, in both dialogs, sets `zero_data_retention`, and a
  provider declared so carries a badge saying it. The same box is on each
  key's dialog on the organization's Providers page. See
  [Zero data retention](#zero-data-retention).
- **Models** carry a serving switch and a price, set in any unit on
  [Entering a rate](#entering-a-rate) (a per-second video price
  goes through `POST /api/v1/pricing` instead); an unpriced model arrives
  switched off. Tabs show one kind of model at a time (the kind comes from
  models.dev; `GET /api/v1/hosted-providers/{provider}/models` takes it as
  `type`), and the search box beside them finds models by name (`search`).
  See [Who is shown which models](models.md#who-is-shown-which-models).
- **Refresh models** asks one provider again. **Update catalog** fetches the
  latest models.dev prices and every provider's listing, and shows the price
  changes and new models before applying them (see
  [Keeping the defaults current](configuration.md#keeping-the-defaults-current)).
  Its optional cleanup removes priced models nothing serves and the model
  lists of providers no longer configured, listing them first.
- A provider kept to one organization shows which. That restriction is set
  through `/api/v1/hosted-providers`.
- The **Countries** column shows the countries a provider serves: two-letter
  codes, typed in the dialogs or loaded from a JSON file (a list, or one under
  `supported_countries`); blank serves every country (see
  [Supported countries](configuration.md#supported-countries)).
- A provider's name opens Models filtered to it. Above the table, a summary
  counts the providers reachable, with **Re-check all** to ask them again.
- The table also lists the `providers:` instances of `config.yml`, marked
  config.yml and read-only, with their reachability. A `hosted_providers:` entry
  carries a `config.yml` badge and cannot be edited, switched or removed here,
  while its models can.
- With no provider yet, the page walks through adding the first one, and offers
  to enable default pricing when pricing is required and nothing sets it.
- **Connect otari.ai** is offered beside adding a provider, on the first-run
  panel and above the table afterwards; see
  [Connecting otari.ai](#connecting-otariai).
- Providers that could not be moved from the retired Deployment providers page
  are listed in a banner with the reason; see
  [From the Deployment providers page](#from-the-deployment-providers-page).

## Deployment-paid credentials

A request served on a hosted provider is paid by the deployment, so it goes
through `BillingPort` (a build may also rebind `ModelProviderPort` through
`OTARI_BOOTSTRAP` to choose the credential differently):

- Before the provider call, the organization is granted any credit it is owed,
  and refused with `402` (`insufficient_funds`) when it has no funds paid in.
- When the request settles, its cost, including the cost an abandoned or failed
  request still owes, is charged to the organization.
- No funds are held while the request runs, so concurrent requests can take an
  organization below zero.

In the open-source build the billing adapter does nothing: a hosted-provider
request is served and recorded as deployment-paid on its usage row, and no
organization is charged.

Requests on other credentials skip billing, except a gateway tool run on the
deployment's account, whose calls are gated and charged the same way (see
[Pricing a gateway-run tool](tools.md#pricing-a-gateway-run-tool)).

A deployment-paid credential's provider account is shared by every
organization the deployment serves, so a request on one cannot name state
stored on that account. A provider `container` id, `previous_response_id`,
`conversation` and an `item_reference` input item are refused with `400`, and a
routing plan passes over the candidate to the next one. Send the previous turns
in `input` instead, or use a model on the organization's own key.

This covers every provider path: Chat Completions, Messages, Responses, the
Playground, the pass-through endpoints (embeddings, images, audio, rerank,
moderations) and batches. Each routing or fallback candidate is checked as it
is reached, so one the organization cannot fund is skipped. A batch is admitted
when created and charged when its results are accounted.

## Upgrading

### From a `providers:` block in config.yml

1. On PostgreSQL, set `OTARI_SECRET_KEY` first if it is not set
   (`otari gen-secret-key`): an entry whose key cannot be encrypted is not
   copied. A SQLite deployment generates its own key, so its entries are copied. To keep serving every
   entry from `config.yml` for now, set `import_configured_providers: false`
   instead.
2. Start the upgraded gateway, and read its log: it names each entry it copied
   and, for each one it kept, why ([Copies of `providers:` entries](#copies-of-providers-entries)).
3. Remove the copied entries from `config.yml`, since they are no longer read,
   and rotate their keys on the Hosted providers page from now on.
4. For an entry the start kept, either leave it in `config.yml`, where it keeps
   serving, or remove what stops the copy (such as `models` or `max_retries`)
   and restart.

A copied provider sits on the last rung of the
[credential ladder](#the-credential-ladder): an organization that holds its own
key for the provider is served on that key, where the `config.yml` entry was
asked before it.

### From the Deployment providers page

The Deployment providers page (`/api/v1/provider-credentials`) is retired. At
startup each provider it stored becomes a hosted provider of the same name, so
selectors, routing policies and aliases keep working. Every model it lists is
switched on, priced or not, and stays on after **Update catalog**; "Serve
models not on its list" starts off.

A provider that cannot be moved is logged, listed in a banner on the Hosted
providers page (`GET /api/v1/hosted-providers/migration`), and serves nothing
until fixed. It cannot be moved when:

- it lists no models
- a hosted provider already holds the name
- no configured `OTARI_SECRET_KEY` decrypts its key
- it is named after another provider
- it is a keyless Bedrock or SageMaker provider on the server's own cloud role
  (put that under `providers:` in `config.yml`)

Once fixed, move it from the banner, or **Discard** it.

A moved provider is asked after an organization's own key, where a stored
provider was asked before it, and a `config.yml` provider of the same name now
serves ahead of it.
