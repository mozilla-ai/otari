# Runtime modes

Otari has three deployment modes. Standalone serves both the control plane and
data plane. Hosted serves only a multi-tenant control plane. Hybrid serves a data
plane connected to an external control plane such as otari.ai.

## Comparison

| | Standalone | Hosted | Hybrid |
| --- | --- | --- | --- |
| Local database | Yes | Yes | No |
| Local management API | Yes | Yes | No |
| Local inference | Yes | No | Yes |
| Provider credentials | Local config or database | Organization-scoped for tenants; deployment config is operator-only | Resolved per request from the platform |
| Usage | Stored locally | Received and stored by the control plane | Reported to the platform |
| Dashboard | Full local dashboard | Organization-scoped control plane | Health and control-plane link |

`GET /api/v1/bootstrap` publishes the effective mode, sign-in methods, available
management surfaces, and the management or data-plane URL the dashboard needs.

## Mode and who runs it

The mode says what a process serves. It does not say who runs the process.

- A **self-hosted** gateway is one you run on your own infrastructure. It runs in
  standalone mode, or in hybrid mode when it connects to otari.ai.
- **otari.ai's own gateway** is the one mozilla.ai runs to serve otari.ai's
  inference. It runs in hybrid mode.

"Hosted" names the control-plane mode. It is not the opposite of
"self-hosted": a hosted deployment serves no inference, and anyone can run one.

## Standalone

Standalone is the default when neither `OTARI_MODE` nor `OTARI_AI_TOKEN` is
set. It serves the full API, stores keys, budgets, and usage locally, and resolves
providers from configuration or stored credentials.

SQLite is useful for evaluation. Use PostgreSQL for a durable deployment.

Standalone supports local aliases and routing policies, including failover,
weighted and priority routing, conditional selection, and learned routing. Multi-provider
fallback is therefore available without otari.ai when you configure a policy.

## Hosted

Set `OTARI_MODE=hosted` when one process is the control plane for multiple
organizations.

Hosted mode serves the management API and `GET /api/v1/models`, but does not serve
inference, files, batches, or other data-plane operations. Those paths return a
descriptive `404`. Set `data_plane_url` so the error and dashboard snippets
point clients to the correct gateway.

The dashboard keeps the organization-scoped provider keys that a standalone
deployment also has, and the Hosted providers page, where an
operator configures the credentials the deployment pays for (see
[Hosted providers and BYO credentials](#hosted-providers-and-byo-credentials)).
It also hides the Playground, whose whole purpose is to dispatch a completion.
Deployment-wide APIs still require operator authority; organization-scoped APIs
apply membership and role checks.

A hosted control plane and its hybrid gateways form one system: the gateway
resolves credentials from the control plane and reports usage back so the
control plane can debit the correct tenant.

## Hybrid, connected to otari.ai

Hybrid mode is selected when `OTARI_AI_TOKEN` is present. Set
`OTARI_MODE=hybrid` as well if you want startup to require that mode. Conflicting
configurations fail at startup:

- `hybrid` without a platform token
- `standalone` or `hosted` with a platform token

The token is the gateway credential created in otari.ai, commonly prefixed
`gw_`. It is not the user token sent by clients.

```bash
export OTARI_AI_TOKEN=gw_your_gateway_token
otari serve
```

Hybrid serves health, bootstrap, Chat Completions, Messages, Responses,
embeddings, rerank, image generation, audio transcription and speech,
moderations, direct MCP tool execution (`POST /api/v1/mcp/execute`), and the
guardrail hook check (`POST /api/v1/hooks/check`). Files, batches, direct search
and decision calls (`/decisions`, `/systemone`, `/routing/recommend`) are
standalone only. It
does not initialize the local management database or use local provider
configuration. Clients authenticate with an otari.ai user
token, accepted in the same header forms as standalone mode
(`Authorization: Bearer`, `Otari-Key`, or `x-api-key`).

The gateway asks the platform to resolve an ordered set of provider attempts,
tries retryable fallbacks before a response begins, and reports each outcome.
Workspace MCP and web-search configuration are resolved through the same control
plane.

## Hosted providers and BYO credentials

A request is served on the first credential the ladder finds. A provider
instance configured on the deployment, in `config.yml` or through the Providers
page, serves every request that names it, including a bare `provider:model`
request whose provider spells that instance's name. Where no instance answers,
the organization's own provider key serves the request. Where that is missing
too, one of the deployment's **hosted providers** does.

A hosted provider is a credential the deployment's operator configures for the
deployment to serve requests on, under a name that defaults to its provider
(a name of its own, such as `openai-eu`, lets two endpoints of one provider
be served side by side), with a roster of the models offered on it and the rate
each is charged at on the deployment price list. A hosted provider named after its
provider is reached by a bare `provider:model` request once neither a
`providers:` instance nor the organization's own key answers. One with a name
of its own is reached only by that name (`openai-eu:gpt-4o`): an organization's
key for the provider never serves it, because that key belongs to another
endpoint. A request whose workspace belongs to no organization is served only
on a hosted provider restricted to no organization.
Only a model switched on in that roster is served on the hosted credential: a
model switched off, or one the roster does not list, is refused with `424`. The
deployment settles the upstream bill for a hosted model, which is why its rate
belongs to the deployment and not to any one organization. A hosted provider
can be restricted to one organization, and every other organization is then
served as if it did not exist. Any standalone or hosted deployment configures
them on the Hosted providers page, through `/api/v1/hosted-providers`, or in the
`hosted_providers:` block of `config.yml` (see
[Configuration](configuration.md#hosted-providers-in-configyml)), where a stored
provider of the same name wins over the configured one.

What a hosted provider contributes to the catalog is its roster: the models
switched on under it, answered for the caller's organization. It also takes part
in the operator's views of the deployment's credentials, beside the configured
instances. `GET /api/v1/models/discoverable` lists what each hosted provider's
stored key can reach, and `GET /api/v1/providers/health` reports whether it
answers, each under the provider's name. Both are the deployment's view, so a
provider restricted to one organization is included, a provider defined in
`config.yml` is too, and a provider switched off is not. A hosted provider whose
stored key cannot be decrypted with the configured `OTARI_SECRET_KEY` is
reported unhealthy with that reason rather than dialed. A configured instance
named like a hosted provider shadows it here as it does at dispatch, so only the
instance is listed.

Being in those views means a hosted provider's model listing is dialed upstream
on the deployment's stored key, at its `api_base` when one is set. The
background discovery refresher does so as the gateway starts and then on every
tick, every `model_cache_ttl_seconds` (sooner, every
`model_discovery_negative_ttl_seconds`, after a round with a failure), for as
long as `model_discovery` is on and that TTL is above 0. A `refresh=true` read
of either view dials at once, and changing or deleting a hosted provider drops
its cached listing so the next read dials the credential as it now stands. A
provider switched off and one whose key cannot be decrypted are never dialed.

otari.ai is one deployment of this feature: its hosted control plane holds the
hosted providers that serve its managed models. Hybrid mode can therefore
receive two kinds of provider credential:

- A workspace's own provider key. The upstream provider bills that workspace.
  Any hybrid gateway can use it, including a self-hosted one.
- A control plane's hosted-provider credential. Usage is billed by the control
  plane. On otari.ai, only otari.ai's own gateway receives it.

Managed model identifiers use the catalog values the control plane publishes.
otari.ai refuses a hosted credential to a self-hosted gateway, so platform-owned
secrets never leave mozilla.ai's infrastructure.

## Deployment-paid credentials in standalone

A deployment can serve a model on a credential it owns, once an organization's
own key and `config.yml` have both missed: the core serves it
on one of the deployment's hosted providers, and a build may rebind
`ModelProviderPort` through `OTARI_BOOTSTRAP` to decide differently. The deployment pays that
upstream bill, so the request also goes through `BillingPort`. Before the
provider call, the organization is granted any credit it is owed and refused
with `402` (`insufficient_funds`) when it has no funds paid in. When the
request settles, its cost, including the cost an abandoned or failed request
still owes, is charged to the organization. No funds are held while the
request runs, so concurrent requests can take an organization below zero. A
request on any other credential never reaches either port. The core's billing
adapter does nothing, so with no overlay a hosted-provider request is served
and recorded as deployment-paid on its usage row, and no organization is
charged.

This covers every path that calls a provider: Chat Completions, Messages,
Responses and the Playground, each candidate of a routing policy with fallbacks,
the pass-through endpoints (embeddings, images, audio, rerank, moderations) and
batches. A routing policy asks for each candidate as it reaches it, so the one
that serves is the one that settles, and one the organization cannot fund is
skipped. A batch is admitted when it is created and charged when its results
are accounted.

## Internal protocol

The gateway and control plane exchange provider resolution, MCP configuration,
web-search policy, and usage reports. Integrators building a compatible control
plane should use the normative [hybrid-mode protocol](hybrid-mode-protocol.md)
rather than this overview.
