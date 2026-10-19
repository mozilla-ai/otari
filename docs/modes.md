# Runtime modes

Otari has two deployment modes, selected by `OTARI_MODE`. Standalone serves both
the control plane and data plane. Hosted serves only a multi-tenant control plane.

## Comparison

| | Standalone | Hosted |
| --- | --- | --- |
| Local database | Yes | Yes |
| Local management API | Yes | Yes |
| Local inference | Yes | No |
| Provider credentials | Local config or database | Organization-scoped for tenants; deployment config is operator-only |
| Dashboard | Full local dashboard | Organization-scoped control plane |

`GET /api/v1/bootstrap` publishes the effective mode, sign-in methods, available
management surfaces, and the data-plane URL the dashboard needs.

Hybrid mode, a gateway attached to otari.ai, has been removed. Otari refuses to
start while `OTARI_AI_TOKEN` is set or `OTARI_MODE` is `hybrid`, and ignores a
leftover `platform:` block in `config.yml` with a warning. The `PLATFORM_*` and
`STREAMING_FALLBACK_*` variables are no longer read. To keep serving, unset the
token and run the gateway in standalone mode with providers of its own.

## Mode and who runs it

The mode says what a process serves. It does not say who runs the process.

A **self-hosted** gateway is one you run on your own infrastructure, in
standalone mode. "Hosted" names the control-plane mode. It is not the opposite of
"self-hosted": a hosted deployment serves no inference, and anyone can run one.

## Standalone

Standalone is the default when `OTARI_MODE` is unset. It serves the full API, stores keys, budgets, and usage locally, and resolves
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

The dashboard hides the Providers page, whose stored instances are deployment-wide
rather than per tenant, leaving the organization-scoped provider keys that a
standalone deployment also has. It keeps the Hosted providers page, where an
operator configures the credentials the deployment pays for (see
[Hosted providers and BYO credentials](#hosted-providers-and-byo-credentials)).
It also hides the Playground, whose whole purpose is to dispatch a completion.
Deployment-wide APIs still require operator authority; organization-scoped APIs
apply membership and role checks.

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

## Deployment-paid credentials in standalone

A deployment can serve a model on a credential it owns, once an organization's
own key, a stored provider and `config.yml` have all missed: the core serves it
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

Today this covers Chat Completions, Messages, Responses and the Playground for a
plain model or a single-target alias or policy. A routing policy with fallback
candidates, the pass-through endpoints (embeddings, images, audio, rerank,
moderations) and batches never serve on a deployment-owned credential. A model
the deployment offers the organization only that way is refused there with `424`
(`managed_credential_unsupported`) rather than `provider_not_configured`, before
anything is sent upstream. A routing policy skips such a candidate as it skips
one with no credential, and answers with that code only when no candidate served.
