# Hosted providers

A hosted provider is a credential the deployment itself pays for, offered to
the organizations it serves. This page covers where it sits among the other
credentials a request can be served on, and how a request on it is billed.

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
be served side by side; `otari` and `hosted` are reserved, so an `otari`
provider always takes a name of its own), with a roster of the models offered on it and the rate
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
served as if it did not exist. An operator configures them on the Hosted
providers page, through `/api/v1/hosted-providers`, or in the
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

## Deployment-paid credentials

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
request on any other credential never reaches either port, unless it may run a
gateway tool on the deployment's account: that tool's calls are gated and
charged the same way, and the model call is not
([Pricing a gateway-run tool](tools.md#pricing-a-gateway-run-tool)). The core's billing
adapter does nothing, so with no overlay a hosted-provider request is served
and recorded as deployment-paid on its usage row, and no organization is
charged.

This covers every path that calls a provider: Chat Completions, Messages,
Responses and the Playground, each candidate of a routing policy or a catalog id
with fallbacks, the pass-through endpoints (embeddings, images, audio, rerank,
moderations) and batches. A routing policy or a catalog id asks for each
candidate as it reaches it, so the one that serves is the one that settles, and
one the organization cannot fund is skipped. A batch is admitted when it is
created and charged when its results are accounted.

