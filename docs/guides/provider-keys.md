# Provider keys

Provider keys are the upstream credentials Otari uses to call LLM providers
such as OpenAI, Anthropic or Mistral. They belong to the organization.
Workspaces inherit them by default and can pin or turn off a key per provider.

## Provider keys and API keys

The two names are close, and they are different things:

- A **provider key** is your upstream credential, such as your OpenAI key. Otari
  uses it to call the provider. It is managed at the organization level.
- An **API key** is the workspace-scoped credential your application sends to
  Otari ([API keys](api-keys.md)).

Your application sends an API key to Otari, and Otari calls the provider with a
provider key. The application never sees the provider key.

## Where a request's credential comes from

A request can be served on one of three kinds of credential, tried in this
order:

1. **A provider the deployment configures itself.** On a self-hosted gateway,
   that is a provider in `config.yml` ([Configure providers](../self-hosting/configure-providers.md)).
2. **Your organization's provider keys**, which owners and admins add in the
   dashboard. The provider bills you directly.
3. **Models the deployment provides** (hosted providers), on keys the deployment
   owns and charges for. On otari.ai these are the Otari-managed models, paid
   from your organization's [balance](../otari-ai/billing.md). On a self-hosted
   gateway, its operator manages them; see [Hosted providers](#hosted-providers)
   below.

Within your organization's keys, a workspace resolves a provider's key this way:

1. If the workspace pins a key for the provider, it uses that key.
2. Otherwise it uses the organization's default key for that provider.
3. If the workspace has turned the organization default off, the request fails
   for that provider.

## Add a provider key

Only owners and admins can add or remove provider keys.

1. Open **Organization** and select **Providers**.
2. Click **Add provider key**.
3. Select the **Provider**.
4. Enter a **Key name** that tells several keys for the same provider apart, such
   as `production`, `staging` or `backup`.
5. Paste the **API key** and click **Add provider key**.

## Set a default key per provider

If the organization has several keys for one provider, mark one as the default
so workspaces inherit it: on the **Providers** tab, click the **star icon** next
to the key.

## Control workspace access

Open the workspace and click **Edit** (the pencil icon). For each provider you
can choose:

- **Following the organization**: use the organization's default key.
- **Never use this key**: the workspace cannot use that key.
- **Always use this key**: the workspace always uses that key. You can restrict
  it further to specific models of that provider by selecting each one and
  clicking **Allow**.

## Rotate a provider key

Add the replacement before removing the old key. Keys are resolved at request
time, so your applications' API keys need no change.

1. Create a new key in the provider's dashboard. Do not revoke the old one yet.
2. In **Organization** → **Providers**, click **Add provider key**, select the
   same provider, give it a distinct name, and paste the new key.
3. Make the new key the provider's default with its **star icon**.
4. For each workspace that pins the old key, click **Edit** and select **Always
   use this key** for the new one. A workspace keeps using a pinned key after
   the organization default changes.
5. Send a test request from each affected workspace and confirm it in **Usage**.
6. Remove the old key in Otari, then revoke it with the provider.

If you suspect the old key was exposed, assign the replacement immediately and
revoke the exposed key with the provider.

## How provider keys are protected

- Provider keys are owned by the organization, not copied into each workspace.
- Each workspace decides which organization keys apply to its workloads.
- Only owners and admins can add or remove key material, or change which keys a
  workspace can use.
- Otari resolves the key at request time, so an API key never carries the
  upstream secret.

Removing a provider key from Otari does not revoke it with the provider. Revoke
it in the provider's dashboard as well.

## Hosted providers

A hosted provider is a provider key the deployment itself owns and pays for,
with the roster of models it offers on that key and the rate it charges for
each. Only a deployment operator manages them: the master key, the bootstrap
operator's session, or the session of an account granted operator access. That
covers every route under `/api/v1/hosted-providers`, reads included. Any other
signed-in account, whatever its role in whichever organization, gets the same
`404` that `/api/v1/admin` gives, and an API key gets `401`. Organization owners
and admins manage their own organization's provider keys, never these.

A hosted provider is the last rung of the credential ladder: a request is
served on it only when no provider instance the deployment configures and no
provider key of the caller's organization covers the model.

Adding a provider offers every model its key lists. A model the deployment
price list already prices is switched on at that rate. Otherwise, where
`default_pricing` is on, the model is seeded with its models.dev rate and
switched on; where it is off, or models.dev does not price the model, the model
is offered switched off until the operator prices it. Refreshing a provider's
models moves a seeded rate nobody has changed to the current default, and
leaves a rate the operator set alone. The API never returns a hosted provider's
key, only its last four characters, and it masks credentials in `client_args`
and in the base URL.
