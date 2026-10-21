# Provider keys

Provider keys are the upstream credentials Otari uses to call LLM providers
such as OpenAI, Anthropic or Mistral. They belong to the organization, and
workspaces inherit them unless a workspace pins or turns off a key per provider.

A provider key is not an [API key](api-keys.md): your application sends an API
key to Otari, and Otari calls the provider with a provider key, so the
application never sees the provider key. Where a request's credential comes
from when the organization has no key for a model (a deployment's own providers,
or its [hosted providers](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/self-hosting/providers/hosted-providers)) is in
[The credential ladder](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/self-hosting/providers/hosted-providers#the-credential-ladder).

Only organization owners and admins can add or remove provider keys, or change
which keys a workspace can use.

## Add a provider key

1. Open **Organization** and select **Providers**.
2. Click **Add provider key**.
3. Select the **Provider**.
4. Enter a **Name** that tells several keys for the same provider apart, such
   as `production`, `staging` or `backup`.
5. Paste the **API key** and click **Add provider key**.

## Set a default key per provider

If the organization has several keys for one provider, mark one as the default
so workspaces inherit it: on **Providers**, click the **star icon** next to the
key. How a workspace picks among several keys is in
[Organization provider keys](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/self-hosting/providers/hosted-providers#organization-provider-keys).

## Control workspace access

Open **Organization**, then **Workspaces**, and click **Edit** (the pencil icon)
on the workspace. For each provider key you can choose:

- **Follow the organization**: use the key as the organization's defaults
  decide.
- **Never use this key**: the workspace cannot use that key.
- **Always use this key**: the workspace always uses that key. You can restrict
  it further to specific models of that provider: pick each one under
  **Allow a model** and click **Allow**.

## Rotate a provider key

Add the replacement before removing the old key. Keys are resolved at request
time, so your applications' API keys need no change.

1. Create a new key in the provider's dashboard. Do not revoke the old one yet.
2. In **Organization**, then **Providers**, click **Add provider key**, select
   the same provider, give it a distinct name, and paste the new key.
3. Make the new key the provider's default with its **star icon**.
4. For each workspace that pins the old key, click **Edit** and select **Always
   use this key** for the new one. A workspace keeps using a pinned key after
   the organization default changes.
5. Send a test request from each affected workspace and confirm it in **Usage**.
6. **Archive** the old key in Otari (an archived key can then be deleted), then
   revoke it with the provider.

If you suspect the old key was exposed, assign the replacement immediately and
revoke the exposed key with the provider.

Removing a provider key from Otari does not revoke it with the provider. Revoke
it in the provider's dashboard as well.
