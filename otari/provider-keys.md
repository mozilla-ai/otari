---
icon: key
---

# Provider keys

Provider keys are the upstream credentials Otari uses to call LLM providers
such as OpenAI, Anthropic or Mistral. They belong to the organization, and
workspaces inherit them unless a workspace pins or turns off a key per provider.

A provider key is not an [API key](https://docs.otari.ai/guides/organizations-and-access/api-keys): your application sends an API
key to Otari, and Otari calls the provider with a provider key, so the
application never sees the provider key. Where a request's credential comes
from when the organization has no key for a model (a deployment's own providers,
or its [hosted providers](https://docs.otari.ai/self-hosting/providers/hosted-providers)) is in
[The credential ladder](https://docs.otari.ai/self-hosting/providers/hosted-providers#the-credential-ladder).

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
[Organization provider keys](https://docs.otari.ai/self-hosting/providers/hosted-providers#organization-provider-keys).

## Control workspace access

Open **Organization**, then **Workspaces**, select the workspace, and open its
**Providers** tab. Every provider the workspace could call is listed there:
the organization's own keys first, then the
deployment's [hosted providers](https://docs.otari.ai/self-hosting/providers/hosted-providers#workspace-access). A
provider with several keys, or a key and a hosted provider, is one row that
opens to show each of them. Past eight providers, a search and an **All**,
**Customized** or **Off** filter narrow the list. Each row has a switch, a
**Models** control and a menu (**⋯**):

- Turn the switch off and the workspace cannot use that key or hosted provider.
- **Models** opens a dialog. Choose **All models, including ones added later**,
  or **Only these models** and select the models the workspace may call. The
  models are listed under their vendor, since one provider can serve several
  (Together serves DeepSeek, Meta and Qwen models): a vendor's box selects all
  of its models, and opening the vendor picks some of them. A list with no
  model in it cannot be saved: to block a provider, turn its switch off.
- Where the organization holds several keys for one provider, **Always use this
  key** in a key's menu pins it for the workspace, whatever the organization's
  default.
- A row you changed is marked **Customized**. **Reset**, in its menu, returns
  it to what the organization decides.

**Require zero data retention**, at the top of the tab, holds the workspace to
providers declared to keep no prompts or completions
([Zero data retention](https://docs.otari.ai/self-hosting/providers/hosted-providers#zero-data-retention)). With it on,
a request is refused with `403` on an organization key or a hosted provider
that is not declared, and on any provider configured in `config.yml` or
endpoint the workspace or its users registered, neither of which can be
declared. The rows it blocks say so, and **Models** lists only
declared offerings for the workspace.

A new workspace starts with every row as the organization decides, so a
workspace only differs from its organization where an admin changed it.

**Models** lists what the workspace selected in the switcher may call, with
these settings applied, whatever your role in the organization.

## Rotate a provider key

Add the replacement before removing the old key. Keys are resolved at request
time, so your applications' API keys need no change.

1. Create a new key in the provider's dashboard. Do not revoke the old one yet.
2. In **Organization**, then **Providers**, click **Add provider key**, select
   the same provider, give it a distinct name, and paste the new key.
3. Make the new key the provider's default with its **star icon**.
4. For each workspace that pins the old key, open its **Providers** tab and select
   **Always use this key** in the new key's menu (**⋯**). A workspace keeps
   using a pinned key after the organization default changes.
5. Send a test request from each affected workspace and confirm it in **Usage**.
6. **Archive** the old key in Otari (an archived key can then be deleted), then
   revoke it with the provider.

If you suspect the old key was exposed, assign the replacement immediately and
revoke the exposed key with the provider.

Removing a provider key from Otari does not revoke it with the provider. Revoke
it in the provider's dashboard as well.
