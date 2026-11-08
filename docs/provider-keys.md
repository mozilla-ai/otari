---
icon: key
---

# Provider keys

Provider keys are the upstream credentials Otari uses to call LLM providers
such as OpenAI, Anthropic or Mistral. They belong to the organization, and
workspaces inherit them unless a workspace pins or turns off a key per provider.

A provider key is not an [API key](api-keys.md): your application sends an API
key to Otari, and Otari calls the provider with a provider key, so the
application never sees the provider key. Where a request's credential comes
from when the organization has no key for a model (a deployment's own providers,
or its [hosted providers](hosted-providers.md)) is in
[The credential ladder](hosted-providers.md#the-credential-ladder).

Only organization owners and admins can add or remove provider keys, or change
which keys a workspace can use.

## Add a provider key

1. Open **Organization** and select **Providers**.
2. Click **Add provider key**.
3. Select the **Provider**.
4. Enter a **Name** that tells several keys for the same provider apart, such
   as `production`, `staging` or `backup`.
5. Paste the **API key** and click **Add provider key**.

## Add a key for your own endpoint

A key can point at a compatible endpoint other than the provider's own, such
as an OpenAI-compatible host like Synthetic, or a vLLM or LM Studio server:
enter its **API base URL** when you add the key. Such a key is a different
service from the provider it speaks the protocol of, so requests name it by a
**Request prefix** of its own rather than by the provider:

```text
synthetic:hf:Qwen/Qwen3.8-27B
```

The prefix is made from the key's name unless you enter one. It uses lowercase
letters, digits, `.`, `_` and `-`, cannot be a provider's name (`openai`,
`vllm`) or a provider instance the deployment configures, and is unique in the
organization. It never changes once set, because your applications send it;
renaming the key leaves it as it was.

`openai:<model>` therefore never reaches a key with its own endpoint, even one
of provider OpenAI: it reaches the organization's OpenAI keys without one, or
the deployment's OpenAI. The key's models, its rates (**Set your rate**) and its
usage are all listed under its prefix, and the key is never a provider's
default (it is the only key its prefix names). A spending ceiling narrowed to the
provider does not cover it either: narrow one to the prefix.

The base URL is set when the key is added. You can change it later, but not
remove it, and a key added without one cannot gain one: add a new key for the
endpoint instead.

## Set a default key per provider

If the organization has several keys for one provider, mark one as the default
so workspaces inherit it: on **Providers**, click the **star icon** next to the
key. How a workspace picks among several keys is in
[Organization provider keys](hosted-providers.md#organization-provider-keys).

## Control workspace access

Open **Organization**, then **Workspaces**, select the workspace, and open its
**Providers** tab. Every provider the workspace could call is listed there:
the organization's own keys first, then the
deployment's [hosted providers](hosted-providers.md#workspace-access). A
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
([Zero data retention](hosted-providers.md#zero-data-retention)). With it on,
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

A key with its own [request prefix](#add-a-key-for-your-own-endpoint) is
rotated in place instead, since no other key can take its prefix: click **Edit**
on the key, paste the new API key, and save.

Removing a provider key from Otari does not revoke it with the provider. Revoke
it in the provider's dashboard as well.
