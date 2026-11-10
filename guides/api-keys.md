# API keys

An API key is the credential an application sends to Otari. Each key belongs to
one workspace and one user, and that decides where its usage and spend land.

Applications send the key as a bearer token. The base URL to use and the other
accepted headers are in [Base URL](https://docs.otari.ai/reference#base-url) and
[Authentication](https://docs.otari.ai/reference#authentication).

## Create a key

1. Open the workspace's **API keys** page and click **Create key**.
2. Enter a **Name** and, optionally, an expiry under **Expires (optional)**.
3. Optionally, click **Advanced** to narrow the models and tools the key can
   use. By default a key inherits your model access, and it can never exceed
   it.
4. Click **Create key**, then copy the key. It is shown once, so store it
   before you close the dialog.

![The New key dialog, with a name and an owner filled in.](assets/screenshots/api-keys-create.png)

## Cap a key's spend

An organization owner or admin, or a deployment operator, can give a key a
budget of its own. In the key's create or edit form, either:

- pick an existing budget under **Budget (optional)**, or
- set a **Limit (USD)** and how often it **Resets**.

The key's owner's budget still applies too. The page's **Budget** column shows
each key's spend this period against its limit. See [Budgets](https://docs.otari.ai/learn/budgets).

## Mismatched `user` fields

A request can name a user (`user` on most routes, `metadata.user_id` on
Messages). By default, an API key only accepts its own user there and refuses
anything else with `403`.

Some clients fill that field with a value of their own: Claude Code and Junie
do, and so do applications that pass their customers' ids. For those keys, set
**Mismatched `user` field** under **Advanced** in the key form, or on the edit
form:

| Choice | A request naming another user |
| --- | --- |
| **Use the deployment setting (default)** | Follows the deployment-wide setting, which rejects unless an operator changed it |
| **Always reject (403)** | Rejected, even where the deployment accepts |
| **Always accept** | Accepted, and billed to the key's own user |

On an ordinary key, spend always stays with the key's user, so accepting a
mismatch never lets a key charge someone else.

An application that passes its own customer ids has two options:

- **Always accept**: every request bills the key's user, under one budget.
- Make the key a [service key](end-users.md): each id becomes an end user with
  its own spend and budget.

Through the API, this is the key's `reject_user_mismatch` field (`null`,
`true` or `false`). Self-hosted deployments can also set it in config.yml; see
[Budgets and API keys in config](https://docs.otari.ai/self-hosting/configuration#budgets-and-api-keys-in-config).

## Regenerate, disable or delete a key

Each key's row has an actions menu:

- **Regenerate…** issues a new secret and keeps the key. The old secret stops
  working immediately. The new one is shown once.
- **Disable** switches the key off; callers get `401` until you enable it again.
- **Delete…** removes the key permanently.

A key declared in a self-hosted deployment's config.yml behaves differently;
see [What config.yml owns](https://docs.otari.ai/self-hosting/configuration#what-configyml-owns).

## How it works

### What a key carries

Besides its user and workspace, a key can have:

- an expiry, and an on/off state
- a list of allowed models and tools (see
  [Model and search tool access](#model-and-search-tool-access))
- a budget of its own, which applies alongside its owner's
- a budget exemption, set only by an operator
- a choice on [mismatched `user` fields](#mismatched-user-fields)
- whether content-free agent telemetry is captured
- whether it is a [service key](end-users.md), which can bill end users
- application metadata

A budget-exempt key is also exempt from `require_pricing`. Use such keys only
for usage import or other traffic you mean to observe without enforcing.

A backend on Google Cloud, or behind another OpenID Connect issuer, can
authenticate as a key with its platform identity token instead of the secret.
See [Workload identity](https://docs.otari.ai/self-hosting/operations/workload-identity).

### Who manages which keys

- **Owners, admins and members** create and manage their own keys on the
  **API keys** page. A key made there is always billed to whoever made it, and
  is never budget-exempt.
- **Owners and admins** also create the organization's own
  [service keys](end-users.md#2-create-the-service-key).
- **Viewers** can list their keys but cannot create, edit, regenerate or delete
  one. An operator can issue a key to a viewer.
- **Deployment operators** manage every key and choose each key's owner; see
  [Deployment-wide account administration](https://docs.otari.ai/self-hosting/sign-in#deployment-wide-account-administration).

Membership changes switch keys off:

- Removing someone from a workspace or the organization switches off every key
  billed to them there. Only an operator can switch those back on.
- Making someone a viewer switches off the keys they made for themselves, and
  leaves alone keys an operator issued them. Promoted again, they can switch
  their own keys back on.
- Keys made before Otari recorded who made them are left alone on a demotion.
  When you demote someone to viewer, check the keys billed to them.

The key routes are in the [OpenAPI specification](https://github.com/mozilla-ai/otari/blob/1ca1e752df18d2edadd0c955b4440b02035bb644/docs/public/openapi.json).

### Model and search tool access

**Models.** A new key inherits its user's `allowed_models` unless it sets its
own list. No list allows any model, an empty list allows none, and entries may
use provider wildcards such as `openai:*`. A key can narrow its user's access,
never widen it.

**Search and fetch tools.** These have their own list, `allowed_tools`, which
governs `POST /api/v1/search` and `POST /api/v1/fetch`:

- Entries are search or fetch tool names as the dashboard shows them,
  `builtin_fetch` for Otari's own fetcher, or `*` for every tool.
- A key's list wins over its user's, and may narrow it but never widen it.
- An empty list allows no tool.
- A model wildcard never grants a tool. A search provider can share its name
  with a model provider (DeepSeek does), and `deepseek:*` covers only
  DeepSeek's models.
- The in-loop `otari_web_search` and `otari_web_fetch` tools do not read this
  list.

**Deprecated: tools through the model list.** With no `allowed_tools` on the
key or its user, search access follows the model list instead:

- A key with unrestricted models may run any search or fetch tool.
- A key with a model list may run only the search tools named there in full as
  `<provider>:<tool>` (such as `exa:exa-search`) that its user's list also
  grants. No model list grants a fetch tool.
- Wildcards no longer grant a search tool, and a model list rejects a wildcard
  or partial name under a search-only provider such as `exa:*`.

A later release removes this form, so give keys and users an `allowed_tools`
list. When a user first gets one, each of their keys that has its own model list
and no `allowed_tools` gets the tools it could already run that are also on the
user's new list, so the switch never widens a key. A list sent back unchanged is
not checked again, so a key naming a since-deleted tool can still save an
unrelated change. No search tool may be named `*`.
