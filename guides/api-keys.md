# API keys

An API key is the credential an application sends to Otari. It belongs to one
workspace and one user, so it decides where a request's usage and spend land.
What a key can carry, and who may manage which keys, is in
[Access control](access-control.md#api-keys).

Send the key as a bearer token (`Authorization: Bearer <key>`); the other
accepted headers are in [Authentication](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/reference#authentication). The
base URL to send it to, and the form each SDK expects, are in
[Connect to Otari](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/connect).

## Create a key

1. Open the workspace's **API keys** page and click **Create key**.
2. Enter a **Name** and, optionally, an expiry under **Expires (optional)**.
3. Optionally, click **Advanced** to narrow the models the key can use. By
   default a key inherits your model access, and it can never exceed it.
4. Click **Create key**, then copy the key. It is shown once, so store it
   before you close the dialog.

A member's keys are always their own and billed to them. A deployment operator
also chooses each key's owner.

## Cap a key's spend

An operator, or an organization owner or admin, can give a key a budget of its
own: in the key's create or edit form, pick an existing budget under
**Budget (optional)**, or set a **Limit (USD)** and how often it **Resets**. The
key's owner's budget still applies beside it. The page's Budget column shows
each key's spend this period against its limit. For budgets in general, see
[Budgets](budgets.md).

## Accept or reject mismatched `user` fields

When the client behind a key puts something other than the key's own user in
the request's `user` field (Claude Code does), set the key's choice under
**Advanced** when creating it, or on its edit form: use the deployment setting,
always reject, or always accept. What each choice does is in
[Mismatched `user` fields](access-control.md#mismatched-user-fields). To give
each of an application's end users spend and a budget of their own instead, use
a service key: [Track spend per end user](end-users.md).

## Regenerate or delete a key

- **Regenerate** on a key's row issues a new secret and keeps the key record.
  The current secret stops working immediately, with no grace period.
- **Delete** on a key's row removes the key permanently.

A key declared in `config.yml` is owned by that file; see
[Budgets and API keys in config](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/self-hosting/configuration#budgets-and-api-keys-in-config).
To authenticate a backend with its platform identity instead of a stored
secret, see [Workload identity](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/self-hosting/operations/workload-identity).
