# Organizations and workspaces

This guide creates, organizes and deletes workspaces, and deletes an
organization. An organization is your company or team, and a workspace is the
boundary for one app, environment, team or customer inside it; what each one
owns, and the roles members hold in them, is under [How it works](#how-it-works).
On otari.ai
the organization also holds the [balance](billing.md) that every workspace
spends from.

Managing workspaces needs an owner or admin of the organization.

## Choose how to split workspaces

If two workloads need different keys, budgets, access or reporting, give them
different workspaces. Common splits:

- **Per application**: one workspace per service or product, so usage and budgets stay separate.
- **Per environment**: `myapp-dev`, `myapp-staging` and `myapp-prod`.
- **Per team**: isolate keys and spending for different teams or departments.
- **Per customer**: for SaaS workloads, one workspace per customer for isolated tracking.

A `{product}-{environment}` naming convention scales better than ad hoc names once
you have more than a handful.

## Create a workspace

1. Open **Organization**, then **Workspaces**.
2. Click **Create workspace**. The workspace switcher offers **Create workspace**
   too.
3. Enter a **Name** and, optionally, a **Description**, and click
   **Create workspace**.

A new workspace can use every provider key its organization holds. To choose
which key a workspace uses for a provider, click **Edit** (the pencil icon) on
the workspace; see [Control workspace access](provider-keys.md#control-workspace-access).

The API, an SDK or automation addresses a workspace by its id, which
`GET /api/v1/workspaces` lists.

## Delete a workspace

1. Open **Organization**, then **Workspaces**.
2. Find the workspace, click **Delete** (the bin icon), and confirm with
   **Delete workspace**.

The workspace goes with its memberships. The gateway refuses to delete one that
still holds API keys, usage, aliases or routing policies, so a workspace's spend
history outlives it: move or delete those first. It also refuses to delete an
organization's last workspace; create another before deleting it.

## Delete an organization

1. Open **Organization**, then **Org settings**.
2. Click **Delete organization** and confirm.

Which organizations can be deleted, what goes with one, and where its members
land afterwards is in [Deleting an organization](#deleting-an-organization).

## How it works

### What each one owns

An organization is the tenant boundary. It owns workspaces, members, provider
keys, pricing overrides, guardrail policy, and organization-wide usage views.

A workspace groups the resources used by a team or application. API keys, usage,
aliases, routing policies, MCP servers, and tool policy are resolved in a
workspace.

### Roles

Organization and workspace memberships use four roles:

| Role | Meaning |
| --- | --- |
| Owner | Full management, including organization administration |
| Admin | Manage most organization or workspace resources |
| Member | Use the workspace and read permitted resources |
| Viewer | Read-only access |

### Workspace-scoped spend

A request's usage lands in the workspace of the API key that authenticated it
(see [How a request is attributed](access-control.md#how-a-request-is-attributed)).
Organization and workspace usage views then apply the signed-in identity's
membership. Deployment operators can read the operator usage API, which covers
every workspace of the operator's active organization (the default organization
for the master key).

Routing fallback and built-in tools can create several internal attempts, but a
successful request remains one caller-visible request. The activity log records
the attempt group and the model that served it.

### Deleting an organization

An owner or admin can delete their active organization, in the dashboard or
with `DELETE /api/v1/organizations/me`, while it has no usage: the gateway
refuses an organization whose workspaces still hold usage, API keys, aliases or
routing policies, so a deletion never takes spend history with it. The
deployment's default organization and the only organization the caller belongs
to cannot be deleted either. Its workspaces, memberships, pending invitations
and organization-level settings go with it. Everyone whose active organization
it was moves to the oldest other organization they belong to; someone who
belongs to no other organization is left with none until they join one.
