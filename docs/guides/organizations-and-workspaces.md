# Organizations and workspaces

An organization is your company or team: the tenant boundary that owns
workspaces, members, provider keys, pricing overrides, guardrail policy and
organization-wide usage. A workspace is the boundary for one app, environment,
team or customer inside it. It keeps provider access, API keys, budgets, usage
history and members separate from other workspaces.

If two workloads need different keys, budgets, access or reporting, they should
usually live in different workspaces. Most dashboard pages are scoped to the
active workspace, including Usage, API Keys, Members, Routing and Tools.

## What lives where

| In a workspace | At the organization |
| --- | --- |
| API keys, which live and die with their workspace | Provider keys, which workspaces inherit and can pin or turn off per provider ([Provider keys](provider-keys.md)) |
| Usage, request history and workspace budgets | Organization-wide spend and budgets, which roll up across workspaces ([Budgets](budgets.md)) |
| Members with access to that workspace | Organization members and their roles ([Members and invitations](members.md)) |
| Routing policies, aliases, MCP servers and tool settings | Guardrail policy and pricing overrides |

On otari.ai the organization also holds the [balance](../otari-ai/billing.md) that
every workspace spends from.

## How teams use workspaces

- **Per application**: one workspace per service or product, so usage and budgets stay separate.
- **Per environment**: `myapp-dev`, `myapp-staging` and `myapp-prod`.
- **Per team**: isolate keys and spending for different teams or departments.
- **Per customer**: for SaaS workloads, one workspace per customer for isolated tracking.

A `{product}-{environment}` naming convention scales better than ad hoc names once
you have more than a handful.

## Create a workspace

1. Open **Organization** at the bottom of the menu. It opens on the **Workspaces** tab.
2. Click **Create workspace**.
3. Give it a name and an optional description, and create it.

A new workspace can use every provider key its organization holds. To choose
which key a workspace uses for a provider, click **Edit** (the pencil icon) on
the workspace; see [Control workspace access](provider-keys.md#control-workspace-access).

If you plan to use the API, an SDK or automation against the workspace, copy its
`workspace_id` from the workspace settings.

## Delete a workspace

1. Open **Organization**, which opens on the **Workspaces** tab.
2. Find the workspace, click **Delete** (the bin icon), and confirm.

Deleting a workspace is permanent. It removes the workspace's provider access
and configuration, its usage history, its budgets, and every API key scoped to
it. Move any application that still uses the workspace before you delete it.

## Roles, scoping and deleting an organization

Organization and workspace memberships use four roles:

| Role | Meaning |
| --- | --- |
| Owner | Full management, including organization administration |
| Admin | Manage most organization or workspace resources |
| Member | Use the workspace and read permitted resources |
| Viewer | Read-only access |

The API key that authenticates a request determines its workspace. A caller
cannot select another workspace with a header. Master-key inference uses the
default workspace.

An owner or admin can delete their active organization from Organization →
General (`DELETE /api/v1/organizations/me`) while it has no usage: the gateway
refuses an organization whose workspaces still hold usage, API keys, aliases or
routing policies, so a deletion never takes spend history with it. The
deployment's default organization and the only organization the caller belongs
to cannot be deleted either. Its workspaces, memberships, pending invitations
and organization-level settings go with it. Everyone whose active organization
it was moves to the oldest other organization they belong to; someone who
belongs to no other organization is left with none until they join one.
