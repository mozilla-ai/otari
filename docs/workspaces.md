# Organizations and workspaces

A workspace is the boundary for one app, environment, team, or customer. It keeps provider access, API keys, budgets, usage history, and members separate from other workspaces.

If two workloads need different keys, budgets, access, or reporting, they should usually live in different workspaces.

Most dashboard tools are scoped to the active workspace, including Overview, Models, Routing, Tools, API keys, and Members.

## What's inside a workspace?

### Provider access

Provider keys are managed at the organization level. By default, a workspace can use any of them. You can pin a workspace to a specific provider key, or disable keys it shouldn't use.

### API keys

Workspace-scoped keys that your application uses to call otari.ai. API keys belong to their workspace.

### Budgets and usage

Spending limits, token counts, costs, and request logs for everything the workspace processes.

### Members and routing

Workspace-specific members, routing rules, MCP server settings, and other runtime behavior for that workload.

## What lives at the organization level instead

A few things are intentionally above workspaces and apply across the whole organization:

* **Spend & budgets**: at the org level, which roll up across workspaces.
* **Providers**: which determine what providers and models are available to the organization.
* **Organization** settings and org-level members.
* **Balance**: which is shared across all workspaces in the organization; see [Billing](billing.md).

The simplest mental model is: organization equals your company, and workspace equals a project or environment inside it.

## How teams use workspaces

Common patterns:

* **Per application**: one workspace per service or product so usage and budgets stay separate
* **Per environment**: `myapp-dev`, `myapp-staging`, and `myapp-prod`
* **Per team**: isolate keys and spending for different teams or departments
* **Per customer**: for SaaS workloads, one workspace per customer for isolated tracking and billing visibility

A naming convention helps. `{product}-{environment}` scales better than ad hoc names once you have more than a handful.

## Roles

Organization and workspace memberships use four roles:

| Role | Meaning |
| --- | --- |
| Owner | Full management, including organization administration |
| Admin | Manage most organization or workspace resources |
| Member | Use the workspace and read permitted resources |
| Viewer | Read-only: sees the API keys billed to them but cannot create, change, rotate or delete one, and cannot run Playground completions or uploads |

## Workspace-scoped spend

A request's usage lands in the workspace of the API key that sent it (see [How a request is attributed](access-control.md#how-a-request-is-attributed)). A request that falls back to another model still counts as one request.

## Constraints worth knowing

* A workspace does not own provider keys. It controls which organization-level provider keys are available to it.
* A workspace that still holds API keys, aliases, routing policies, or usage cannot be deleted. Usage from requests Otari served is kept for good, so a workspace that has served requests stays. The organization's last workspace cannot be deleted either.
* Workspace IDs are stable. If an application is already wired to a workspace ID, migrate it before deleting that workspace.
* An owner or admin can delete an organization from **Organization** → **Org settings**, under the same condition: none of its workspaces may hold usage, API keys, aliases, or routing policies. Members move to another organization they belong to.

Ready to create a workspace? See [Manage workspaces](manage-workspaces.md).
