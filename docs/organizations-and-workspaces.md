# Organizations and workspaces

This guide creates, organizes and deletes workspaces, and deletes an
organization. An organization is your company or team, and a workspace is the
boundary for one app, environment, team or customer inside it; what each one
owns, and the roles members hold in them, is in
[Access control](access-control.md#organizations-and-workspaces). On otari.ai
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

Only an organization with no usage, API keys, aliases or routing policies can be
deleted. Which organizations can never be deleted, what goes with one, and where
its members land afterwards is in
[Access control](access-control.md#organizations-and-workspaces).
