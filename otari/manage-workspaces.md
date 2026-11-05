---
description: Create, switch, organize, and delete workspaces in the dashboard.
icon: desk
---

# Manage workspaces

A workspace holds the API keys, routing policies, tools and usage of one team,
application or environment. What workspaces are, and how they fit in an
organization, is in [Organizations and workspaces](workspaces.md).

## Create a workspace

1. Click **Organization** in the bottom-left corner. It opens on **Workspaces**.
2. Click **Create workspace**.
3. Give it a name and, optionally, a description, and create it.

A new workspace uses every provider key the organization has. To choose which
key it uses for a provider, click **Edit** (the pencil icon) on the workspace;
see [Control which keys a workspace uses](manage-provider-keys.md#control-which-keys-a-workspace-uses).

## Find your way around a workspace

Select the workspace in the top-left corner. Its menu has:

- **Overview**: its requests, spend and sessions.
- **Models**: the models its requests can use.
- **Routing**, then **Policies**: its [routing policies](create-a-multi-provider-routing.md).
- **Guardrails**: the guardrails that run on its requests.
- **Tools**: its [web search](configure-web-search.md), [code execution](configure-code-execution.md) and [MCP servers](add-an-mcp-server.md).
- **API keys** and **Members**: who and what can call it.

The organization's own settings are under **Organization**: **Usage**,
**Workspaces**, **Members & roles**, **Spend & budgets**, **Providers**,
**Guardrails** and **Org settings**.

## Organize workspaces by environment

A common pattern is one workspace per environment (`production`, `staging`,
`development`) or per application. Each gets its own API keys and its own
budget, so a runaway test cannot spend production's money, and usage reads per
environment without filtering.

## Delete a workspace

1. Open **Organization**, then **Workspaces**.
2. Find the workspace, click **Delete** (the bin icon), and confirm.

A workspace that still holds API keys, aliases, routing policies or usage cannot
be deleted. Keys, aliases and policies can be removed first, but the usage of
requests the gateway served is kept for good, so a workspace that has served
requests stays; only imported usage can be deleted. The organization's last
workspace cannot be deleted either. Move every application that still uses the
workspace before you delete it.
