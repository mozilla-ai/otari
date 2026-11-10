---
icon: computer-classic
---

# Add an MCP server

MCP servers allow requests from a workspace to connect to external tools and services. Each server is configured at the **workspace level**, so you can control which MCP tools are available to requests within that workspace.

There are two ways in: add a server by its URL, or pick servers from the catalog.

## Add a server by URL

1. Select the workspace in the top-left corner.
2. Go to **Tools** → **MCP servers**.
3. Click **Add MCP server**.
4. Enter a **Name** for the server. This must be unique within the workspace and is used to identify the server’s tools.
5. Enter the MCP server **URL**. The endpoint must be reachable from the Otari gateway.
6. If required, enter an **Authorization token**.
7. Optionally, add a **Purpose hint** describing when the model should use this server, for example, “Use for repository and issue lookups.”
8. Under **Allowed tools**, specify which tools from the MCP server the model can access. Enter the tool names separated by commas, or leave the field blank to make all tools offered by the server available.
9. Use **Enabled** to control whether the MCP server is available for requests. You can disable it without removing its configuration.
10. Under **Default policy**, choose **Allow** to let every key in the workspace use the server, or **Deny** to let none use it until you add rules for them.
11. Click **Add MCP server** to save it.

![The New MCP server dialog, with a name, URL, token, purpose hint and allowed tools filled in.](assets/screenshots/add-an-mcp-server.png)

## Add servers from the catalog

The catalog lists remote MCP servers from the official MCP Registry, so you do not have to find a URL. The servers your organization recommends are listed first. Being listed permits nothing: a server is available to a workspace only once it is added there.

1. Select the workspace in the top-left corner.
2. Go to **Tools** → **MCP servers**.
3. Click **Browse catalog**.
4. Type in **Search the catalog** to find a server by its name or description.
5. Select each server you want. A server the workspace already has is marked and cannot be selected again.
6. Click **Review**.
7. Check each name, which must be unique within the workspace. If a server asks for an authorization token, paste it, or leave it blank and set it later on the server's page.
8. Click the button that adds the servers.

Each server is added and its connection is checked, one after another, and the dialog then reports each one:

- **Connected**: the server answered, and the line says how many tools it offers.
- **Needs sign-in**: the server was added, but it wants a credential before it answers. Give it a token on its page.
- **Check failed**: the server was added, but its connection check failed, and the line says why.
- **Not added**: the gateway refused it, and the line gives its reason, such as a name already in use or the workspace's limit of 50 servers.

A server added from the catalog lets every key in the workspace use it, as **Allow** does when you add one by URL. The catalog's description is not stored as a purpose hint, because a purpose hint is sent to the model.

If the list is empty, the deployment has not synced the catalog, or its operator switched the sync off. See [MCP server catalog](mcp-catalog.md). Adding a server by URL works either way.

## After a server is added

Once added, the MCP server becomes available to requests in that workspace according to the configuration and allowed tools you defined. To limit it to some keys or users, see [Control access to an MCP server](control-access-to-an-mcp-server.md). How requests use MCP servers is described in [MCP](mcp.md).
