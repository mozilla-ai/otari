---
icon: computer-classic
---

# Add an MCP server

MCP servers allow requests from a workspace to connect to external tools and services. Each server is configured at the **workspace level**, so you can control which MCP tools are available to requests within that workspace.

To add an MCP server:

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

Once added, the MCP server becomes available to requests in that workspace according to the configuration and allowed tools you defined. To limit it to some keys or users, see [Control access to an MCP server](control-access-to-an-mcp-server.md). How requests use MCP servers is described in [MCP](https://docs.otari.ai/reference/gateway/mcp).
