---
icon: computer-classic
---

# Add an MCP server

Each workspace decides which MCP servers its requests can reach. These steps are
in the dashboard; how a request uses the servers is described in [MCP](mcp.md).

1. Select the workspace in the top-left corner.
2. Go to **Tools**, then **MCP servers**, and click **Add MCP server**.
3. Enter a **Name**, unique in the workspace. Its tools are named after it.
4. Enter the server's **URL**, which the gateway must be able to reach, and an
   **Authorization token** if the server needs one.
5. Optionally, add a **Purpose hint** saying when the model should use the
   server, for example "Use for repository and issue lookups."
6. Under **Allowed tools**, list the tools the model may call, separated by
   commas, or leave it blank to allow every tool the server offers.
7. Leave **Enabled** on.
8. Under **Default policy**, choose **Allow** to let every key in the workspace
   use the server, or **Deny** to let none use it until you add rules for them.
9. Click **Add MCP server**.

With **Allow**, every key in the workspace can use the server's allowed tools
from then on. To limit it to some keys or users, or to read-only tools, see
[Control access to an MCP server](control-access-to-an-mcp-server.md).
Turning **Enabled** off keeps the server's settings but stops requests from
using it. How a request reaches these servers, and their limits, are in
[Workspace-scoped servers](mcp.md#workspace-scoped-servers).
