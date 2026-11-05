---
icon: computer-classic
---

# Add an MCP server

Each workspace decides which MCP servers its requests can reach. These steps are
in the dashboard; how a request uses the servers is described in [MCP](https://docs.otari.ai/reference/gateway/mcp).

1. Select the workspace in the top-left corner.
2. Go to **Tools**, then **MCP servers**, and click **Add MCP server**.
3. Enter a **Name**, unique in the workspace. Its tools are named after it.
4. Enter the server's **URL**, which the gateway must be able to reach, and an
   **Authorization token** if the server needs one.
5. Optionally, add a **Purpose hint** saying when the model should use the
   server, for example "Use for repository and issue lookups."
6. Under **Allowed tools**, list the tools the model may call, separated by
   commas, or leave it blank to allow every tool the server offers.
7. Leave **Enabled** on, and click **Add MCP server**.

Requests in the workspace can use the server's allowed tools from then on.
Turning **Enabled** off keeps the server's settings but stops requests from
using it. How a request reaches these servers, and their limits, are in
[Workspace-scoped servers](https://docs.otari.ai/reference/gateway/mcp#workspace-scoped-servers).
