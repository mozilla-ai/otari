# Configure a workspace's tools

Each workspace decides which built-in tools its requests can use and which MCP
servers they can reach. These steps are in the dashboard; the settings they
change are described in [Built-in tools](tools.md) and [MCP](mcp.md).

A workspace setting can only narrow what the deployment allows: it cannot allow
web access the deployment blocks, or turn on code execution when the deployment
has no sandbox.

## Configure web search

1. Select the workspace in the top-left corner.
2. Go to **Tools**, then **Web search**.
3. Under **Web access**, choose **Follow the deployment default**, **Allowed** or
   **Blocked**. Blocked refuses both web search and web fetch for the
   workspace's requests.
4. Set **Max results** to cap the results each search returns.
5. Optionally, add a **Prompt hint** saying when to search, used when a request
   gives no instructions of its own.
6. Use **Allowed domains** to limit searches and fetched pages to some sites, or
   **Blocked domains** to keep them away from some.

The policy these steps set is described in
[Per-workspace web-access policy](tools.md#per-workspace-web-access-policy).

## Configure code execution

1. Select the workspace in the top-left corner.
2. Go to **Tools**, then **Code execution**.
3. Under **Code execution**, choose **Follow the deployment default**,
   **Allowed** or **Blocked**.
4. Optionally, add a **Prompt hint** saying when to run code, used when a
   request gives no instructions of its own.
5. Set **Max tool-loop iterations** to cap the model-to-tool rounds of one
   request, and **Execution timeout** to cap how long one run can take.
6. Under **Who runs provider code tools**, choose whether a provider's own code
   tool or Otari's sandbox runs the code.
7. Where offered, pick the **Sandbox image** from those the deployment approves,
   and choose which code-execution tools the workspace can use.

The policy these steps set is described in
[Per-workspace code policy](tools.md#per-workspace-code-policy), and who runs the
code in [Code-execution executor](tools.md#code-execution-executor).

## Add an MCP server

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
[Workspace-scoped servers](mcp.md#workspace-scoped-servers).
