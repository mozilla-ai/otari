# Configure a workspace's tools

Each workspace decides which built-in tools its requests can use and which MCP
servers they can reach. These steps are in the dashboard; the settings they
change are described in [Built-in tools](https://docs.otari.ai/reference/gateway/tools) and [MCP](https://docs.otari.ai/reference/gateway/mcp).

A workspace setting can only narrow what the deployment allows: it cannot allow
web access the deployment blocks, or turn on code execution when the deployment
has no sandbox.

## Configure web search

1. Select the workspace in the top-left corner.
2. Go to **Tools**, then **Web search**.
3. Turn **Allow in <workspace>** on or off. Off refuses both web search and
   web fetch for the workspace's requests, and direct searches too.
4. Under **Search with** and **Fetch with**, choose which of the organization's
   keys the workspace uses, or the organization's default, or the deployment's
   default tool. Only a workspace owner or admin can choose; any other member
   sees the list read-only.
5. Under **Who runs provider web search** and **Who runs provider web fetch**,
   choose whether a provider's own search and fetch tools run on the provider or
   on Otari, or leave them **Not set** for the request and the deployment to
   decide.
6. Under **Advanced**, set **Max results for this workspace** to cap the results
   each search returns, use **Allowed domains** to limit searches and fetched
   pages to some sites, or **Blocked domains** to keep them away from some, and
   set the options of the provider the workspace's searches run on.

The policy these steps set is described in
[Per-workspace web-access policy](https://docs.otari.ai/reference/gateway/tools#per-workspace-web-access-policy), the
keys in [Organization search keys](https://docs.otari.ai/reference/gateway/tools#organization-search-keys), and who
runs a provider's own tools in
[Web-search interception](https://docs.otari.ai/reference/gateway/tools#web-search-interception). The purpose hint is
set through the API.

## Configure code execution

1. Select the workspace in the top-left corner.
2. Go to **Tools**, then **Code execution**.
3. Turn **Allow in <workspace>** on or off.
4. Under **Who runs provider code tools**, choose whether a provider's own code
   tool or Otari's sandbox runs the code.

The other fields of the policy, such as the iteration cap, the execution timeout
and the sandbox image, are set through the API.

The policy these steps set is described in
[Per-workspace code policy](https://docs.otari.ai/reference/gateway/tools#per-workspace-code-policy), and who runs the
code in [Code-execution executor](https://docs.otari.ai/reference/gateway/tools#code-execution-executor).

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
[Workspace-scoped servers](https://docs.otari.ai/reference/gateway/mcp#workspace-scoped-servers).
