---
icon: globe
---

# Configure web search

Each workspace decides whether its requests can search and fetch the web, and
with which keys. These steps are in the dashboard; the settings they change are
described in [Built-in tools](tools.md).

A workspace setting can only narrow what the deployment allows: it cannot allow
web access the deployment blocks.

1. Select the workspace in the top-left corner.
2. Go to **Tools**, then **Web search**.
3. Turn **Allow in <workspace>** on or off. Off refuses the web search and web
   fetch Otari runs for the workspace's requests, and direct searches too. A
   provider's own search or fetch tool that the provider runs itself is not
   affected; step 5 decides who runs those.
4. Under **Search with** and **Fetch with**, choose which of the organization's
   keys the workspace uses, or the organization's default, or the deployment's
   default tool. Only an owner or admin of the workspace or of the organization
   can choose; any other member sees the list read-only.
5. Under **Who runs provider web search** and **Who runs provider web fetch**,
   choose whether a provider's own search and fetch tools run on the provider or
   on Otari, or leave them **Not set** for the request and the deployment to
   decide.
6. Under **Advanced**, set **Max results for this workspace** to cap the results
   each search returns, use **Allowed domains** to limit searches and fetched
   pages to some sites, or **Blocked domains** to keep them away from some, and
   set the options of the provider the workspace's searches run on. Direct
   searches apply the switch and the result cap, but not the domain lists or
   the provider options; see
   [Per-workspace web-access policy](tools.md#per-workspace-web-access-policy).

The policy these steps set is described in
[Per-workspace web-access policy](tools.md#per-workspace-web-access-policy), the
keys in [Organization search keys](tools.md#organization-search-keys), and who
runs a provider's own tools in
[Web-search interception](tools.md#web-search-interception). The purpose hint is
set through the API.
