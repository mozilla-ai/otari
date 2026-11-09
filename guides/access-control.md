# Access control

Otari separates who runs the deployment, who signs in to the dashboard, the
credentials workloads send, and the limits on what they spend. This page
explains how those pieces fit together; each one has a page of its own with the
steps and the rules.

| Concept | What it is | Page |
| --- | --- | --- |
| Organization | The tenant boundary: owns workspaces, members, provider keys and organization-wide settings | [Organizations and workspaces](https://docs.otari.ai/learn/workspaces) |
| Workspace | One app's, environment's, team's or customer's resources inside an organization | [Organizations and workspaces](https://docs.otari.ai/learn/workspaces) |
| Member | A person's membership of an organization and some of its workspaces, with a role in each | [Members and invitations](members.md) |
| User | Who a request's spend is attributed to, and who per-user budgets apply to | [Sign-in and accounts](https://docs.otari.ai/self-hosting/sign-in#users-and-identities) |
| Identity | Who signs in to the dashboard | [Sign-in and accounts](https://docs.otari.ai/self-hosting/sign-in) |
| API key | A workload credential bound to one user and one workspace | [API keys](api-keys.md) |
| Service key and end user | A key whose requests bill the people an application serves | [Track spend per end user](end-users.md) |
| Budget | A spend, token or request limit over a period | [Budgets](https://docs.otari.ai/learn/budgets) |
| Provider key | The organization's credential for an upstream LLM provider | [Provider keys](https://docs.otari.ai/learn/provider-keys) |
| Operator | Runs a self-hosted deployment as a whole | [Deployment-wide account administration](https://docs.otari.ai/self-hosting/sign-in#deployment-wide-account-administration) |

## How they relate

An organization holds workspaces, and a member belongs to the organization and
to the workspaces they were given, with one of four
[roles](https://docs.otari.ai/learn/workspaces#roles) in each. A member signs in as an
identity, and the user record behind that identity is what their spend is
recorded against.

Applications never sign in. They send an API key, and the key ties each request
to one workspace and one user. The workspace decides which provider keys,
aliases, routing policies and tools the request can use, and each stored MCP
server's [access list](https://docs.otari.ai/reference/gateway/mcp#server-access) decides which keys reach it; the user, narrowed by
the key, decides which models it can reach, and the user's budget is the one it
spends. The provider keys
Otari calls upstream with belong to the organization, and the application never
sees them.

Budgets sit over all of it. A per-user budget limits each user on it, and a
scoped budget caps an organization, a workspace, a membership or a single API
key. A request has to fit every budget that applies.

Organization roles stop at their organization; on a self-hosted deployment,
[operators](https://docs.otari.ai/self-hosting/sign-in#deployment-wide-account-administration) act on the whole
deployment.

## How a request is attributed

The API key that authenticates a request determines its workspace. A caller
cannot select another workspace with a header. The request's usage is
recorded in that workspace and its spend billed to the key's user.

A client-provided `user` value never moves spend away from the key's user:
by default a request naming another user is refused, and a key set to accept it
still bills its own user (see [Mismatched `user` fields](api-keys.md#mismatched-user-fields)).
The one exception is a [service key](end-users.md), which bills end users that
belong to its own user.

A dashboard session authorizes the management API, not inference; see
[Dashboard sessions and identity](https://docs.otari.ai/self-hosting/sign-in#dashboard-sessions-and-identity).

The generated [OpenAPI specification](https://github.com/mozilla-ai/otari/blob/53930f2870d0fd22454c187bfc1880e800a764d0/docs/public/openapi.json) is the source of
truth for endpoint schemas.
