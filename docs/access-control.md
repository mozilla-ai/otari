# Access control

Otari separates who runs the deployment, who signs in to the dashboard, the
credentials workloads send, and the limits on what they spend. This page
explains how those pieces fit together; each one has a page of its own with the
steps and the rules.

| Concept | What it is | Page |
| --- | --- | --- |
| Organization | The tenant boundary: owns workspaces, members, provider keys and organization-wide settings | [Organizations and workspaces](organizations-and-workspaces.md) |
| Workspace | One app's, environment's, team's or customer's resources inside an organization | [Organizations and workspaces](organizations-and-workspaces.md) |
| Member | A person's membership of an organization and some of its workspaces, with a role in each | [Members and invitations](members.md) |
| User | Who a request's spend is attributed to, and who per-user budgets apply to | [Sign-in and accounts](sign-in.md#users-and-identities) |
| Identity | Who signs in to the dashboard | [Sign-in and accounts](sign-in.md) |
| API key | A workload credential bound to one user and one workspace | [API keys](api-keys.md) |
| Service key and end user | A key whose requests bill the people an application serves | [Track spend per end user](end-users.md) |
| Budget | A spend, token or request limit over a period | [Budgets](budgets.md) |
| Provider key | The organization's credential for an upstream LLM provider | [Provider keys](provider-keys.md) |
| Master key and operators | Authority over the deployment as a whole | [Deployment-wide account administration](sign-in.md#deployment-wide-account-administration) |

## How they relate

An organization holds workspaces, and a member belongs to the organization and
to the workspaces they were given, with one of four
[roles](organizations-and-workspaces.md#roles) in each. A member signs in as an
identity, and the user record behind that identity is what their spend is
recorded against.

Applications never sign in. They send an API key, and the key ties each request
to one workspace and one user. The workspace decides which provider keys,
aliases, routing policies and tools the request can use; the user, narrowed by
the key, decides which models it can reach, and the user's budget is the one it
spends. The provider keys
Otari calls upstream with belong to the organization, and the application never
sees them.

Budgets sit over all of it. A per-user budget limits each user on it, and a
scoped budget caps an organization, a workspace, a membership or a single API
key. A request has to fit every budget that applies.

The master key, and an identity granted operator access, act on the whole
deployment; organization roles stop at their organization.

## How a request is attributed

The API key that authenticates a request determines its workspace. A caller
cannot select another workspace with a header, and an inference request on the
master key uses the deployment's default workspace. The request's usage is
recorded in that workspace and its spend billed to the key's user.

A client-provided `user` value never moves spend away from the key's user:
by default a request naming another user is refused, and a key set to accept it
still bills its own user (see [Mismatched `user` fields](api-keys.md#mismatched-user-fields)).
The one exception is a [service key](end-users.md), which bills end users that
belong to its own user.

A dashboard session authorizes the management API, not inference; see
[Dashboard sessions and identity](sign-in.md#dashboard-sessions-and-identity).

The generated [OpenAPI specification](public/openapi.json) is the source of
truth for endpoint schemas.
