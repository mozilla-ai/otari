# Access control

Otari separates who runs the deployment, who signs in to the dashboard, the
credentials applications send, and the limits on what they spend. This page
shows how those pieces fit together. Each one has its own page with the steps.

| Concept | What it is | Page |
| --- | --- | --- |
| Organization | The tenant boundary: owns workspaces, members, provider keys and organization-wide settings | [Organizations and workspaces](workspaces.md) |
| Workspace | One app's, environment's, team's or customer's resources inside an organization | [Organizations and workspaces](workspaces.md) |
| Member | A person's membership of an organization and some of its workspaces, with a role in each | [Members and invitations](members.md) |
| User | Who a request's spend is attributed to, and who per-user budgets apply to | [Sign-in and accounts](sign-in.md#users-and-identities) |
| Identity | Who signs in to the dashboard | [Sign-in and accounts](sign-in.md) |
| API key | A workload credential bound to one user and one workspace | [API keys](api-keys.md) |
| Service key and end user | A key whose requests bill the people an application serves | [Track spend per end user](end-users.md) |
| Budget | A spend, token or request limit over a period | [Budgets](budgets.md) |
| Provider key | The organization's credential for an upstream LLM provider | [Provider keys](provider-keys.md) |
| Operator | Runs a self-hosted deployment as a whole | [Deployment-wide account administration](sign-in.md#deployment-wide-account-administration) |

## How they relate

- **People** belong to an organization and to some of its workspaces, with one
  of four [roles](workspaces.md#roles) in each. A person signs in as an
  identity, and their spend is recorded against the user behind it.
- **Applications** never sign in. They send an API key, which ties each request
  to one workspace and one user.
- **The workspace** decides which provider keys, aliases, routing policies and
  tools a request can use. Each stored MCP server's
  [access list](mcp.md#server-access) decides which keys reach it.
- **The user**, narrowed by the key, decides which models a request can reach,
  and the user's budget is the one it spends.
- **Provider keys** belong to the organization; applications never see them.
- **Budgets** sit over all of it. A per-user budget limits each user on it, and
  a spend ceiling caps an organization, a workspace, a membership or a single
  API key. A request has to fit every budget that applies.

Organization roles stop at their organization. On a self-hosted deployment,
[operators](sign-in.md#deployment-wide-account-administration) act on the whole
deployment.

## What a page shows depends on who is signed in

- **Spend & budgets.** An organization owner or admin manages their
  organization's budgets and ceilings. A deployment operator also sees the
  deployment's own budgets and the gateway users assigned to them.
- **API keys.** An operator manages every key in the organization and picks each
  key's owner. A member sees only their own keys, always billed to themselves and
  never budget-exempt. An operator, or an organization owner or admin, can cap a
  key's spend from its create and edit forms, and the Budget column shows each
  key's spend against its limit. Where the deployment enables it, a key's actions
  menu offers **Workload identity…**; see
  [Rules on a member's own key](workload-identity.md#rules-on-a-members-own-key).
- **Overview and Routing** show an operator their active organization only. What
  the Overview flags for each role is in
  [What needs attention](review-usage-and-requests.md#what-needs-attention).

## How a request is attributed

The API key decides a request's workspace; no header can pick another. Usage is
recorded in that workspace and spend is billed to the key's user.

A `user` value in the request never moves spend away from the key's user. By
default a request naming another user is refused, and a key set to accept it
still bills its own user (see
[Mismatched `user` fields](api-keys.md#mismatched-user-fields)). The exception
is a [service key](end-users.md), which bills end users that belong to its own
user.

A dashboard session works on the management API, not for inference; see
[Dashboard sessions and identity](sign-in.md#dashboard-sessions-and-identity).
