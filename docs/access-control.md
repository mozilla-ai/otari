# Access control

Otari separates deployment administration, human identity, workload credentials,
and spend limits. This page explains how those pieces relate. The generated
[OpenAPI specification](public/openapi.json) is the source of truth for endpoint
schemas.

## Deployment-wide account administration

The master key controls the deployment. A dashboard session can perform
deployment-wide operations only when its identity has operator authority.
Organization roles do not implicitly grant access to process-wide settings or
credentials.

Use the master key through `Authorization: Bearer <master-key>` or
`Otari-Key: <master-key>`. Applications should use scoped API keys instead.

## Organizations and workspaces

An organization is the tenant boundary. It owns workspaces, members, provider
keys, pricing overrides, guardrail policy, and organization-wide usage views.

A workspace groups the resources used by a team or application. API keys, usage,
aliases, routing policies, MCP servers, and tool policy are resolved in a
workspace.

Organization and workspace memberships use four roles:

| Role | Meaning |
| --- | --- |
| Owner | Full management, including organization administration |
| Admin | Manage most organization or workspace resources |
| Member | Use the workspace and read permitted resources |
| Viewer | Read-only access |

The API key that authenticates a request determines its workspace. A caller
cannot select another workspace with a header. Master-key inference uses the
default workspace.

## Users and identities

Otari maintains identities for dashboard sign-in and user records for request
attribution and per-user budgets. Management flows connect them where needed.
Client-provided `user` values are never trusted to move spend away from the API
key's bound user.

A user's `allowed_models` is inherited by newly created keys unless the key
defines its own list. A missing list allows any model, an empty list allows none,
and entries may use provider wildcards such as `openai:*`.

Deleting or deactivating a user prevents future access but preserves historical
usage.

## API keys

API keys are workload credentials. Each key has a fixed user and workspace and
may also define:

- expiration and active status
- allowed models
- budget exemption
- whether mismatched client `user` fields are accepted
- whether content-free agent telemetry is captured
- application metadata

The plaintext key is returned only when it is created or rotated. Store it then.
Rotation preserves the key record and invalidates the previous secret.

A budget-exempt key is also exempt from `require_pricing`. Reserve such keys for
usage import or other intentional observability-only traffic.

`/v1/keys` manages every key in the caller's organization and requires the
deployment operator's standing. A signed-in member without it manages their own
keys at `/v1/organizations/me/keys`, which derives the owner rather than
accepting one, mints only into a workspace the caller may see, and never issues
a budget-exempt key.

## Budgets

Otari enforces budgets before dispatch and reconciles actual cost afterwards.
Requests must pass every applicable limit.

Two budget forms exist:

- A per-user budget limits each attached user independently.
- A scoped budget limits an organization, workspace, membership, or API key and
  can optionally narrow the limit to one provider.

A budget caps up to three things over its period, each set independently and
each unlimited when left unset: spend in USD (`max_budget`), total tokens
(`token_limit`), and requests (`request_limit`). A request must have room on
every axis the budget caps. Spend and tokens are held at an upper bound before
dispatch and reconciled to the measured figures afterwards; only the unused part
of an over-estimate is released, and what the request measurably used stays
charged. A request counts as one request when it is admitted. A model priced at
zero spends no dollars, and still spends tokens and one request.
Endpoints that hold no token estimate (embeddings, rerank, and the other
pass-through routes) are refused once a token cap is exhausted rather than
reserving headroom for themselves, so a token cap can be passed by the requests
already in flight when it runs out.

Scoped budgets can use a rolling duration or a UTC calendar boundary. A key with
`exclude_from_budget`, or a deployment with `budget_strategy: disabled`,
bypasses enforcement.

Imported usage is retrospective and never counts toward a budget. Batch cost is
also settled after submission, so operators should not treat those paths as a
hard real-time cap. Batch settles dollars alone: its results arrive outside the
reservation that gated the submission, so a batch counts as the one request that
created it and contributes no tokens to a token cap, however many prompts it
carried. The same holds for the vision side-call a request makes to describe an
attachment. Cap batch-heavy workloads in dollars rather than in tokens. See
[Importing external usage](external-usage.md).

## Workspace-scoped spend

Usage is attributed to the workspace bound to the authenticating API key.
Organization and workspace usage views then apply the signed-in identity's
membership. Deployment operators can read the deployment-wide usage API.

Routing fallback and built-in tools can create several internal attempts, but a
successful request remains one caller-visible request. The activity log records
the attempt group and the model that served it.

## Dashboard sessions and identity

The first operator signs in with the master key. Otari exchanges it for an
opaque, HttpOnly session cookie. After the operator sets an email and password,
the dashboard uses that identity for sign-in; the master key remains an API
credential and recovery path.

Sessions are revocable and expire after `dashboard_session_ttl_hours`. Password
changes, master-key rotation, sign-out, and identity deactivation revoke relevant
sessions.

### Passkeys

Passkeys are optional and additive to password sign-in. Set
`public_base_url` to establish the origin and relying-party ID. Use
`webauthn_rp_id` only when passkeys must be bound to a parent domain.

Changing the relying-party ID makes existing passkeys unusable. The dashboard
continues listing unusable credentials so the owner can remove them.

### OAuth sign-in (Google and GitHub)

OAuth sign-in requires `public_base_url` and the provider's client ID and
secret. Register this redirect URI with the provider:

```text
{public_base_url}/auth/{provider}/callback
```

OAuth signs in an existing Otari identity whose email the provider verifies. It
does not provision arbitrary provider accounts.

## Invitations

An owner or admin can invite a person to an organization and selected workspaces.
If mail is configured, Otari sends the accept link. Otherwise the API and
dashboard expose the link for manual delivery.

Invitation tokens are bearer credentials. Do not put them in logs or analytics.
The browser validates and accepts them through the public invitation endpoints.

A signed-in person also sees the invitations addressed to them, and accepts or
declines one without a token: they are already authenticated as the addressee,
so the membership is addressed by id instead. Declining cancels the invitation
and suspends the paired membership, which is what stops the emailed link from
reviving it; a later invitation to the same address revives the membership.

## Related documentation

- [Admin dashboard](dashboard.md)
- [Configuration](configuration.md)
- [API reference](api-reference.md)
