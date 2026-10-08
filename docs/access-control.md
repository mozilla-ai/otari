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
key's bound user. The one exception is a [service key](#service-keys-and-end-users),
which bills end users that belong to its own user.

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
- whether it is a service key, which may name end users
- application metadata

The plaintext key is returned only when it is created or rotated. Store it then.
Rotation preserves the key record and invalidates the previous secret.

A budget-exempt key is also exempt from `require_pricing`. Reserve such keys for
usage import or other intentional observability-only traffic.

`/api/v1/keys` manages every key in the caller's organization and requires the
deployment operator's standing. A signed-in member without it manages their own
keys at `/api/v1/organizations/me/keys`, which derives the owner rather than
accepting one, mints only into a workspace the caller may see, and never issues
a budget-exempt key.

## Service keys and end users

A service key lets one application track spend per end user without sharing
the master key or minting a key per end user. Mark a key with `is_service_key`
on `POST` or `PATCH /api/v1/keys`; only a deployment operator can.

A request on a service key names its end user the way any client names a user:
the `user` field on `/v1/chat/completions`, `/v1/responses` and `/v1/search`,
and `metadata.user_id` on `/v1/messages`. Otari then:

- Bills the request to that end user, creating it on first use. An end user is
  a user record owned by the key's user, so a key can only bill end users of
  its own user: two services that both name `alice` get two separate end users,
  and naming another key's user creates an end user of your own rather than
  reaching theirs.
- Caps each end user at a budget copied onto it when it is created: the one
  the request names in the `Otari-End-User-Budget` header, or else the key's
  `end_user_budget_id`. Each end user gets the full limit and its own reset
  period. Changing the key's setting affects end users created afterwards. See
  [Several budgets on one key](#several-budgets-on-one-key) and
  [Managing end users](#managing-end-users).
- Applies the per-minute limits of the end user's budget (`rpm_limit`,
  `tpm_limit`), and any `per: user` rate limit rule, each counting every end
  user on its own. Since the limits live on the budget, two service keys with
  different end-user budgets give their end users different limits, and moving
  an end user to another budget moves its limits too.
- Checks the key's own ceiling as well, so a scoped budget on the API key pools
  every end user behind it. The key's user's per-user budget is not checked for
  an end user's request; use the key's ceiling as the pool.
- Keeps the key's user's rate limit and model allow-list, and any member
  ceiling of the key's user, in force for every end user. The rate limit is
  shared by all of them and is checked before an end user is created.

A request that names nobody, or the key's own user, bills the key's user as it
would on any other key. Blocking the key's user stops its end users too.

Each distinct `user` value creates an end user, whether or not the request is
then admitted, and nothing else caps how many a key can create. Set a rate
limit on a deployment that issues service keys, and send a stable id per end
user rather than a per-session or per-request value. An id is at most 256
characters, and a new one may not contain `/` or be `.` or `..`, so that it
can be named in the path of the end user routes below.

### Several budgets on one key

One key can start its end users on different budgets, for a service whose
features each carry their own per-user limits. List the budgets the key may
assign in `end_user_budget_ids`, and keep `end_user_budget_id` as the default
for a request that names none:

```http
POST /api/v1/keys
{"user_id": "mlpa", "is_service_key": true,
 "end_user_budget_ids": ["end-user-budget-ai", "end-user-budget-memories"],
 "end_user_budget_id": "end-user-budget-ai"}
```

A key with a default and no list may assign the default alone, which is how
every key behaves until it is given a list. When both are set, the default must
be on the list. Each entry must be a deployment budget, not an organization's.
Deleting a budget takes it off every list, but a budget that is some key's
default cannot be deleted until that key's default changes: without one, the
key's new end users would start uncapped.

A request then names the budget for a new end user in a header:

```http
POST /v1/chat/completions
Otari-End-User-Budget: end-user-budget-memories

{"model": "...", "user": "fxa123:memories", "messages": [...]}
```

The header applies only when the request creates the end user. An existing end
user keeps its budget, whatever a later request names, so a request cannot undo
a move an operator made. A budget that is not on the key's list is refused with
403 and the code `end_user_budget_not_allowed`, even for an end user that
already exists. The response carries `Otari-End-User-Budget` with the budget
the end user is on, so a caller can see when it differs from the one it named.

Give budgets ids of your own with `PUT /api/v1/budgets/{budget_id}`, which
creates the budget under that id or replaces it, so the same request can run
at every deploy. An id is up to 128 letters, digits, `.`, `_` and `-`, and
starts with a letter or digit.
`POST /api/v1/budgets` still generates an id, and `GET /api/v1/budgets`
reports how many users are on each budget in `user_count`.

### Managing end users

An end user is addressed by the key and the id the service named it by:

| Request | Does |
|---|---|
| `GET /api/v1/keys/{key_id}/end-users/{external_id}` | Reads the end user: its budget, counters and whether it is blocked |
| `PUT /api/v1/keys/{key_id}/end-users/{external_id}` `{"budget_id": ...}` | Puts the end user on a budget, creating it first if it has not made a request yet (201) |
| `PATCH /api/v1/keys/{key_id}/end-users/{external_id}` `{"blocked": true}` or `{"budget_id": ...}` | Blocks, unblocks or moves the end user |

A budget set this way must be on the key's list too. Moving an end user
restarts its period on the new budget and applies that budget's per-minute
limits, but keeps its spend, tokens and requests so far, as moving a user
through `/api/v1/users` does: an end user that used up one budget can be over
the next one's limits until that period resets. End users belong to the key's
user, so every service key of one user reaches the same end users.

To list end users, use `/api/v1/users`, where each one carries
`parent_user_id` (the key's user) and `external_id`.
`GET /api/v1/users?parent_user_id=...&external_id=...` filters on them, and
`include_total=true` counts every match in the `Otari-Total-Count` header. The
users API can also put an end user on any deployment budget, not only one on a
key's list.

### Where end users apply

End users are supported on the endpoints above. The other endpoints
(embeddings, files, batches and the other pass-through
routes) treat a service key as an ordinary key, so a `user` naming someone else
is handled by the `reject_user_mismatch` setting there. Hybrid mode resolves
users on the platform and does not support service keys.

## Budgets

Otari enforces budgets before dispatch and reconciles actual cost afterwards.
Requests must pass every applicable limit.

Two budget forms exist:

- A per-user budget limits each attached user independently.
- A scoped budget, which the dashboard calls an organization budget, is a limit,
  a reset cycle, and the entities it applies to: the organization, workspaces,
  organization members, workspace members, API keys, providers, and models. Any
  entity can be narrowed to one provider, or to one model on a provider; the
  dashboard offers providers and models as narrowings of the organization. Each
  entity draws on its own allowance of the limit rather than sharing one pool,
  and an entity carries at most one budget.

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

A budget can also limit each user on it per minute: `rpm_limit` requests and
`tpm_limit` tokens. These are counted in `rate_limit_store`, so with Redis they
hold across replicas, and tokens are counted on what each request used, as
LiteLLM counts them: a request is admitted while the user's minute is under the
limit. They apply to a user's own budget on chat completions, messages,
responses and search, not to a scoped budget's entities, and a refusal is a 429
naming the rule `budget`.

A budget's reset cycle is one of: never, every N hours, every N days, daily,
weekly on chosen weekdays, monthly on a day of the month (1 to 28), or yearly on
a date. Calendar cycles reset at 00:00 UTC; interval cycles count whole steps
from the budget's anchor, so a quiet entity does not walk its reset forward.
Changing a cycle moves every entity and user on the budget to the new boundary.
Spend reads as zero once a period ends, even before the next request rolls the
counters.

A request is held to every budget covering it: its API key, its workspace, the
member, the organization, and any provider or model entity matching where it is
sent. A model entity is checked before the provider's, and its refusal names the
model. A request that falls over to another model stays held to the budgets
resolved for the first one. A Playground request is held to its workspace's
budgets, as [Workspace-scoped spend](#workspace-scoped-spend) describes.

Deleting an API key removes it from every budget applied to it. Deleting an
organization budget removes it from every entity it applies to in the
organization. It is refused while a workspace's member default names the budget,
or while something outside the organization still holds it: another
organization's entity, or a gateway user assigned to it. What suspending a
member takes back is under [Invitations](#invitations).

A key with `exclude_from_budget`, or a deployment with
`budget_strategy: disabled`, bypasses enforcement.

Imported usage is retrospective and never counts toward a budget. Batch cost is
also settled after submission, so operators should not treat those paths as a
hard real-time cap. Batch settles dollars alone: its results arrive outside the
reservation that gated the submission, so a batch counts as the one request that
created it and contributes no tokens to a token cap, however many prompts it
carried. The same holds for the vision side-call a request makes to describe an
attachment. Cap batch-heavy workloads in dollars rather than in tokens. See
[Importing external usage](external-usage.md).

## Workspace-scoped spend

Usage is attributed to the workspace bound to the authenticating API key. A
Playground request, which runs on a dashboard session rather than a key, is
attributed to the workspace it runs in and held to that workspace's budgets.
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

Email and password sign-in is offered whenever any active identity holds a
password, not only once the operator has claimed the deployment. A member added
to the roster and signed up before that point signs in on the same screen, which
offers the master-key box beside the form while both credentials still work.

Sessions are revocable and expire after `dashboard_session_ttl_hours`. Password
changes, master-key rotation, sign-out, and identity deactivation revoke relevant
sessions.

A session authorizes the management API. It does not authorize
`/api/v1/chat/completions` or any other data-plane path, which take an API key
or the master key and nothing else: a keyless request resolves to the
deployment's default workspace, so honoring a cookie there would let any member
of any organization spend that workspace's provider credential.

The Playground is the one surface that runs a completion from a session, and it
is a separate endpoint rather than a relaxation of that rule.
`POST /api/v1/playground/chat/completions` resolves the caller's own attribution
user and proves their membership of the workspace it will bill before the
request reaches the pipeline, so the request is billed to the person who sent it
in a workspace that is theirs. No credential is minted for the browser and none
is held there. The usage row it writes carries no `api_key_id` and its own
endpoint label, which is what keeps in-product traffic separable from a
customer's integration.

### Passkeys

Passkeys are optional and additive to password sign-in. Set
`public_base_url` to establish the origin and relying-party ID. Use
`webauthn_rp_id` only when passkeys must be bound to a parent domain.

Where an edge serves the dashboard on a different host to the gateway, set
`webauthn_rp_id` to a domain that is a parent of both and list the dashboard
origin in `webauthn_allowed_origins`. The ID is not derived from `ui_base_url`,
so without this the ceremony fails in the browser and nothing is logged here
([#1134](https://github.com/mozilla-ai/otari/issues/1134)).

Changing the relying-party ID makes existing passkeys unusable. The dashboard
continues listing unusable credentials so the owner can remove them.

To turn passkeys off for a deployment, set `passkeys_enabled` to `false`
(`OTARI_PASSKEYS_ENABLED=false`). It defaults to `true`, so a self-hosted
gateway is unchanged. With it off, none of the passkey routes are mounted (they
answer 404, not 503), no `passkey` sign-in method is published, the dashboard
hides its passkeys page, and the `webauthn_*` settings are not checked at
startup, so values left over from before are inert. Setting it is the only way
to switch passkeys off while keeping `public_base_url`, which OAuth redirects
and mail links need. Passkeys already registered stay in the database, and work
again if the setting is turned back on.

### OAuth sign-in (Google and GitHub)

OAuth sign-in requires `public_base_url` and the provider's client ID and
secret. Register this redirect URI with the provider:

```text
{public_base_url}/auth/{provider}/callback
```

The gateway answers that path itself and redirects the browser into the
dashboard to finish. Where an edge serves the dashboard elsewhere, set
`ui_base_url` too; see [Configuration](configuration.md#the-interface-address).

OAuth signs in the account that holds the address the provider verified. If no account holds that address, the result depends on `open_signup`, as it does for [signup](#signup). When `open_signup` is disabled, the sign-in is refused. When it is enabled, Otari creates an account with its own organization and workspace, and sends no verification email, because the provider has already verified the address. A sign-in is always refused if the provider did not verify the address, or if the account for it is deactivated.

A provider sign-in on an address that is not yet verified marks it verified. It also removes any password and verification link set on that address before then: the provider confirms who owns the address, not who chose that password. The person can set a new password from their account page once signed in. A password on an address that was already verified is kept.

### Signup

Signup sets a password for an address and sends a verification link. What an
unknown address does depends on `open_signup`:

- `false` (default): signup only completes an identity an owner or admin already
  added or invited by address. An address nobody has added gets no account. This
  is the posture a single-tenant deployment wants, since anyone who can reach the
  dashboard can reach the form.
- `true`: an unknown address is registered, with an organization and workspace of
  its own. Use it where the deployment serves many tenants.

Signup never sets a password on an address that is already verified. That is the
state a Google or GitHub sign-in leaves, and the person who signs in that way
adds a password from Account settings while signed in.

Either way the response says the same thing whether the address was unknown,
already claimed, already verified, or genuinely just claimed, so its body
discloses nothing about the address. Response *timing* still does, because the
eligible path sends mail before it answers; that is [otari#720](https://github.com/mozilla-ai/otari/issues/720)
and it applies to both postures.

Signup needs mail configured, because an account that cannot verify its address
cannot sign in.

Open signup puts tenant creation on an unauthenticated route. The per-IP
throttle on the public auth routes is the only bound on it today, and nothing
expires the organization an unverified signup leaves behind, so run it behind
whatever edge controls the deployment has.

## Invitations

An owner or admin can invite a person to an organization and selected workspaces.
The dashboard always shows the accept link after an invite, so it can be shared
by hand. If mail is configured, Otari also emails it.

To invite several people at once, paste their addresses into the invite dialog,
separated by commas or new lines (up to 100). Everyone gets the same role and
workspaces. Each address is invited or refused on its own, so one that is already
a member does not stop the rest, and the result lists every address with whether
its email went out, or its accept link to share when it did not.

Opening the link lets the invitee accept. If the invited address has never signed
in, the accept page asks them to choose a first password, and once they accept
they can sign in straight away. No verification email is needed, so this works on
a deployment without mail. An address that can already sign in, by password or
through a provider, just accepts; the link cannot set or replace a password on
it.

Invitation tokens are bearer credentials. Whoever holds an unused link can join
as the invited address, and choose its first password if it has never signed
in, so send it only to that person.
Do not put tokens in logs or analytics. The browser validates and accepts them
through the public invitation endpoints. A link works once, and expires after
`invitation_expiry_hours`.

A signed-in person also sees the invitations addressed to them, and accepts or
declines one without a token: they are already authenticated as the addressee,
so the membership is addressed by id instead. Declining cancels the invitation
and suspends the paired membership, which is what stops the emailed link from
reviving it; a later invitation to the same address revives the membership.

Suspending a membership, by removing the member, setting their status to
suspended, revoking their invitation or declining it, takes back what it
granted: their workspace memberships in the organization are deleted, and so
is every budget applied to those memberships or to their organization
membership. Setting the status back to active, or inviting them again, does
not restore either; assign workspaces and budgets again. Their API keys belong
to the workspace, so they keep working and keep their budgets until the keys
are deleted.

## Email-domain auto-join

An owner or admin can claim an email domain so that colleagues join the
organization without being invited one at a time. It is the second way somebody
becomes a member, and unlike an invitation nobody decides about the individual,
so it is fenced accordingly.

A claim grants nothing on its own. Otari mints a verification token, the admin
publishes it as a TXT record at the claimed domain's apex, and only a claim whose
record Otari has found admits anyone:

```text
otari-domain-verification=<token>
```

Claiming is not exclusive; proving is. Several organizations may hold an unproven
claim on one domain and the first to publish the record takes it, which is what
stops a claim on a domain somebody else owns from locking out its real owner.

Once a domain is proven, anyone signing in with a **verified** address at that
domain joins as the role the claim names. The rules around that are deliberately
narrow:

- Only `member` and `viewer` can be handed out. Proving control of a domain is
  not a decision about a person, so it never confers organization management.
- An unverified address is skipped, so signing up on an address without reading
  the mail is not enough.
- An existing membership is left alone. A suspended one is not revived by
  signing in, and an established role is not overwritten.
- The person's active organization is never changed by joining.
- Public email providers cannot be claimed.

A proof expires. Domains change hands, so a claim stops admitting anyone once its
proof passes the deployment's TTL, and an admin renews it by verifying again from
the Email domains page. Nothing re-reads DNS between verifications, so that TTL is
the window in which a transferred domain still admits people.

## Related documentation

- [Admin dashboard](dashboard.md)
- [Configuration](configuration.md)
- [API reference](api-reference.md)
