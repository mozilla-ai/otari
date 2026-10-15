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

An operator manages every account on the deployment from the dashboard's
Accounts page (`/api/v1/admin/users`): deactivate or reactivate one, grant or
remove operator access, and send a password link. Deactivating an account ends
its sessions and voids its pending verification and reset links, so
reactivating it does not bring an old link back.

A password link is how an operator helps somebody who cannot sign in, and the
only way back for a forgotten password on a deployment that sends no mail.
Where mail is configured, the link is emailed to the account's address, as
"Forgot your password?" does. Otherwise, or when the send fails, the operator
is given the link to pass on. It works once, expires after
`password_reset_expiry_hours`, and replaces any earlier reset link. Because the
mailbox never saw a link that was passed on, a password chosen through one
records the address as vouched for rather than proven, the same as a password
chosen through an [invitation](#invitations) link. A link is refused for a
deactivated account, one with no address, one that has no password and signs in
with a provider or a passkey (its owner adds a password from Account settings
once signed in), and, where `open_signup` is disabled, an unverified address no
organization still has as a member or invitee. Where `open_signup` is enabled
such an address gets a link, and choosing a password through it gives the
account an organization and workspace of its own, as [signup](#signup) does.

`PUT /api/v1/auth/password` with the master key acts on the operator's own
account, never another one: an address sent with it is the operator's sign-in
address, and on an unclaimed deployment it claims the deployment. It refuses an
address another account holds, and that refusal points to the password link.

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
A key can also be declared in config.yml with a secret you choose, in which
case config.yml owns it and writes it back at every start; see
[Budgets and API keys in config](configuration.md#budgets-and-api-keys-in-config).
Rotating a declared key through the API lasts only until the next start, which
restores the secret config.yml declares.

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
characters and may otherwise be any string, `/` included, such as a base64 key
id. The routes below name every end user with the `external_id` query
parameter, which carries any id intact.

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

### Declaring a service key in config

A service whose keys and budgets should not depend on a provisioning script can
declare them in config.yml instead. The two requests above become:

```yaml
budgets:
  end-user-budget-ai: {max_budget: 0.1, reset_alignment: calendar_day, rpm_limit: 40, tpm_limit: 2000}
  end-user-budget-memories: {max_budget: 0.05, reset_alignment: calendar_day}
  mlpa-global: {max_budget: 100, reset_alignment: calendar_day}

api_keys:
  mlpa:
    secret: ${MLPA_SERVICE_KEY}
    user_id: mlpa
    is_service_key: true
    end_user_budget_ids: [end-user-budget-ai, end-user-budget-memories]
    end_user_budget_id: end-user-budget-ai
    ceiling: mlpa-global
```

`ceiling` is the key's own ceiling, the scoped budget on the API key that pools
every end user. The service reads the same `MLPA_SERVICE_KEY` and presents it as
its API key, so nothing has to hand a freshly minted secret from one to the
other, and a restart or a second replica does not change it. Every start writes
the declared settings back over changes made through the API; see
[What config.yml owns](configuration.md#what-configyml-owns).

### Managing end users

A service manages its own end users with its service key, addressing each one
by the id it names it by. It needs neither the master key nor its key's id:

| Request | Does |
|---|---|
| `GET /api/v1/end-users` | Lists the end users, oldest first |
| `GET /api/v1/end-users?external_id=...` | Reads the end user: its budget, counters and whether it is blocked, as a list of one, or none when no end user goes by the id |
| `PUT /api/v1/end-users?external_id=...` `{"budget_id": ...}` | Puts the end user on a budget, creating it first if it has not made a request yet (201) |
| `PATCH /api/v1/end-users?external_id=...` `{"blocked": true}` or `{"budget_id": ...}` | Blocks, unblocks or moves the end user |
| `GET /api/v1/end-user-budgets` | Lists the budgets on the key's list, with how many end users are on each |

```http
PATCH /api/v1/end-users?external_id=fxa123:memories
Authorization: Bearer <service key>

{"budget_id": "end-user-budget-ai"}
```

The query parameter names any end user, so use it for an id that may contain
`/`, and URL-encode the id as any query value is (`/` as `%2F`, `+` as `%2B`).
The id can also go in the path, `/api/v1/end-users/{external_id}`, for `GET`,
`PUT` and `PATCH`, which answer as their query forms do (the `GET` with the
end user itself, or 404). That form suits an id without `/`: a
proxy can merge the `//` of an id like `ab//cd`, and an HTTP client collapses
`.` and `..` segments, so in a path such an id can reach another end user. An
id with a `.` or `..` segment is refused there with 400.

These routes take only a service key: another API key is refused with 403,
and the master key and a dashboard session do not authenticate them. A service
key whose own user is blocked or deleted is refused with 403 as well, since its
end users cannot spend either; the operator's `/api/v1/keys/{key_id}/end-users`
routes still reach them.
`GET /api/v1/end-users` takes `skip` and `limit` (up to 1000), narrows on
`budget_id`, `blocked` and `external_id`, and with `include_total=true` counts
every match in the `Otari-Total-Count` header. Its list and the counts on
`/api/v1/end-user-budgets` leave out deleted end users, and the counts include
blocked ones, which `blocked_end_user_count` also gives on its own.

A budget set this way must be on the key's list: any other is refused with 403
and `end_user_budget_not_allowed`, so a key cannot put an end user on a budget
it was not given. Moving an end user restarts its period on the new budget and
applies that budget's per-minute limits, but keeps its spend, tokens and
requests so far, as moving a user through `/api/v1/users` does: an end user
that used up one budget can be over the next one's limits until that period
resets.

End users belong to the key's user, so every service key of one user reaches
the same end users, and a key of another user reaches none of them: an id it
does not own answers 404, as one that does not exist does. Give each service
its own user to keep their end users apart.

A deployment operator reaches any service key's end users through the same
routes under the key, with the master key:
`GET`, `PUT` and `PATCH /api/v1/keys/{key_id}/end-users?external_id=...`, with
`GET` taking the list's other parameters too, and the path form
`/api/v1/keys/{key_id}/end-users/{external_id}`. A budget set there must be on
that key's list too. The users API lists end users across
keys, where each one carries `parent_user_id` (the key's user) and
`external_id`: `GET /api/v1/users?parent_user_id=...&external_id=...` filters on
them, and it can put an end user on any deployment budget, not only one on a
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
- A scoped budget limits an organization, workspace, membership, or API key and
  can optionally narrow the limit to one provider.

A routing fallback to another provider must also pass that provider's budgets, as [Routing](routing.md#what-is-billed-and-what-the-caller-sees) describes.

A budget caps up to three things over its period, each set independently and
each unlimited when left unset: spend in USD (`max_budget`), total tokens
(`token_limit`), and requests (`request_limit`). A request must have room on
every axis the budget caps. Spend and tokens are held at an upper bound before
dispatch and reconciled to the measured figures afterwards; only the unused part
of an over-estimate is released, and what the request measurably used stays
charged. A request counts as one request when it is admitted. A model priced at
zero spends no dollars, and still spends tokens and one request.
A request whose dollar hold is larger than what remains is refused even when
nothing has been spent yet, and the refusal says so with both amounts:
`User 'alice' has $0.0015 of budget remaining, less than this request's
estimated cost of $0.01025`. On chat completions, messages and responses the
estimate prices the request's output token limit (`max_tokens`, or
`max_output_tokens` on responses; `budget_estimate_default_output_tokens` when it
sets none), and where the model charges for output the refusal names that
figure, so a lower limit may let the request fit. A
budget already at its cap is refused as `has exceeded budget limit`.
Endpoints that hold no token estimate (embeddings, rerank, and the other
pass-through routes) are refused once a token cap is exhausted rather than
reserving headroom for themselves, so a token cap can be passed by the requests
already in flight when it runs out.

A budget can also limit each user on it per minute: `rpm_limit` requests and
`tpm_limit` tokens. These are counted in `rate_limit_store`, so with Redis they
hold across replicas, and tokens are counted on what each request used, as
LiteLLM counts them: a request is admitted while the user's minute is under the
limit. They apply to a user's own budget on chat completions, messages,
responses and search, not to a scoped ceiling, and a refusal is a 429 naming the
rule `budget`.

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

### Reservations are taken up front

The hold is the request's worst case, taken before dispatch: its prompt at
roughly four characters per token, plus its whole output token limit. A request
with `max_tokens: 8192` therefore holds the price of 8192 output tokens however
few the model ends up writing, and a user close to the cap is refused once that
hold no longer fits, even when the answer would have cost a fraction of it.
The trade is deliberate: a request is admitted only when its estimated worst
case fits, at the price of refusing some requests near the cap that would have
fit. The prompt side is an estimate, and settlement records what the request
actually cost, so a request can still settle a little past its hold (and its
budget past the cap) when the prompt tokenizes heavier than the estimate
assumes. A client that sends a realistic output limit, rather than the model's
maximum, keeps more of its allowance usable near the cap.

Holds are also what bound concurrent requests from one user or end user. Each
admitted request keeps its hold until it settles, and a new request is admitted
only while recorded spend, every hold still in flight and its own hold together
fit under the cap, checked in one conditional update so two requests cannot both
take the last of it. Ten parallel requests near the cap are therefore admitted
only as far as their combined worst case fits, and the rest are refused. A hold a crashed request leaves behind is returned after
`budget_reservation_ttl_sec` (15 minutes by default). Scoped ceilings hold the
same way, so a key's ceiling bounds all of its end users' in-flight requests
together.

### Refusals and `Retry-After`

A refusal is a 403 with `Otari-Error-Code: budget_exceeded`, and
`Otari-Budget-Scope` names the budget that refused: `user`, or the ceiling's
scope. When that budget resets, the refusal also carries `Retry-After`, the
whole seconds (rounded up) until its current window ends: the next UTC
midnight, ISO week or month for a calendar budget, and the end of the user's or
ceiling's own period for a rolling one, which only the gateway knows. Where
several budgets apply, it is the window of the one that refused. A budget with
no period never resets and sends no `Retry-After`, and neither does a refusal of
a hold larger than the whole cap, since a fresh window would refuse it too.

## Workspace-scoped spend

Usage is attributed to the workspace bound to the authenticating API key.
Organization and workspace usage views then apply the signed-in identity's
membership. Deployment operators can read the operator usage API, which covers
every workspace of the operator's active organization (the default organization
for the master key).

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
sessions. A request whose session cookie no longer authenticates is refused
with 401 "Your dashboard session has ended. Sign in again."

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

Otari links each provider account to one Otari account by the provider's own ID for that person, not by the address. An Otari account holds at most one link for each provider, so it can link both a Google and a GitHub account. A sign-in through a link still works after the address changes at the provider. A second account at the same provider is refused, even with the same address. Otari has no way to remove a link yet, so a person whose provider account was replaced signs in with a password or a passkey instead.

The first sign-in with a provider account links it to the account that holds the address the provider verified. Otari links by address only for Google and GitHub, because no tenant can set what they say about an address. If no account holds that address, the result depends on `open_signup`, as it does for [signup](#signup). When `open_signup` is disabled, the sign-in is refused, and so is one on an account an owner or admin added or invited whose address nobody has verified yet, unless it is still an active member or has an invitation it can accept, the same rule [signup](#signup) follows. When it is enabled, Otari creates an account with its own organization and workspace, and sends no verification email, because the provider has already verified the address. The first sign-in is refused if the provider did not verify the address. Any sign-in is refused if the provider says the address is not verified, or if the account is deactivated. A provider that shares no address at all is refused with a message saying so. A provider that fails, cannot be reached, or refuses this deployment's client credentials while finishing the sign-in answers 502, so a provider outage or a misconfigured client is not counted as a client error; only a code the provider calls invalid, expired or spent answers 400.

The first sign-in on an address that is not yet verified, or that an organization only vouched for (see [Invitations](#invitations)), marks it verified. It also removes everything set up on that address before then: its password, any verification or reset link, every session signed in with that password, and every passkey registered from one. On an unverified address it replaces the display name chosen at signup with the provider's, and renames the organization that signup created after it, unless someone else belongs to that organization. The provider confirms who owns the address, not who chose that password or that name. The callback's response says so (`password_removed`), and the dashboard tells the person before they go on, so the password does not just stop working. They can set a new password from their account page once signed in. A password on an address that was already verified is kept, with its sessions and passkeys. A later sign-in through the link does not change the account's password or verification, unless the address has since become only vouched for (an operator's [password link](#deployment-wide-account-administration) that was passed on rather than mailed does that): the sign-in then proves the address the same way, and removes the password, links, sessions and passkeys set up through the vouch.

Where the deployment publishes terms (`terms_url`), a sign-in that creates an account, or that is the first way in to one an owner or admin added or invited, needs the person's acceptance of them, as the signup form and a first password from an invitation link do. The dashboard asks before it sends the person to the provider and returns the acceptance with the callback (`terms_accepted`). Without it the sign-in is refused with the code `terms_not_accepted`, and nothing it would have set up is kept: no account, organization, workspace, provider link or session. With it, the acceptance is recorded on the account. A sign-in to an account someone already claimed, or through a provider account already linked to one, does not ask, and an account that never accepted the terms is not asked to later.

A sign-in that creates an account, or that is the first way in to one an owner or admin added, is reported as a signup to whatever growth integration the deployment runs, as the signup form, a first password set from an invitation link, and a first password set by reset are.

### Signup

Signup sets a password for an address and sends a verification link. The link
verifies the address only together with that password: the verification page
asks for it, and a wrong one is refused without spending the link, with a
message saying the link still works. A link that is unknown, expired or spent
gets a different message, which says nothing about the password. Wrong
passwords are counted per client address, so a held link cannot be used to guess
the password at speed; a right one is never held back. Anyone can
sign up with an address they do not hold, and the link then reaches the
address's real owner, whose click must not activate a password a stranger
chose. Signing up again before verifying changes nothing, so the first password
is the one the link takes. An owner who never signed up, or who forgot that
password, uses "Forgot your password?" instead, which replaces it. The
verification and reset pages check their link as they open, without spending
it, and say at once when it has expired or was already used (or a newer link
replaced it). Where the deployment publishes terms (`terms_url`),
signup is refused unless they are accepted, with the form or with a provider (see
[OAuth sign-in](#oauth-sign-in-google-and-github)). What an unknown address does depends
on `open_signup`:

- `false` (default): signup only completes an identity an owner or admin already
  added or invited by address, and only while it is still wanted: an active
  member, or an invitee whose invitation is still pending. An address nobody has
  added gets no account, and neither does one whose invitation was revoked or
  expired or whose membership was removed. This is the posture a single-tenant
  deployment wants, since anyone who can reach the dashboard can reach the form.
  The rule holds when a link is used as well as when it is mailed, so revoking
  the invitation or removing the member also stops a verification link, a resent
  one, and a reset link minted before then. For the same reason, creating an
  organization requires an active membership in one already, so a member who was
  removed cannot make themselves a tenant.
- `true`: an unknown address is registered, with an organization and workspace of
  its own. So is an address an organization once added or invited but no longer
  wants (its invitation was revoked, declined or expired, or the member was
  removed) when it is claimed, that is, when its owner first proves it: by
  signing up with the form (or verifying that signup), by a first sign-in with a
  provider, or by choosing a password through "Forgot your password?" while the
  address is unverified or only vouched for (see [Invitations](#invitations)).
  It gets its own organization and workspace, and signs in to them. An address
  is claimed once: someone who already proved their address and later declines
  their last invitation or is removed keeps the account but has no
  organization, and signs in to the empty state [Invitations](#invitations)
  describes, where they can create one. An address with an invitation still
  waiting is claimed as it is, and finds the invitation under Invitations. Use
  it where the deployment serves many tenants.

Signup never sets a password on an address that is already verified. That is the
state a Google or GitHub sign-in leaves, and the person who signs in that way
adds a password from Account settings while signed in. Such a signup mails the
address a notice that it already has an account and how it signs in: with its
password (and "Forgot your password?" if it is forgotten), or with its provider.

"Forgot your password?" mails a link to set or reset the password of an
address that has a password, and also of one an owner or admin added or invited
that nobody has signed in to yet, so that person can choose a first password.
Completing it verifies the address, since the link reached its mailbox, and where
the address was unverified or only vouched for it also ends every session and
removes every passkey set up before then. Where it was unverified, the display
name chosen at signup is dropped too, and the organization that signup created is
renamed after the address, unless someone else belongs to it. Where `open_signup`
is disabled, every address that is not yet verified, with or without a password,
follows the signup rule above: it gets a link, and the link completes, only while it is an active
member or holds an invitation it can still accept, so a revoked invitee gets
none. Where the deployment publishes terms and the account has accepted none,
the reset page asks for them. An address that signs in only through a provider
gets no link; it is mailed the same notice signup sends, saying which provider
it signs in with and that a password can be added from Account settings.

Either way the response says the same thing whether the address was unknown,
already claimed, already verified, or genuinely just claimed, so its body
discloses nothing about the address. A notice goes only to the address's own
mailbox, so it tells nobody else anything either. Response *timing* still does, because the
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
If mail is configured, Otari emails the accept link. When the email does not go
out (no mail configured, the send failed, or a mail limit below was reached), the
dashboard shows the link instead, so it can be shared by hand. A link that was
emailed is never shown to the inviter, so only the invitee's mailbox holds it. The
email names the inviter by their address as well as the name they chose, and an
address the mail server would refuse, such as one ending in a dot, is refused
when it is invited.

Invitation email is limited, because it carries text an organization chose (its
name, the inviter's name) from the deployment's own sender, and on a deployment
where anyone can create an organization that would otherwise make it an open
relay. In any 24 hours an organization sends at most
`invitation_mail_limit_per_organization` invitation emails (250 by default), and
an address receives at most `invitation_mail_limit_per_address` (5 by default)
from every organization together, so revoking and inviting again cannot flood it.
An invitation over a limit is still created, and its link is returned to share by
hand, as when mail is off: the link costs the deployment nothing, and refusing
would stop a legitimate admin partway through a bulk invite. The counts are kept
in `rate_limit_store`, so with Redis they hold across replicas.

A link that was lost can be reissued. Reissuing gives a pending or expired
invitation a new link, emailed or returned as above, and the old link stops
working. Inviting an address whose invitation is still pending is refused and says
to reissue it instead. On Members, an invited row says when its invitation has
expired, and offers Reissue and Revoke either way.

The API can also add a member directly (`POST /api/v1/organizations/me/members`),
active at once with no invitation to accept and no email. A new address added
that way gets in by signing up, by "Forgot your password?" or through a provider,
so on a deployment with no mail and no provider it waits for an operator's
password link. Invite it instead, since the invitation link itself can set its
first password.

To invite several people at once, paste their addresses into the invite dialog,
separated by commas or new lines (up to 100). Everyone gets the same role and
workspaces. Each address is invited or refused on its own, so one that is already
a member does not stop the rest, and the result lists every address with whether
its email went out, or its accept link to share when it did not.

Opening the link lets the invitee accept. If the invited address has never signed
in, the accept page asks them to choose a first password, and once they accept
they can sign in straight away. No verification email is needed, so this works on
a deployment without mail. Where the deployment publishes terms, choosing that
password requires accepting them, as signup does. Accepting without a password
records no acceptance, so the provider sign-in that follows it asks for the terms. An address that can already sign in, by password or
through a provider, just accepts; the link cannot set or replace a password on
it. Neither can it for an address another organization has already added: the
inviter holds the link too, so that person accepts without a password and then
signs in through a provider or resets their password, either of which proves the
address.

An invitee does not need the link. Signing up with the invited address, or
signing in with Google or GitHub, completes the account the invitation created,
and every invitation still waiting on it is listed under Invitations in the
dashboard, where it can be accepted or declined. Someone who is not yet a member
of any organization is taken there straight after signing in.

A password chosen through a link counts as verified so the person can sign in,
but Otari records that an organization vouched for the address rather than that
the person proved it. A vouched address does not trigger
[email-domain auto-join](#email-domain-auto-join), and another organization
cannot add or invite it, since the vouching organization's admins could have
chosen its password. A later provider sign-in on it replaces the password with
the provider, and resetting the password by email also proves the address;
either lifts both restrictions.

A link stops working as soon as its membership stops waiting on it: when the
person is added directly, removed, or accepts or declines from their inbox, any
invitation still pending for them is closed. Revoking an invitation suspends
the membership only while it is still waiting; a person who became a member
another way keeps their access. An expired link says it expired every time it is
opened, and every refusal says to ask the inviter for a new link. An address
whose account a deployment operator deactivated cannot be invited or added until
it is reactivated, and an invitation already pending for it stops working until
then too; neither refusal gives the reason, since the admin asking may be
another tenant's. For the same reason, refusing an address another organization
vouched for says what would lift the refusal and nothing about that
organization. An organization name cannot carry line breaks or other control
characters, since it is quoted in invitation emails.

When a link may not set a password (the address belongs to someone another
organization still has as a member or invitee), the accept page says so before
the invitee accepts, and names the way in this deployment offers once they have:
a provider sign-in where one is configured, or "Forgot your password?" where mail
is, or else asking whoever administers the gateway. A password sent anyway is
refused with the same advice. Memberships that were revoked, declined, expired or
removed do not count.

Signing in answers which organization the session acts in. If the person was
removed from the one they last used, it is the oldest one they still belong to,
and if they belong to none it is empty; the dashboard then shows their
invitations, and offers to create an organization where `open_signup` is
enabled. Being removed, declining or having an invitation revoked moves the
person's active organization to the oldest one they still belong to. For
someone who belongs to none, `GET /api/v1/organizations/me` answers 404 saying
so, without naming an organization.

Invitation tokens are bearer credentials. Whoever holds an unused link can join
as the invited address, and choose its first password where the link allows one,
so send it only to that person.
Do not put tokens in logs or analytics. The browser validates and accepts them
through the public invitation endpoints. A link works once, and expires after
`invitation_expiry_hours`.

A signed-in person also sees the invitations addressed to them, and accepts or
declines one without a token: they are already authenticated as the addressee,
so the membership is addressed by id instead. Declining cancels the invitation
and suspends the paired membership, which is what stops the emailed link from
reviving it; a later invitation to the same address revives the membership.

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
  the mail is not enough. So is one only vouched for through an invitation
  link, since any organization can invite any address.
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
