# API keys

An API key is the credential an application sends to Otari. It belongs to one
workspace and one user, so it decides where a request's usage and spend land.
How to send it, and the base URL to send it to, is on
[Connect to Otari](../get-started/connect.md).

To create one in the dashboard, open the workspace's **API Keys** tab and click
**Create key**. Give it a name and, optionally, an expiry date. A key inherits
your model access unless you limit it to specific models under **Advanced**.
The key is shown once, so copy it then.

The generated [OpenAPI specification](../public/openapi.json) is the source of
truth for the key endpoints' schemas.

## Key fields and rules

API keys are workload credentials. Each key has a fixed user and workspace and
may also define:

- expiration and active status
- allowed models
- budget exemption
- a budget of its own: a spend ceiling on the key, which its owner's budget does
  not replace (both apply)
- whether mismatched client `user` fields are accepted (see
  [Mismatched `user` fields](#mismatched-user-fields))
- whether content-free agent telemetry is captured
- whether it is a service key, which may name end users
- application metadata

The plaintext key is returned only when it is created or rotated. Store it then.
Rotation preserves the key record and invalidates the previous secret.
A key can also be declared in config.yml with a secret you choose, in which
case config.yml owns it and writes it back whenever its entry changes; see
[Budgets and API keys in config](../reference/configuration.md#budgets-and-api-keys-in-config).
Rotating a declared key through the API lasts until config.yml changes the
key's entry or its secret, which restores the secret config.yml declares.

A budget-exempt key is also exempt from `require_pricing`. Reserve such keys for
usage import or other intentional observability-only traffic.

A backend on Google Cloud, or behind another OpenID Connect issuer, can
authenticate as a key with its platform identity token instead of the key's
secret, once the operator trusts the issuer and gives the key a federation
rule; see [Workload identity](../self-hosting/workload-identity.md).

`/api/v1/keys` manages every key in the caller's organization and requires the
deployment operator's standing. A signed-in member without it manages their own
keys at `/api/v1/organizations/me/keys`, which derives the owner rather than
accepting one, mints only into a workspace the caller may see, and never issues
a budget-exempt key.

### Mismatched `user` fields

A request can name a user: `user` on Chat Completions, Responses and the
pass-through routes, `metadata.user_id` on Messages. On an ordinary key that
value may only be the key's own user; anything else is refused with `403`. A
key's `reject_user_mismatch` changes that for the key alone:

| Value | Dashboard choice | A request naming another user |
| --- | --- | --- |
| `null` (default) | Use the deployment setting | Follows the deployment-wide `reject_user_mismatch`, which rejects unless changed |
| `true` | Always reject (403) | Rejected, even where the deployment accepts |
| `false` | Always accept | Accepted, and billed to the key's user |

Spend binds to the key's own user whichever value applies, so accepting a
mismatch never lets a key charge someone else.

Accept mismatches on a key whose client puts something other than an Otari
user ID in that field. Claude Code sends a telemetry value there (see
[Claude Code](../integrations/claude-code.md)). So does an application that
sends every request on one key and passes its own end user's ID. That
application has two choices:

- Accept mismatches, and every request bills the key's user: one budget for
  the whole application.
- Make the key a [service key](#service-keys-and-end-users), and each ID
  becomes an end user with spend and a budget of its own.

Set the value in the dashboard's key form (under **Advanced** when creating a
key, and on the edit form), as `reject_user_mismatch` on `POST` or `PATCH`
`/api/v1/keys` and `/api/v1/organizations/me/keys`, or on a key declared in
config.yml. The deployment-wide default is a runtime setting, changed on the
Settings page or with `PATCH /api/v1/settings`.

The check is a standalone one. A hybrid gateway, such as otari.ai's, does not
compare the field with the key's owner, so a key there needs no setting: the
value only feeds the
[end-user identifier](../reference/hybrid-mode-protocol.md#end-user-identifier) sent to the
provider.

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
other, and a restart or a second replica does not change it. A start writes
the declared settings back over changes made through the API only when the
key's entry in config.yml changed; see
[What config.yml owns](../reference/configuration.md#what-configyml-owns).

### Managing end users

A service manages its own end users with its service key, addressing each one
by the id it names it by. It needs neither the master key nor its key's id:

| Request | Does |
|---|---|
| `GET /api/v1/end-users` | Lists the end users, oldest first |
| `GET /api/v1/end-users?external_id=...` | Reads the end user: its budget, counters and whether it is blocked, as a list of one, or none when no end user goes by the id |
| `PUT /api/v1/end-users?external_id=...` `{"budget_id": ...}` | Puts the end user on a budget, creating it first if it has not made a request yet (201) |
| `PATCH /api/v1/end-users?external_id=...` `{"blocked": true}` or `{"budget_id": ...}` | Blocks, unblocks or moves the end user |
| `DELETE /api/v1/end-users?external_id=...` | Deletes the end user, answering `{"deleted": 1}`, or 404 |
| `DELETE /api/v1/end-users?created_before=...` | Deletes up to `limit` (default 100, at most 1000) of the oldest end users created before that time, answering `{"deleted": n}`; repeat until `n` is 0 |
| `GET /api/v1/end-user-budgets` | Lists the budgets on the key's list, with how many end users are on each |

```http
PATCH /api/v1/end-users?external_id=fxa123:memories
Authorization: Bearer <service key>

{"budget_id": "end-user-budget-ai"}
```

The query parameter names any end user, so use it for an id that may contain
`/`, and URL-encode the id as any query value is (`/` as `%2F`, `+` as `%2B`).
The id can also go in the path, `/api/v1/end-users/{external_id}`, for `GET`,
`PUT`, `PATCH` and `DELETE`, which answer as their query forms do (the `GET`
with the end user itself, or 404, and the `DELETE` with 204, or 404). That form suits an id without `/`: a
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

Deleting an end user erases its telemetry and agent traces, as deleting any
user does, and stops it being listed or counted. Its row stays, with what it
has spent and used: a request that names the same id again revives it with
those counters, on the budget it was on, so deleting an end user cannot reset
its budget. Its usage history stays too. A service whose end users are
short-lived, such as one id per guest session, can delete the expired ones by
age with `created_before` (`2026-10-10T00:00:00Z`), a page per call. When
erasure fails the call answers 5xx and deletes nothing, so it can be repeated.

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
