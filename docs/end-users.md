# Track spend per end user

One application that calls Otari on behalf of many people can give each of them
spend, a budget and per-minute limits of their own, without sharing the master
key or minting a key per person. It does so with a **service key**: a key whose
requests name an end user. For how service keys fit with users, keys and
budgets, see [Access control](access-control.md#service-keys-and-end-users).

This guide uses a service called `mlpa` with two features, each carrying its
own per-user limit. You need the master key (or an operator session).

## 1. Create the end-user budgets

Give each budget an id of your own with `PUT /api/v1/budgets/{budget_id}`, which
creates the budget under that id or replaces it, so the same request can run at
every deploy:

```http
PUT /api/v1/budgets/end-user-budget-ai
{"max_budget": 0.1, "reset_alignment": "calendar_day", "rpm_limit": 40, "tpm_limit": 2000}
```

Create `end-user-budget-memories` the same way.

An id is up to 128 letters, digits, `.`, `_` and `-`, and starts with a letter
or digit. `POST /api/v1/budgets` generates an id instead, and
`GET /api/v1/budgets` reports how many users are on each budget in
`user_count`. Each entry a key may assign must be a deployment budget, not an
organization's.

## 2. Create the service key

Mark the key with `is_service_key` on `POST` or `PATCH /api/v1/keys`; only a
deployment operator can. List the budgets the key may start its end users on in
`end_user_budget_ids`, and set `end_user_budget_id` as the default for a request
that names none:

```http
POST /api/v1/keys
{"user_id": "mlpa", "is_service_key": true,
 "end_user_budget_ids": ["end-user-budget-ai", "end-user-budget-memories"],
 "end_user_budget_id": "end-user-budget-ai"}
```

- A key with a default and no list may assign the default alone.
- When both are set, the default must be on the list.
- Deleting a budget takes it off every list, but a budget that is some key's
  default cannot be deleted until that key's default changes, since the key's
  new end users would otherwise start uncapped.
- Give each service its own user: end users belong to the key's user, so every
  service key of one user reaches the same end users.

To pool every end user under one cap, give the key a budget of its own (a scoped
budget on the API key, its ceiling).

### Or declare it in config.yml

A service whose keys and budgets should not depend on a provisioning script can
declare both steps in `config.yml`:

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

`ceiling` is the key's own ceiling, which pools every end user. The service
reads the same `MLPA_SERVICE_KEY` and presents it as its API key, so nothing has
to hand a freshly minted secret from one to the other, and a restart or a
second replica does not change it. For the fields, the secret's format and what
a start writes back, see
[Budgets and API keys in config](configuration.md#budgets-and-api-keys-in-config).

## 3. Name the end user on each request

Send the end user's id the way any client names a user: the `user` field on
`/v1/chat/completions`, `/v1/responses` and `/v1/search`, and
`metadata.user_id` on `/v1/messages`. To start a new end user on a budget other
than the key's default, name it in the `Otari-End-User-Budget` header:

```http
POST /v1/chat/completions
Authorization: Bearer <service key>
Otari-End-User-Budget: end-user-budget-memories

{"model": "...", "user": "fxa123:memories", "messages": [...]}
```

- Send a stable id per end user, not a per-session or per-request value: each
  distinct id creates an end user, whether or not the request is then admitted,
  and nothing else caps how many a key can create. Set a
  [rate limit](configuration.md#rate-limit-rules) on a deployment that issues
  service keys.
- An id is at most 256 characters and may otherwise be any string, `/`
  included, such as a base64 key id.
- The header applies only when the request creates the end user. An existing
  end user keeps its budget, whatever a later request names, so a request
  cannot undo a move an operator made.
- A budget not on the key's list is refused with `403` and the code
  `end_user_budget_not_allowed`, even for an end user that already exists.
- The response carries `Otari-End-User-Budget` with the budget the end user is
  on, so the caller can see when it differs from the one it named.

A request that names nobody, or the key's own user, bills the key's user as it
would on any other key.

## 4. Manage end users

The service manages its own end users with its service key, addressing each one
by the id it names it by. It needs neither the master key nor its key's id. For
example, to move an end user to another budget:

```http
PATCH /api/v1/end-users?external_id=fxa123:memories
Authorization: Bearer <service key>

{"budget_id": "end-user-budget-ai"}
```

URL-encode the id as any query value is (`/` as `%2F`, `+` as `%2B`). To clean
up short-lived end users, such as one id per guest session, delete them by age
with `DELETE /api/v1/end-users?created_before=2026-10-10T00:00:00Z`, repeating
until the answer's `deleted` is 0. The routes are listed under
[End-user routes](#end-user-routes).

## How end users are billed and capped

Otari, on a request that names an end user on a service key:

- Bills the request to that end user, creating it on first use. An end user is
  a user record owned by the key's user, so a key can only bill end users of
  its own user: two services that both name `alice` get two separate end users,
  and naming another key's user creates an end user of your own rather than
  reaching theirs.
- Caps each end user at a budget copied onto it when it is created: the one the
  request names in `Otari-End-User-Budget`, or else the key's
  `end_user_budget_id`. Each end user gets the full limit and its own reset
  period. Changing the key's setting affects end users created afterwards.
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

Blocking the key's user stops its end users too.

Moving an end user (to a budget on the key's list) restarts its period on the
new budget and applies that budget's per-minute limits, but keeps its spend,
tokens and requests so far, as moving a user through `/api/v1/users` does: an
end user that used up one budget can be over the next one's limits until that
period resets.

Deleting an end user erases its telemetry and agent traces, as deleting any
user does, and stops it being listed or counted. Its row stays, with what it
has spent and used: a request that names the same id again revives it with
those counters, on the budget it was on, so deleting an end user cannot reset
its budget. Its usage history stays too. When erasure fails the call answers
5xx and deletes nothing, so it can be repeated.

### Capping a key's end users

Nothing caps how many end users a service key creates unless the key sets
`max_end_users` (in `POST` or `PATCH /api/v1/keys`, or in config.yml). Set a
rate limit on a deployment that issues service keys, and send a stable id per
end user rather than a per-session or per-request value.

With `max_end_users` set, a request that would create one more end user, or
revive one that was deleted, is refused with 403 and the code
`end_user_limit_reached`, and nothing is created; the end users the key already
has keep working. Deleting end users makes room again. `null`, the default, is
no cap, and `0` admits no new end user at all.

The cap counts the end users of the key's user that are on a budget the key may
assign (its `end_user_budget_ids`, or its default alone), or every end user of
that user when the key assigns none. A key with a list and no default also
counts the end users on no budget, which is where a request naming none starts
one. So two service keys of one user, each with budgets of its own, such as one
for signed-in users and one for guests, keep separate caps. For a key with
budgets of its own, an end user moved to a budget outside its list stops
counting toward it, and moving one onto the list from outside takes a place, so
that move is refused with `end_user_limit_reached` when the key is full. A key
that assigns no budget counts every end user wherever it is moved.

Replicas admitting end users at once cannot exceed the cap: each new or revived
end user is counted and created while the key's user is locked. Requests from
live end users never take that lock.

## End-user routes

| Request | Does |
|---|---|
| `GET /api/v1/end-users` | Lists the end users, oldest first |
| `GET /api/v1/end-users?external_id=...` | Reads the end user: its budget, counters and whether it is blocked, as a list of one, or none when no end user goes by the id |
| `PUT /api/v1/end-users?external_id=...` `{"budget_id": ...}` | Puts the end user on a budget, creating it first if it has not made a request yet (201) |
| `PATCH /api/v1/end-users?external_id=...` `{"blocked": true}` or `{"budget_id": ...}` | Blocks, unblocks or moves the end user |
| `DELETE /api/v1/end-users?external_id=...` | Deletes the end user, answering `{"deleted": 1}`, or 404 |
| `DELETE /api/v1/end-users?created_before=...` | Deletes up to `limit` (default 100, at most 1000) of the oldest end users created before that time, answering `{"deleted": n}`; repeat until `n` is 0 |
| `GET /api/v1/end-user-budgets` | Lists the budgets on the key's list, with how many end users are on each |

- **Authentication.** These routes take only a service key: another API key is
  refused with 403, and the master key and a dashboard session do not
  authenticate them. A service key whose own user is blocked or deleted is
  refused with 403 as well, since its end users cannot spend either.
- **Ownership.** A key of another user reaches none of a user's end users: an id
  it does not own answers 404, as one that does not exist does.
- **Budgets.** A budget set through these routes must be on the key's list; any
  other is refused with 403 and `end_user_budget_not_allowed`.
- **Listing.** `GET /api/v1/end-users` takes `skip` and `limit` (up to 1000),
  narrows on `budget_id`, `blocked` and `external_id`, and with
  `include_total=true` counts every match in the `Otari-Total-Count` header. Its
  list and the counts on `/api/v1/end-user-budgets` leave out deleted end users,
  and the counts include blocked ones, which `blocked_end_user_count` also gives
  on its own.
- **Path form.** The id can also go in the path,
  `/api/v1/end-users/{external_id}`, for `GET`, `PUT`, `PATCH` and `DELETE`,
  which answer as their query forms do (the `GET` with the end user itself, or
  404, and the `DELETE` with 204, or 404). Use it only for an id without `/`: a
  proxy can merge the `//` of an id like `ab//cd`, and an HTTP client collapses
  `.` and `..` segments, so in a path such an id can reach another end user. An
  id with a `.` or `..` segment is refused there with 400. The query parameter
  carries any id intact.

### Operator routes

A deployment operator reaches any service key's end users through the same
routes under the key, with the master key: `GET`, `PUT` and
`PATCH /api/v1/keys/{key_id}/end-users?external_id=...`, with `GET` taking the
list's other parameters too, and the path form
`/api/v1/keys/{key_id}/end-users/{external_id}`. These still reach the end users
of a key whose user is blocked or deleted. A budget set there must be on that
key's list too.

The users API lists end users across keys, where each one carries
`parent_user_id` (the key's user) and `external_id`:
`GET /api/v1/users?parent_user_id=...&external_id=...` filters on them, and it
can put an end user on any deployment budget, not only one on a key's list.

## Where end users apply

End users are supported on `/v1/chat/completions`, `/v1/responses`,
`/v1/search` and `/v1/messages`. The other endpoints (embeddings, files,
batches and the other pass-through routes) treat a service key as an ordinary
key, so a `user` naming someone else is handled by the key's
[`reject_user_mismatch`](access-control.md#mismatched-user-fields) setting
there.
