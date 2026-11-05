# Track spend per end user

One application that calls Otari on behalf of many people can give each of them
spend and a budget of their own, without minting a key per person. It does so
with a **service key**: a key whose requests name an end user. How end users
relate to the key's user, its budgets and its limits is in
[How end users are billed and capped](#how-end-users-are-billed-and-capped).

This guide uses a service called `mlpa` with two features, each carrying its
own per-user limit. You need to be an owner or admin of the organization; a
member is refused with 403. On a deployment you run, the operator can also make
service keys on the deployment's own budgets, or declare them in config.yml:
see [Service keys on a deployment you run](operator-service-keys.md).

## 1. Create the end-user budgets

Create one budget per feature in **Spend & budgets**, or with
`POST /api/v1/organizations/me/budgets`:

```http
POST /api/v1/organizations/me/budgets
{"name": "end-user-budget-ai", "max_budget": 0.1, "reset_alignment": "calendar_day"}
```

Create `end-user-budget-memories` the same way. Each answer carries the
budget's generated `budget_id`, which the key names in the next step. An
organization's budget limits spend, tokens (`token_limit`) and requests
(`request_limit`) over its period; it carries no per-minute limits.

## 2. Create the service key

In **API keys**, **Create key**, open **Advanced**, tick **Service key**, pick
the **End-user budgets** the key may start its end users on, and the
**Default end-user budget** for a request that names none. The same key from
the API:

```http
POST /api/v1/organizations/me/keys
{"key_name": "mlpa", "is_service_key": true,
 "end_user_budget_ids": ["<ai budget id>", "<memories budget id>"],
 "end_user_budget_id": "<ai budget id>"}
```

- A key with a default and no list may assign the default alone.
- When both are set, the default must be on the list.
- Its end users go on the organization's own budgets, and a budget of the
  deployment or of another organization is refused with 404.
- Deleting one of the organization's budgets is refused while a key starts its
  end users on it by default, or while end users are on it; a key that only
  lists it loses it from the list.

Such a key belongs to the organization rather than to the person who made it:

- It is owned by a user created for it alone. End users belong to their key's
  user, and a person's own user serves every organization they belong to, so
  this keeps the key's end users out of reach of any other key or organization.
  That owner has no member ceiling, which is why only an owner or admin, who
  sets those ceilings, may create one.
- Every owner and admin lists, edits, rotates and deletes it on the same
  routes, and removing its creator from the organization leaves it working.
  Other members do not see it.
- It always has a `max_end_users`, 10,000 unless the request sets another, and
  an update cannot clear it (see
  [Capping a key's end users](#capping-a-keys-end-users)). Whether a key is a
  service key is fixed when it is created.

Its responses carry `is_organization_service_key: true`, which tells it apart
from a service key a deployment operator made in the same organization. To pool
every end user under one cap, give the key a spend ceiling of its own (see
[Set a budget and ceiling](set-a-budget-and-ceiling.md)).

## 3. Name the end user on each request

Send the end user's id the way any client names a user: the `user` field on
`/v1/chat/completions`, `/v1/responses` and `/v1/search`, and
`metadata.user_id` on `/v1/messages`. To start a new end user on a budget other
than the key's default, name it in the `Otari-End-User-Budget` header:

```http
POST /v1/chat/completions
Authorization: Bearer <service key>
Otari-End-User-Budget: <memories budget id>

{"model": "...", "user": "fxa123:memories", "messages": [...]}
```

- Send a stable id per end user, not a per-session or per-request value: each
  distinct id creates an end user, whether or not the request is then admitted,
  and only the key's
  [`max_end_users`](#capping-a-keys-end-users) caps how many it can create.
- An id is at most 256 characters and may otherwise be any string, `/`
  included, such as a base64 key id.
- The header applies only when the request creates the end user. An existing
  end user keeps its budget, whatever a later request names, so a request
  cannot undo a move made on the
  [end-user routes](#end-user-routes).
- A budget not on the key's list is refused with `403` and the code
  `end_user_budget_not_allowed`, even for an end user that already exists.
- The response carries `Otari-End-User-Budget` with the budget the end user is
  on, so the caller can see when it differs from the one it named.

A request that names nobody, or the key's own user, bills the key's user as it
would on any other key.

## 4. Manage end users

The service manages its own end users with its service key, addressing each one
by the id it names it by. It needs the key alone: no dashboard session, and not
the key's id. For
example, to move an end user to another budget:

```http
PATCH /api/v1/end-users?external_id=fxa123:memories
Authorization: Bearer <service key>

{"budget_id": "<ai budget id>"}
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
  `tpm_limit`) where it has them (a deployment's budget can; an organization's
  has none), and any `per: user` rate limit rule the deployment sets, each
  counting every end user on its own. Since the limits live on the budget, two service keys with
  different end-user budgets give their end users different limits, and moving
  an end user to another budget moves its limits too.
- Checks the key's own ceiling as well, so a spend ceiling on the API key pools
  every end user behind it. The key's user's per-user budget is not checked for
  an end user's request; use the key's ceiling as the pool.
- Keeps the key's user's rate limit and model allow-list, and any member
  ceiling of the key's user, in force for every end user. The rate limit is
  shared by all of them and is checked before an end user is created.

Blocking the key's user stops its end users too.

Moving an end user (to a budget on the key's list) restarts its period on the
new budget and applies that budget's per-minute limits, but keeps its spend,
tokens and requests so far, as moving any user between budgets does: an
end user that used up one budget can be over the next one's limits until that
period resets.

Deleting an end user erases its telemetry and agent traces, as deleting any
user does, and stops it being listed or counted. Its row stays, with what it
has spent and used: a request that names the same id again revives it with
those counters, on the budget it was on, so deleting an end user cannot reset
its budget. Its usage history stays too. When erasure fails the call answers
5xx and deletes nothing, so it can be repeated.

### Capping a key's end users

An organization's service key always has a `max_end_users`, 10,000 unless its
create or update request sets another. A key a deployment operator makes has
none unless the operator sets one (see
[Service keys on a deployment you run](operator-service-keys.md)). Either way,
send a stable id per end user rather than a per-session or per-request value.

With `max_end_users` set, a request that would create one more end user, or
revive one that was deleted, is refused with 403 and the code
`end_user_limit_reached`, and nothing is created; the end users the key already
has keep working. Deleting end users makes room again. `null`, the default, is
no cap on an operator's key, and `0` admits no new end user at all.

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
| `GET /api/v1/end-user-budgets` | Lists the budgets on the key's list, with each one's name, limits and period and how many end users are on it |

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

A deployment operator also reaches any service key's end users with the master
key: see [Operator routes](operator-service-keys.md#operator-routes).

## Where end users apply

End users are supported on `/v1/chat/completions`, `/v1/responses`,
`/v1/search` and `/v1/messages`. The other endpoints (embeddings, files,
batches and the other pass-through routes) treat a service key as an ordinary
key, so a `user` naming someone else is handled by the key's
[`reject_user_mismatch`](api-keys.md#mismatched-user-fields) setting
there.
