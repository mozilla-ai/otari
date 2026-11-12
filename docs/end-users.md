# Track spend per end user

An application that calls Otari for many people can give each of them their own
spend and budget, without a key per person. It uses a **service key**: a key
whose requests name an end user.

This guide sets up a service called `mlpa` with two features, each with its own
per-user limit. You need to be an organization owner or admin. To make service
keys on a deployment you run, see
[Service keys on a deployment you run](operator-service-keys.md).

## 1. Create the end-user budgets

Create one budget per feature in **Spend & budgets**, as in
[Create a budget](set-a-budget-and-ceiling.md#create-a-budget). For example:

- `end-user-budget-ai`, $0.10 a day
- `end-user-budget-memories`, with its own limit

Each end user gets a copy of the budget it starts on: the full limit and its own
reset period. An organization's budget can limit spend, tokens and requests per
period, but not requests or tokens per minute.

While a key uses a budget as its default, or end users are on it, the budget
cannot be deleted.

## 2. Create the service key

1. Open **API keys** and click **Create key**.
2. Enter a **Name**, such as `mlpa`.
3. Click **Advanced** and tick **Service key**.
4. Under **End-user budgets**, pick the budgets the key may start end users on.
5. Pick the **Default end-user budget**, used when a request names none. It
   must be one of the budgets you picked.
6. Set **Max end users**, or keep 10,000; see
   [Capping a key's end users](#capping-a-keys-end-users).
7. Click **Create key** and copy the key.

The key belongs to the organization, not to you:

- Every owner and admin can see, edit, regenerate and delete it. Other members
  do not see it.
- It keeps working if you leave the organization.
- Its end users stay out of reach of every other key and organization.
- It always has a cap on its end users, which cannot be cleared.

Whether a key is a service key is fixed when you create it. To cap the total
spend of all its end users together, give the key its own spend ceiling (see
[Apply a spend ceiling](set-a-budget-and-ceiling.md#apply-a-spend-ceiling)).

Through the API, a service key is created with `is_service_key`,
`end_user_budget_ids`, `end_user_budget_id` and `max_end_users`; see the
[OpenAPI specification](public/openapi.json).

## 3. Name the end user on each request

Send the end user's id in the request's `user` field (`metadata.user_id` on
Messages). A new end user starts on the key's **Default end-user budget**, or
with no budget of its own when the key has none. To start it on another budget on the key's
list, name that budget in the `Otari-End-User-Budget` header:

```http
POST /v1/chat/completions
Authorization: Bearer <service key>
Otari-End-User-Budget: <memories budget id>

{"model": "...", "user": "fxa123:memories", "messages": [...]}
```

- **Use a stable id per person.** Every new id creates an end user, even if the
  request is then refused, so a per-session or per-request id fills the key's
  [cap](#capping-a-keys-end-users).
- An id is 1 to 256 characters and can be any string, `/` included. An empty
  id counts as naming no one.
- The header only matters when the request creates the end user. An existing end
  user keeps its budget; to change it, [move the end user](#4-manage-end-users).
- A budget not on the key's list is refused with `403`
  (`end_user_budget_not_allowed`), even for an existing end user.
- The response's `Otari-End-User-Budget` header names the budget the end user is
  actually on, including on a `403` (`budget_exceeded`) from that budget.

A request that names no one, or the key's own user, bills the key's user as on
any other key. Such a request, or any request on a key that is not a service
key, is refused with `400` (`end_user_budget_without_end_user`) if it sends
`Otari-End-User-Budget`, since there is no end user for the budget to apply to.

End users work on `/v1/chat/completions`, `/v1/responses`, `/v1/search` and
`/v1/messages`. Other endpoints treat a service key as an ordinary key, so a
`user` naming someone else there follows the key's
[mismatched `user` field](api-keys.md#mismatched-user-fields) setting.

## 4. Manage end users

To see a key's end users, open the key's menu in **API keys** and click **End
users**: each one's budget, spend, requests and whether it is blocked. Search
by the id your service sends. The list is read-only.

Your service changes its end users with its service key, addressing each by the
id it sends. For example, to move an end user to another budget:

```http
PATCH /api/v1/end-users?external_id=fxa123:memories
Authorization: Bearer <service key>

{"budget_id": "<ai budget id>"}
```

URL-encode the id (`/` as `%2F`, `+` as `%2B`). To clean up short-lived end
users, such as one per guest session, delete them by age with
`DELETE /api/v1/end-users?created_before=2026-10-10T00:00:00Z`, and repeat until
the answer's `deleted` is 0. Every route is listed under
[End-user routes](#end-user-routes).

## How end users are billed and capped

For a request that names an end user, Otari:

- **Bills the end user**, creating it on first use. End users belong to the
  key's user, so two services that both name `alice` get two separate end users.
- **Checks the end user's budget**, including its per-minute limits where the
  budget has them (a deployment's budget can) and any `per: user` rate limit
  rule the deployment sets.
- **Checks the key's own spend ceiling**, which pools all the key's end users.
  The key's user's own budget is not checked.
- **Keeps the key's user's rate limit, model access and member ceiling** in
  force. The rate limit is shared by all end users.

Blocking the key's user stops all its end users.

When an end user's own budget refuses a request, or the end user is blocked,
the refusal names it by the id the request gave, in the message and as
`end_user` in the body:
`{"detail": "End user 'alice' is blocked", "code": "user_blocked", "end_user": "alice"}`.
A refusal from any other budget, such as the key's ceiling, carries no
`end_user`.

**Moving an end user** to another budget on the key's list restarts its period
and applies the new budget's limits, but keeps what it has spent so far. An end
user that used up one budget can be over the next one's limit until its period
resets.

**Deleting an end user** erases its telemetry and agent traces and stops it
being listed or counted, but keeps its spend counters and usage history. A
request naming the same id later revives it with those counters, on its old
budget, so deleting cannot reset a budget. If erasure fails, the call answers
5xx, deletes nothing, and can be repeated.

### Capping a key's end users

Set the cap in **Max end users** when you create or edit the key in **API
keys**, or as `max_end_users` through the API. An organization's service key
always has one, 10,000 unless you set another, and an edit cannot clear it. A
key a deployment operator makes has no cap unless the operator sets one (see
[Service keys on a deployment you run](operator-service-keys.md)).

When the cap is reached, a request that would create or revive an end user is
refused with `403` (`end_user_limit_reached`). Existing end users keep working,
and deleting end users makes room. A cap of `0` admits no new end user.

The cap counts the end users on budgets the key can assign. So two service keys
of one user, each with its own budgets (say, one for signed-in users and one for
guests), keep separate caps:

- A key with a list and no default also counts end users on no budget, where a
  request naming none starts them.
- Moving an end user off the key's budgets frees a place; moving one on takes
  a place, and is refused when the key is full.
- A key that assigns no budget counts every end user of its user.

The cap holds across replicas.

## End-user routes

All of these take only the service key. Another API key, the master key and a
dashboard session are refused, and so is a service key whose own user is blocked
or deleted.

| Request | Does |
|---|---|
| `GET /api/v1/end-users` | Lists end users, oldest first |
| `GET /api/v1/end-users?external_id=...` | Reads one end user (its budget, counters and blocked state) as a list of one, or an empty list |
| `PUT /api/v1/end-users?external_id=...` `{"budget_id": ...}` | Puts the end user on a budget, creating it if it has not made a request yet (201) |
| `PATCH /api/v1/end-users?external_id=...` `{"blocked": true}` or `{"budget_id": ...}` | Blocks, unblocks or moves the end user |
| `DELETE /api/v1/end-users?external_id=...` | Deletes the end user, answering `{"deleted": 1}`, or 404 |
| `DELETE /api/v1/end-users?created_before=...` | Deletes up to `limit` (default 100, at most 1000) of the oldest end users created before that time, answering `{"deleted": n}` |
| `GET /api/v1/end-user-budgets` | Lists the key's budgets, with each one's limits, period and end-user count |

- An id the key does not own answers 404, as one that does not exist does.
- A budget set through these routes must be on the key's list, or the request is
  refused with `403` (`end_user_budget_not_allowed`).
- `GET /api/v1/end-users` takes `skip` and `limit` (up to 1000), filters on
  `budget_id`, `blocked` and `external_id`, and with `include_total=true` puts
  the match count in the `Otari-Total-Count` header. Lists and counts leave out
  deleted end users; counts include blocked ones, which
  `blocked_end_user_count` also gives on its own.
- The id can also go in the path, `/api/v1/end-users/{external_id}`, for `GET`,
  `PUT`, `PATCH` and `DELETE` (the `GET` answers with the end user itself, the
  `DELETE` with 204). Use it only for ids without `/`: proxies and clients can
  rewrite `//`, `.` and `..` in a path and reach another end user. An id with a
  `.` or `..` segment is refused there with 400.

A deployment operator can also reach any service key's end users with the master
key; see [Operator routes](operator-service-keys.md#operator-routes). An owner
or admin lists the end users of the organization's service keys with a
dashboard session at `GET /api/v1/organizations/me/keys/{key_id}/end-users`,
which takes the same list parameters.
