# Service keys on a deployment you run

On a deployment you run, the operator can make
[service keys](end-users.md) whose end users go on the deployment's own budgets,
which can carry per-minute limits, and can declare them in config.yml. This
needs the master key or an operator session. An organization's owners and
admins make their own service keys without either, as
[Track spend per end user](end-users.md) describes; naming end users, managing
them and how they are billed is the same for both and is on that page.

The example is the same service, `mlpa`, with two features, each carrying its
own per-user limit.

## Create deployment budgets for end users

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
`user_count`. Each entry an operator's key may assign must be a deployment
budget, not an organization's; an organization's own service key lists that
organization's budgets instead (see
[Track spend per end user](end-users.md#2-create-the-service-key)).

## Create the service key

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

## Or declare it in config.yml

A service whose keys and budgets should not depend on a provisioning script can
declare both in `config.yml`:

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

## Limit how many end users a key creates

Each distinct id a request names creates an end user, so set a
[rate limit](configuration.md#rate-limit-rules) on a deployment that issues
service keys. An operator's service key has no `max_end_users` unless it is set
in `POST` or `PATCH /api/v1/keys`, or in config.yml; `null` is no cap. How the
cap counts is in
[Capping a key's end users](end-users.md#capping-a-keys-end-users).

## Operator routes

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
