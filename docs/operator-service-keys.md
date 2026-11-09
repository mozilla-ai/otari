# Service keys on a deployment you run

With the master key or an operator session, the operator can make
[service keys](end-users.md) whose end users go on the deployment's own budgets
(which can carry per-minute limits), and can declare them in `config.yml`.
Organization owners and admins make their own service keys instead; naming,
managing and billing end users works the same way and is on
[Track spend per end user](end-users.md).

The example is a service, `mlpa`, with two features, each with its own per-user
limit.

## Create deployment budgets for end users

Give each budget your own id with `PUT /api/v1/budgets/{budget_id}`, which
creates or replaces it, so the same request can run at every deploy:

```http
PUT /api/v1/budgets/end-user-budget-ai
{"max_budget": 0.1, "reset_alignment": "calendar_day", "rpm_limit": 40, "tpm_limit": 2000}
```

Create `end-user-budget-memories` the same way.

- An id is up to 128 letters, digits, `.`, `_` and `-`, starting with a letter
  or digit. `POST /api/v1/budgets` generates one instead.
- `GET /api/v1/budgets` reports the users on each budget in `user_count`.
- An operator's key may assign only deployment budgets, not an organization's
  (an organization's own service key lists that organization's budgets; see
  [Track spend per end user](end-users.md#2-create-the-service-key)).

## Create the service key

Signed in as the deployment operator, open **API keys**, click **Create key**,
click **Advanced** and tick **Service key**. Pick both budgets under
**End-user budgets** and `end-user-budget-ai` as the **Default end-user
budget**, or end users a request starts without a budget header are uncapped.
Through the API, only a deployment
operator can set `is_service_key` on `POST` or `PATCH /api/v1/keys`. List the
budgets the key may start its end users on in `end_user_budget_ids`, and set
`end_user_budget_id` as the default for requests that name none:

```http
POST /api/v1/keys
{"user_id": "mlpa", "is_service_key": true,
 "end_user_budget_ids": ["end-user-budget-ai", "end-user-budget-memories"],
 "end_user_budget_id": "end-user-budget-ai"}
```

- A key with a default and no list may assign the default alone.
- When both are set, the default must be on the list.
- Deleting a budget takes it off every list. A budget that is a key's default
  cannot be deleted until that default changes, or new end users would start
  uncapped.
- Give each service its own user: end users belong to the key's user, so every
  service key of one user reaches the same end users.

To pool every end user under one cap, give the key its own budget (its
ceiling).

## Or declare it in config.yml

To avoid a provisioning script, declare keys and budgets in `config.yml`:

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

`ceiling` pools every end user. The service reads the same `MLPA_SERVICE_KEY`
and presents it as its API key, so no minted secret has to be handed over, and
restarts and replicas do not change it. For the fields, the secret's format and
what a start writes back, see
[Budgets and API keys in config](configuration.md#budgets-and-api-keys-in-config).

## Limit how many end users a key creates

Each distinct id a request names creates an end user, so set a
[rate limit](configuration.md#rate-limit-rules) on a deployment that issues
service keys. An operator's service key has no cap unless you set **Max end
users** on the key in **API keys**, `max_end_users` in `POST` or
`PATCH /api/v1/keys`, or `max_end_users` in `config.yml`; empty (`null`) means
no cap. **End users** in the key's menu lists them. How the cap counts is in
[Capping a key's end users](end-users.md#capping-a-keys-end-users).

## Operator routes

With the master key, an operator reaches any service key's end users through:

- `GET`, `PUT` and `PATCH /api/v1/keys/{key_id}/end-users?external_id=...`
  (`GET` also takes the list's other parameters)
- the path form `/api/v1/keys/{key_id}/end-users/{external_id}`

These work even when the key's user is blocked or deleted. A budget set there
must be on the key's list.

The users API lists end users across keys. Each carries `parent_user_id` (the
key's user) and `external_id`, and
`GET /api/v1/users?parent_user_id=...&external_id=...` filters on them. It can
put an end user on any deployment budget, not only one on a key's list.
