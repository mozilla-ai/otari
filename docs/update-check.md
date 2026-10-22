# Update check

The update check tells an operator when a new otari release is out, and above
all when a security fix covers the version they run. Once a day, a deployment
that has it on asks otari.ai whether a newer release or a security advisory for
its version exists, and shows the answer in the dashboard and in the logs.

Nothing is sent until somebody decides. The first time a deployment operator
opens the dashboard, they are asked; the answer is stored and the question does
not come back. Closing the question without answering sends nothing and asks
again on the next visit.

## What is sent

Each check is one `POST` to `https://api.otari.ai/api/v1/update-checks` (the
default; `update_check_url` in `config.yml`, or `OTARI_UPDATE_CHECK_URL`,
sends it elsewhere) with this body, and nothing else:

```json
{"version": "0.30.1", "install_id": "2f1c7e0e-8a43-4f53-9a6e-2c1d4f1e9b70"}
```

- `version` is the otari release this deployment runs.
- `install_id` is a random UUID created the first time the deployment reads
  its update check state, and stored in its database. It is not derived from
  the host, its address or anything else about the deployment, and it is the
  same on every replica that shares the database. It is what lets otari.ai
  count one install once a day rather than once per restart.

The request carries no API key, no session, and nothing about users, keys,
budgets, providers, models or traffic. It is a `POST` rather than a `GET` so the
install id is not part of the URL, and so not in URL access logs on the way. A
proxy that logs request bodies would still see it.

otari.ai keeps, for each install id, the days it checked and the version it ran
that day: not the time of day, and not the address the check came from. An
install seen on one day only is dropped after 30 days. Only installs seen on
several days are counted, so a deployment tried once and removed is not.

The Settings page shows the install id, the version and the address of each
check, so what is sent can be read off the deployment itself.

## What comes back

The newest otari release the otari team chose to announce, a link to its notes,
and the published security advisories for otari whose affected range includes
the version that was sent. Not every release is announced, so a busy release
week does not interrupt you daily; security advisories are never held back.

The dashboard shows a band to deployment operators when an advisory covers the
running version, or when a newer announced release exists, and the gateway logs
a warning for an advisory. Dismissing the release band hides it in that browser
for that release, and for 30 days for any later one.

## Turning it on or off

- **In the dashboard:** **Settings → Update check**, or the question asked on
  first visit. Operators only.
- **In configuration:** `update_check: true` or `false` in `config.yml`, or
  `OTARI_UPDATE_CHECK=true|false`. This answers for the operators: the
  dashboard shows the answer and cannot change it. Use it for a deployment
  nobody opens the dashboard of, or to decide for many deployments at once.
- `DO_NOT_TRACK=1` ([Console Do Not Track](https://consoledonottrack.com/)) and
  `CI` both answer no, unless `update_check` is set.

Turning the check off forgets what the last check heard. The install id is kept.

## When it runs

A deployment checks once a day, at a time of day its install id picks, so
installs spread over the whole day rather than checking together, and an
install that restarts keeps its time. An operator who turns the check on gets
a first check within a minute; a deployment turned on by configuration waits
for its time of day. Every replica polls, and the replica whose update claims
the due check is the only one that sends it.

A check that fails is retried after a randomized wait that doubles from five
minutes up to six hours, never later than the next daily check, and a
`Retry-After` from otari.ai is honored. A failure is logged at info level and
never affects serving.

## Where it does not run

A development build, whose version is not a release (`0.0.0-dev`, or the
commit an image built from `main` carries), is never checked: only the release
images report a version to compare.

## API

`GET /api/v1/update-check` returns the state: whether checks are on and who
decided (`config`, `operator`, or nobody yet), what is sent and where, and what
the last check heard. `PUT /api/v1/update-check` with `{"enabled": true}` or
`{"enabled": false}` records an operator's answer, and answers `409` when the
configuration decides. Both need the master key or a deployment operator's
session.
