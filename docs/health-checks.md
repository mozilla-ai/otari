# Health checks

On a schedule, Otari sends one small completion to every chat model the
deployment's hosted providers serve, and the Health page shows what each one
answered. An alert list is emailed when a model stops answering and again when
it recovers.

Health checks are for operators only. To remove the page, its routes and its
worker, set `health_checks_enabled: false` in `config.yml` (or
`OTARI_HEALTH_CHECKS_ENABLED=false`) and restart.

## What is checked

A run checks the providers on the **Hosted providers** page and those under
`hosted_providers:` in `config.yml`. It checks the models the deployment would
serve: switched on, under an enabled provider whose key the deployment can
read, including providers restricted to one organization. Models listed only
for their price, switched-off models and switched-off providers are skipped.

## The Health page

Open **Deployment → Health** in the dashboard. It shows:

- One row per provider: OK, some failing, or down, how many of its models
  answered, the most common error, and a bar per run over the last 24 hours.
- The models that are failing, with the HTTP status, the error and when the
  model started failing, 25 to a page. A model leaves the list as soon as a run
  finds it answering, or once it is switched off. **Show all** lists every model.
- **Disable all**, which switches off every failing model on its hosted provider
  after a confirmation. The models stay listed and priced on the Hosted providers
  page, where they are switched back on, and are not checked while they are off.
- **Run now**, which starts a run within a minute whether or not the schedule is on.

## What a check sends

Every chat model gets the prompt (`Reply with the single word: pong` by
default), capped at 16 output tokens. Any answer passes; an error, or no answer
within 60 seconds, fails.

Models of kind embedding, rerank, transcription, image, speech, video, realtime
or `other` are skipped. The kind comes from the provider's listing, else from
models.dev (see `type` in [Models](models.md)); a model of no known kind is
checked as chat.

A check goes straight to the provider with that hosted provider's key and
endpoint. It writes no usage and counts against no budget or rate limit, but
the provider charges for every request.

## Schedule

The schedule is off until an operator turns it on in the **Schedule** card:

| Setting | Default | Range |
| --- | --- | --- |
| Minutes between runs | 60 | 15 to 1440 |
| Prompt | `Reply with the single word: pong` | 1 to 2000 characters |
| Max output tokens | 16 | 1 to 4096 |
| Max output tokens for reasoning models | 256 | 1 to 8192 |
| Include reasoning models every Nth run | 6 | 0 to 48, where 0 leaves them out and 1 includes them every run |

Reasoning models get a larger cap because they think before answering, and
join only every Nth run because they cost the most.

With several replicas, exactly one starts each run. Results are kept for 48
hours. When a failing model answers again, the outage (start, end, failed runs,
last error) is recorded and kept for 90 days.

## Email alerts

Add addresses to **Email alerts** on the Health page. Each run compares its
results with what the list was last told:

- A model that fails and was not reported is emailed as stopped answering.
- A reported model that answers is emailed as answering again, with how long it
  was failing.
- A model that keeps failing, or keeps answering, sends nothing.

An outage therefore sends two emails however long it lasts. One run's changes
share one email, grouped by provider, with the deployment's host in the
subject.

The list is also emailed, once per model, when a hosted model is switched off
because its provider refused to serve it (see
[Hosted providers](hosted-providers.md)). A run that meets that refusal
switches the model off itself and leaves it out of the run's alert.

Alerts need outgoing mail: the `smtp_*` settings, `mail_from_email` and
`public_base_url` (see [Mail](configuration.md#mail)). When no address receives
an alert, the next run sends it again. An address added later hears about the
next change, not earlier history.

## API

The page uses five operator endpoints under `/api/v1/health-checks`:
`GET` for the status, `POST /run` to start a run, `POST /disable-failing` to
switch off every failing model (what **Disable all** calls), and `GET` and
`PATCH /settings`.
See the [API reference](api-reference.md).
