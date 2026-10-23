# Health checks

Health checks tell an operator which provider is failing before a caller does.
On a schedule, Otari sends one small completion to every chat model the
deployment's hosted providers serve, and the Health page shows what each one
answered. An alert list is emailed when a model stops answering and again when
it answers.

Health checks are for operators only, and every deployment offers them.
`health_checks_enabled: false` in `config.yml` (or
`OTARI_HEALTH_CHECKS_ENABLED=false`) removes the page, its routes and its worker;
it is read at startup, so changing it needs a restart.

## What is checked

A run checks the providers on the **Hosted providers** page, and those declared
under `hosted_providers:` in `config.yml`. A model is checked when it is switched
on under a provider that is enabled and whose key the deployment can read: the
models the deployment would serve a request for. A model listed only because it
has a price, a model switched off, and a provider switched off are not checked.
A provider restricted to one organization is checked like any other.

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

Every chat model is sent the prompt, `Reply with the single word: pong` by
default, capped at 16 output tokens. Any answer passes, whatever it says. An
error, or no answer within 60 seconds, fails. Embedding, image, speech and
rerank models are skipped.

A check goes straight to the provider with that hosted provider's own key and
endpoint. It
writes no usage, is billed to no organization and counts against no budget or
rate limit, but the provider charges for every request.

## Schedule

The schedule is off until an operator turns it on in the Schedule card. Its
settings:

| Setting | Default | Range |
| --- | --- | --- |
| Minutes between runs | 60 | 15 to 1440 |
| Prompt | `Reply with the single word: pong` | 1 to 2000 characters |
| Max output tokens | 16 | 1 to 4096 |
| Max output tokens for reasoning models | 256 | 1 to 8192 |
| Include reasoning models every Nth run | 6 | 0 to 48, where 0 leaves them out and 1 includes them every run |

Reasoning models spend tokens thinking before they answer, so they get a larger
cap and, because they cost the most, join only every Nth run.

With several replicas, every replica polls once a minute and exactly one starts
each run. Results are kept for 48 hours. When a model that was failing answers
again, the outage is recorded in the database (when it started, when it ended,
how many runs it failed and its last error) and kept for 90 days.

## Email alerts

Add addresses to **Email alerts** on the Health page. Each run compares its
results with what the list was last told:

- A model that fails and was not reported is emailed as stopped answering.
- A reported model that answers is emailed as answering again, with how long it
  was failing.
- A model that keeps failing, or keeps answering, sends nothing.

So an outage sends two emails however many runs it lasts. One run's changes
share one email, grouped by provider, and its subject names the deployment's
host, so mail from several deployments says which one it is about.

Alerts need outgoing mail; see [Configuration](configuration.md) for the
`smtp_*` settings, `mail_from_email` and `public_base_url`. When no address
receives an alert, the next run sends it again. A failure found while the list
is empty is remembered, so an address added later hears about the next change
rather than the history before it.

## API

The page is built on five operator endpoints under `/api/v1/health-checks`:
`GET` for the status, `POST /run` to start a run, `POST /disable-failing` to
switch off every failing model (what **Disable all** calls), and `GET` and
`PATCH /settings`.
See the [API reference](api-reference.md).
