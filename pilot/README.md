# Pilot: Firefox → MLPA → Otari, on one machine

The pilot from `pilot.md`, runnable locally. MLPA runs with
`GATEWAY_BACKEND=otari` (MLPA branch `otari-pilot`) in front of this Otari
branch, which calls stand-ins for Vertex, Mistral, Exa and Liner.

```
Firefox (Smart Window) ─┐
check.py (as desktop,   ┼──► MLPA :8080 ──► Otari :8100 ──► fakes.py :9100
  Android and iOS)      ┘                    │   (vertex_ai, mistral, exa_answers,
                                             │    liner_answers, searx)
                                             ├──► Postgres :55432 (otari, app_attest)
                                             └──► Redis :56379 (rate_limits)
```

## Run it

```bash
./run.sh up        # Postgres + Redis in Docker, then fakes, Otari, MLPA
./run.sh check     # 30 end-to-end checks through MLPA
./run.sh firefox   # Smart Window on a fresh profile, pointed at the local MLPA
./run.sh down      # stops everything and drops the databases
```

`MLPA_DIR` (default `~/MLPA`) is the MLPA checkout; `FIREFOX_APP` (default the
artifact build under `~/firefox`) is the Firefox `./run.sh firefox` starts. The
Otari dashboard is at http://127.0.0.1:8100 with master key `sk-otari-pilot`.

`up` provisions MLPA in Otari with MLPA's own script
(`scripts/otari_provision.py`), the only writer: per service type, an Otari
budget named after MLPA's budget id, carrying MLPA's dollar cap and its per-user
RPM and TPM, and an owner user `mlpa-<service type>` with one service key. MLPA
itself only reads at startup.

## Firefox

`./run.sh firefox` copies `firefox/user.js` into a fresh profile. Sign in to a
Mozilla account in that profile: Smart Window sends its FxA token to MLPA,
which verifies it against production accounts (`MLPA_DEBUG=false`). Keeping
the answers path (Exa citations) on needs the Firefox branch
`smartwindow-mlpa-pilot-endpoint`, which adds
`browser.smartwindow.endpoint.isMLPA`; without it, any overridden endpoint
turns that path off.

Android and iOS reach the same MLPA through their debug endpoint settings,
which only offer dev, stage and prod. They need a deployed MLPA, so
`check.py` stands in for them here. It sends their service types, models and
streaming modes, and authenticates with an MLPA access token
(`use-play-integrity`).

## What the checks cover

- **Clients.** Smart Window chat (streamed and not), memories, search and
  answers; iOS summarize and Quick Answers via both Exa and Liner; Android
  Shake to Summarize; the eval harness's `vertex_ai/...` model name.
- **Citations.** Exa and Liner citations arrive under
  `message.provider_specific_fields.citations`.
- **Billing.** Spend lands on the end user, under its service type's budget,
  for search as well as chat. MLPA's `metadata` never reaches a provider.
- **Refusal codes.** Per-user budget gives `{error: 1}`, per-user RPM gives
  `{error: 2}` (each service type separately), a prompt too long for the model
  gives `{error: 3}`, a provider 429 gives `{error: 5}`, and an unknown model
  gives `{error: 8}`. MLPA maps these from Otari's error code, with no text
  matching.
- **Admin API.** MLPA's block, unblock, budget move, user info, list and
  per-service-type counts, all backed by Otari's users API. Moving a user to
  another budget moves its per-minute limits too.

## What it took

Otari, branch `feat/mlpa-pilot`:

- Stable refusal codes, as an `Otari-Error-Code` header, as `code` in the
  body, and as `error.code` in a stream's error event. Budget refusals also
  send `Otari-Budget-Scope`, which tells MLPA's per-user code 1 from its
  global code 10.
- `rpm_limit` and `tpm_limit` on budgets, per user, where LiteLLM keeps them.
  Tokens count what requests used: MLPA sends `max_tokens: 8192` against a
  2,000 TPM per user, so admitting on an estimate would refuse every request
  (`tpm_admission: used` gives `rate_limits` rules the same option).
- Search bills a service key's end users and counts the per-minute limits.
- `GET /users` filters (`parent_user_id`, `external_id`, `blocked`) and
  `include_total`.
- Provider extras are copied under `provider_specific_fields`.

MLPA, branch `otari-pilot`:

- `GATEWAY_BACKEND=otari` and `OtariService`. The service is a drop-in for the
  LiteLLM database service, so the admin API, signup cap and startup code are
  unchanged.
- One service key per service type, plus the provision script.
- Error mapping by code, from the header or the body, and from a coded
  stream error event.
- Startup only reads; the provision script is the one writer.
- Readiness and version read from Otari.
- The stream parser now accepts OpenAI-shaped chunks (`usage: null`,
  `choices: []`).

## Not covered yet

- **Real upstreams.**
  - Otari's `vertexai` provider calls `generateContent`. Whether Vertex's
    partner models (Qwen, and Mistral on Vertex) accept that is unconfirmed.
    The fallback is an `openai` instance on Vertex's OpenAI-compatible
    endpoint.
  - Liner's quick-answer API is not OpenAI-compatible (`answer` plus
    `references`). `fakes.py` holds the translation; done properly, it is an
    any-llm provider.
- **Spend tags.** Purpose and country per request (`spend_logs_metadata`) have
  no Otari equivalent yet.
- **`x-litellm-*` metrics.** MLPA's backend, fallback and cost metrics read
  `unknown` or 0 under Otari.
- **Search response.** Otari returns `date` where Firefox reads
  `publishedDate`. LiteLLM's response has the same gap.
- **Global budget (code 10).** It maps from any non-user budget scope, but no
  scoped ceiling is provisioned on the service keys.
- **Budget admission.** Otari reserves each request's estimated cost up front.
  With `max_tokens: 8192`, a user near their cap is refused sooner than
  LiteLLM, which checks only spend so far.
- **Signup cap.** A user admitted under the cap whose first request then fails
  before reaching Otari keeps the claim until MLPA's next startup reconcile.
