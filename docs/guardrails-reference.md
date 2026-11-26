# Guardrails reference

The fields, headers, states and endpoints of guardrails. A guardrail checks
what a model is asked before the provider is called, on
`/api/v1/chat/completions`, `/api/v1/messages` and `/api/v1/responses`.
[Configure guardrails](configure-guardrails.md) covers setting one up in the
dashboard, in a request and over the API.

Contents:

- [The `guardrails` request field](#the-guardrails-request-field)
- [What a guardrail reads](#what-a-guardrail-reads)
- [Modes](#modes)
- [Reading the `Otari-Guardrails` header](#reading-the-otari-guardrails-header)
- [When the guardrails service is unreachable](#when-the-guardrails-service-is-unreachable)
- [How the layers compose](#how-the-layers-compose)
- [Mandate fields](#mandate-fields)
- [Definition fields](#definition-fields)
- [How Otari builds a definition](#how-otari-builds-a-definition)
- [Telling whether a definition is running](#telling-whether-a-definition-is-running)
- [Which profiles exist, and what they take](#which-profiles-exist-and-what-they-take)
- [The guardrail catalog](#the-guardrail-catalog)
- [What a guardrails service answers](#what-a-guardrails-service-answers)
- [How long a service has to answer](#how-long-a-service-has-to-answer)

## The `guardrails` request field

A top-level list on `/api/v1/chat/completions`, `/api/v1/messages` and
`/api/v1/responses`. Each entry carries:

| Field | Meaning |
| --- | --- |
| `profile` | Required. The profile to run, as [the profiles read](#which-profiles-exist-and-what-they-take) names it. |
| `mode` | `monitor` (default) or `block`. See [Modes](#modes). |
| `on_unavailable` | `block` (default) or `monitor`. See [When the guardrails service is unreachable](#when-the-guardrails-service-is-unreachable). |
| `validate_kwargs` | Extra arguments for the check, merged on top of the profile's own. |
| `url` | A guardrails service to send this check to instead of the deployment's. It is the service's base address: Otari adds `/validate` to it, so `https://guard.example/api` is called at `https://guard.example/api/validate`, and a `url` with a query string or a fragment is refused with a `400`. It must be on a public address: loopback, private and other internal addresses are refused with a `400`, and Otari connects to the address it checked, with no redirect followed. |
| `credential` | Sent to `url` as `Authorization: Bearer`. Requires `url`, which must then be `https`, and an entry carrying one must be the only entry for its profile. Dropped when an organization or a routing policy mandates the same profile, because the mandate's endpoint replaces yours. |
| `on` | What the guardrail reads: any of `input`, `tool_results`, `tool_calls` and `output`. Defaults to `["input"]`. See [What a guardrail reads](#what-a-guardrail-reads). |

An entry runs a profile on a guardrails service: the entry's own `url`, or else
the deployment's `guardrails_url`. A guardrail your organization defined from
the [catalog](#the-guardrail-catalog) runs only through an organization
[mandate](#mandate-fields), on otari.ai as on any deployment, so an entry can
tighten a mandated profile but cannot start a catalog guardrail by naming it.
An entry with no service to run on is
[not configured](#when-the-guardrails-service-is-unreachable).

## What a guardrail reads

`on` lists what a guardrail reads, on all three endpoints:

| `on` | What it reads | When |
| --- | --- | --- |
| `input` | The user content that arrived since the model's last turn. A final assistant message (a prefill) is not a turn. | Before the provider is called. |
| `tool_results` | Every tool result in the request (a Chat Completions `tool` message, a Messages `tool_result` block, a Responses `function_call_output`), and each result of a tool Otari runs (MCP, web search, web fetch, code execution) before the model reads it. | Before the provider is called, and inside the tool loop. |
| `tool_calls` | Each call the model makes: to a tool Otari runs, before it runs, and to one the client runs, before the answer reaches the client. A call's arguments are read as JSON. | Inside the tool loop, and once the answer is complete. |
| `output` | The text of the model's answer. | Once the answer is complete. |

An older tool result is read again on every request because a client resends
the results it holds, and a result withheld once has to stay withheld. Otari's
own `[tool error]` messages are never read.

Each piece of content is its own check, and the checks on one request run
concurrently. A gateway worker keeps the verdicts it got for about an hour, by the check (profile, endpoint, credential and arguments, or a
definition's version) and the exact text, so a tool result resent on every turn
of an agent loop costs one call to the guardrail per worker. A check that gave no
verdict is asked again.

Images and documents are not read, and neither is a Responses `shell_call_output`,
whose output is stdout and stderr chunks rather than text. A guardrail on `output`
or `tool_calls` needs the whole answer, so a streamed answer is held until it is complete and then sent
at once.

## Modes

| Mode | What it does when the guardrail fires |
| --- | --- |
| `monitor` (default) | Records the verdict on the `Otari-Guardrails` response header and the request's trace, and changes nothing. |
| `block` | Refuses flagged input with `403` (`guardrail_violation`) before the provider is called, and a flagged answer with the same `403` after it ran (the answer is billed). A flagged tool result is withheld: the model reads a `[tool error]` message saying Otari withheld it and which profile flagged it, and the request goes on, so an agent can recover. A flagged call to a tool Otari runs is skipped the same way. A flagged call the client would run refuses the answer. |

The client still holds the original of a tool result Otari withheld; Otari
replaces it again each time it is sent.

A `403` lists each guardrail that refused:

```json
{"detail": {"message": "Request blocked by guardrail policy.", "code": "guardrail_violation",
            "guardrails": [{"profile": "lakera", "explanation": null, "score": 1.0, "categories": ["prompt_attack"]}]}}
```

`explanation` and `score` are the guardrail's own. `categories` names what the
guardrail says fired, such as `prompt_attack` or `pii/phone_number`, and is an
empty list when it names nothing. The Messages endpoint puts the same fields in
its `error` object. The `Otari-Guardrails` header carries the same names. The
request's trace carries them too, as long as they fit: it stores them as one
comma-separated value of at most 128 characters, so it keeps the leading names
that fit and leaves off the rest. Two 64-character names, for example, leave
the trace only the first.

Where the names come from:

- A guardrail an organization [defined](#definition-fields): the categories the
  any-guardrail library reports as triggered. Lakera Guard names the detectors
  that fired, from the per-detector breakdown it returns while the definition
  leaves `breakdown` on, which is the default; with `breakdown` off it names
  nothing. A guardrail whose library reports no categories, such as Bedrock
  Guardrails, names nothing.
- A guardrails service: the optional `categories` in its
  [`/validate` result](#what-a-guardrails-service-answers).

A name comes from a vendor, or from a service the caller chose, so only short
identifiers are kept: at most 8, each at most 64 characters of letters, digits
and `_ . : / @ + -`, and none shaped like an email address or a credential.
Anything else is dropped rather than shown, and so is the vendor's own message.

## Reading the `Otari-Guardrails` header

When input guardrail results are available, both non-streamed and streamed
responses carry one header entry per profile. When no guardrails apply, the header
is absent. A `block` guardrail refuses a flagged request before any bytes are sent.

```json
[{"profile": "prompt-injection", "mode": "monitor", "valid": null, "score": null, "status": "unavailable", "reason": "not_configured"}]
```

A profile that read several pieces of content reports the one that says the
most: a flag, then a check that could not run, then one with no verdict. A tool
result read again only so it stays withheld is reported only when it was.
`withheld` counts the tool results and tool calls the profile withheld, and is
absent at zero:

```json
[{"profile": "prompt-injection", "mode": "block", "valid": false, "score": 1.0, "status": "evaluated", "withheld": 1}]
```

`categories` lists what the guardrail says fired, as a [refusal](#modes) does,
and is absent when it names nothing:

```json
[{"profile": "lakera", "mode": "monitor", "valid": false, "score": 1.0, "status": "evaluated", "categories": ["pii/phone_number"]}]
```

A streamed response's header is sent before the answer, so it does not count what
the tool loop or the answer's guardrails did; the request's trace records each
check.

`valid: null` alone cannot tell a check that did not run from one that ran and
gave no verdict, so `status` says which:

| `status` | Meaning |
| --- | --- |
| `evaluated` | The check ran. `valid` is `true` or `false`. |
| `inconclusive` | The check ran and the guardrail returned no verdict. |
| `unavailable` | The check could not run. The request was served anyway, or was refused with a `502` or a `424` that carries this header. `reason` says why. |

`reason` appears only on an `unavailable` entry:

- `not_configured`: no guardrails service is configured for the profile: the
  entry names no `url` and the deployment has no `guardrails_url`.
- `http_<status>`: the service answered the `/validate` call with that error
  status, for example `http_404`. A `404` means the service has no `/validate`
  at the address Otari called, or does not serve the profile.
- `evaluation_failed`: everything else (the service is down or answered
  something malformed). The detail goes to the gateway's log, not to the
  caller.

The guardrail's own explanation is never put in the header.

## When the guardrails service is unreachable

A `block` guardrail that cannot be evaluated at all (service down, no URL
configured, malformed response) **fails closed**: input it cannot read is
refused with a `502` and the code `guardrail_unavailable` rather than forwarded
unchecked, and an answer with a `403`.
When the reason is that no guardrails service is configured for the profile,
the refusal is a `424` with the code `guardrail_not_configured` instead, since
no retry can fix it. Its message says what to change: a profile you named in
the request needs a `url` on its entry, or the entry removed, and a mandated
one needs an administrator of your organization or of the deployment to
configure it.
A tool result it cannot read is withheld instead, as a flagged one is, with a
message saying it could not be checked, and a tool call Otari would run is
skipped. A `monitor` guardrail fails open, since it was never enforcing.

A service that answers with no verdict (`valid: null`) has not evaluated the
input either, so a `block` guardrail takes the same two paths for it: the
default refuses with the same `502`, and `on_unavailable: monitor` serves the
request and records the check as `inconclusive`.

A mandated entry whose endpoint fails its safety check (including a host that
has stopped resolving) counts as unevaluable too, and takes the same two paths.
A `url` sent in the request body that fails the check is a `400` naming what
was wrong with it.

`on_unavailable` sets this per entry:

| `on_unavailable` | A `block` entry that cannot be evaluated |
| --- | --- |
| `block` (default) | Refuses the request with `502`, or `424` when no service is configured for it. |
| `monitor` | Serves the request and records the check as `unavailable`, or `inconclusive` when the service gave no verdict, trading enforcement for availability. |

A mandate that names a [definition](#definition-fields) is checked inside the
gateway, by the client Otari built from it. When the definition is not
available (paused, deleted, failed to build, or not yet picked up by this
worker), the check cannot be evaluated and takes the same two paths. The caller
is told the profile only; details stay in the gateway's log.

## How the layers compose

Three layers can name a guardrail: the caller's request, the caller's
organization, and an operator's [routing policy](routing.md). They are merged by
profile. Each layer can add or tighten a check but never weaken one: `block` beats
`monitor`, for both `mode` and `on_unavailable`, and what a guardrail reads (`on`)
is the union of the layers. A caller who sends `monitor` for
a profile their organization mandates in `block` mode still gets `block`. An
organization mandate adds to what callers send; it never replaces it.

Where two layers name the same profile, the outer layer decides which check runs
and with what arguments, so a caller cannot redirect a mandated check. The
routing policy is outermost.

## Mandate fields

Entries under `/api/v1/organizations/me/guardrails` (master key or a dashboard
session, and an organization owner or admin) carry:

| Field | Meaning |
| --- | --- |
| `profile` | The profile on the guardrails service. One entry per profile per organization. |
| `mode`, `on`, `on_unavailable` | The same three settings a request-body entry has, with the same meanings. `on` defaults to `["input"]`. |
| `url` | The base address of a service of the organization's own, on a public address, held to the same rules as a request's `url`: Otari adds `/validate` to it. Omit it to use the deployment's `guardrails_url`, which the operator sets and which may be on a private address. |
| `definition_id` | One of the organization's own guardrail definitions, for Otari to build and run the check itself. Exclusive with `url` and `credential`, which name a service instead. An explicit `null` clears it. |
| `credential` | Sent to that endpoint as `Authorization: Bearer`. Requires `url`, which must then be `https`, so the credential is never sent to the deployment URL, which may be a plain-http sidecar. Encrypted at rest, never returned. |
| `validate_kwargs` | Forwarded to the guardrails service `/validate` call. A parameter whose name looks credential-shaped (it contains `key`, `secret`, `token`, `password`, `authorization` or `credential`) is read back as `***` rather than its stored value. Sending `***` back keeps what is stored, so editing the rest of an entry does not overwrite the parameter you were never shown. |
| `enabled` | `false` stops this mandate everywhere without discarding the entry. |
| `applies_to_all_workspaces` | `true` runs it in every workspace, including any created later. |
| `workspace_ids` | The workspaces it runs in, when it does not apply to all of them. |
| `applies_to_memory` | `true` (the default) also runs it over what agents propose to [agent memory](memory.md#how-a-proposal-becomes-memory) in those workspaces, before it is kept. `false` checks requests only, for a check that misreads memory notes. |

A new workspace inherits the entries marked `applies_to_all_workspaces` and
nothing else. A workspace cannot opt out of an entry scoped to it: the scope is
the organization's to set.

## Definition fields

Entries under `/api/v1/organizations/me/guardrail-definitions` (master key or a
dashboard session, and an organization owner or admin) carry:

| Field | Meaning |
| --- | --- |
| `name` | The organization's own label. One definition per name per organization. |
| `guardrail_name` | The guardrail to build, as the catalog names it. |
| `create_kwargs` | The constructor arguments. Both halves of the form go here, and the catalog's own `secret` flag decides which of them are credentials. |
| `enabled` | `false` stops the guardrail everywhere it is mandated, in one write, without losing the arguments it took to set up. |

A read gives the plain arguments back as they were stored and each credential as
its name alone:

```json
{ "name": "prod-lakera",
  "guardrail_name": "lakera_guard",
  "create_kwargs": { "endpoint": "https://api.lakera.ai" },
  "create_secrets": { "api_key": "***" },
  "secrets_decryptable": true,
  "enabled": true,
  "build_state": "built" }
```

`create_kwargs` comes back in clear because the credentials were already taken
out of it by the catalog's `secret` flag. On a write:

- `***` keeps the credential stored under that name.
- A new value rotates it.
- A credential left out of a sent map is cleared.
- Omitting `create_kwargs` altogether leaves both columns untouched and reads
  neither. An admin can therefore still disable a definition whose credentials
  no longer decrypt (because `OTARI_SECRET_KEY` changed) and repair it by typing
  the credential again. Such a row reports `secrets_decryptable: false` and an
  empty `create_secrets` rather than failing the whole listing.

A write saves the row and then builds it on the worker that served it, so a
guardrail this deployment cannot construct is still stored, and the response
says what happened.

### Which definitions are refused

What may be defined is the catalog's answer, not a list Otari keeps. A
definition is refused when:

- its guardrail is not one this deployment can build;
- an argument is not one that guardrail declares;
- an argument is a live object rather than configuration (Bedrock's
  `boto3_session` and watsonx's `api_client`; the message names the arguments
  to use instead);
- a required argument that no environment variable can supply is missing.
  Whether the variable is *set* is never consulted, because the process that
  writes the row is not always the one that builds the guardrail.

Any argument whose value carries a scheme (an `endpoint`, `base_url` or `url`,
including inside a nested argument such as Alinia's `detection_config`) is
checked as an address at write time. It refuses plain `http`, an address with
no host, and a host that resolves to anything but a public address, loopback
included. No setting relaxes it. For `watsonx_guardian` and `openai_moderation`,
Otari also checks the address on every request and connects to the address it
checked, following no redirect.

The other guardrails that take an address open their connection in the
vendor's own library, which follows redirects, so their address must be on the
vendor's own hosts:

| Guardrail | Hosts an endpoint may name |
| --- | --- |
| `lakera_guard` | `api.lakera.ai`, `eu.api.lakera.ai`, `us.api.lakera.ai`, `ap-southeast-1.api.lakera.ai` |
| `patronus` | `api.patronus.ai` |
| `azure_content_safety`, `azure_prompt_shields` | a subdomain of `cognitiveservices.azure.com` (your resource's endpoint) |
| `alinia` | none: Alinia publishes no API host, so leave the endpoint out to use the operator's `ALINIA_ENDPOINT` |

A definition that names its own address must carry its own credentials too
(`api_key`, and for watsonx a `project_id` or `space_id`), each as non-empty
text. Otherwise the guardrail would read the operator's environment variable
and send the operator's key to that address, so such a definition is refused.
A credential sent as anything but text (`false`, `0`, a list) is refused on any
definition.

An address's host must be an IP address or plain letters, digits, dots and
hyphens, and the URL may hold no user information, backslash, whitespace or
control character, since the vendor libraries could read such a URL as a
different host.

These rules run when a definition is saved and again whenever its guardrail is
built, so a definition saved before them stops running. Its `build_state` reads
`failed` and `build_refusal` names the rule and the argument. Neither covers an endpoint supplied by an environment variable
(`ALINIA_ENDPOINT`, `CONTENT_SAFETY_ENDPOINT`, `WATSONX_URL`), which is the
operator's own.

Two guardrails the catalog lists cannot be defined this way:

- `bedrock_guardrails` needs both AWS keys; without them boto3 would fall back
  to the instance role of the host Otari runs on, the operator's identity.
- `any_llm` is refused outright: it takes no credential of its own, so it would
  call an LLM on the deployment's environment key, unmetered.

## How Otari builds a definition

Otari builds every enabled definition when it starts and keeps it ready in
memory, so a request never waits for its client to be built. A hosted-API
guardrail still waits for the vendor's verdict on every check.

- **A write rebuilds the definition** on the worker that served it, and the
  response reports the outcome. Other workers pick up changes within about
  thirty seconds.
- **A failed build is not retried** until the definition changes, since retrying
  a wrong credential cannot help. A build that only timed out is retried.
- **Errors never expose credentials.** The log names the guardrail, the
  definition's id and the error type only.
- **A missing vendor SDK fails the build.** A definition can save and then fail
  to build when an install left out an SDK the
  [catalog](#the-guardrail-catalog) needs, such as a plain `pip install` that
  skipped `any-guardrail`'s extras. The log shows the full `ImportError`, naming
  the extra to install.

## Telling whether a definition is running

A read of a definition carries `build_state`:

| Value | Meaning |
| --- | --- |
| `built` | The guardrail is constructed from the arguments you are looking at, and mandates pointing at it are being evaluated. |
| `failed` | This exact version was tried and would not build. Every mandate pointing at it is unevaluable, which with the default `on_unavailable: block` means those requests are refused. |
| `pending` | Nothing is held for this version yet. Normal right after a write on a worker that did not serve it, and normal for a few seconds if the build is still running. |
| `disabled` | `enabled` is `false`, so nothing is built on purpose. |

**It answers for the worker that served your request.** There is no
deployment-wide answer here: with several workers or replicas, a read taken
seconds after a write can say `pending` on one and `built` on the next, and both
are true. Treat a lasting `pending` as worth a second look rather than as a
failure, and `failed` as the one to act on. `built` does not prove the vendor
accepts the credential: some guardrails reach the vendor only when they check.

## Which profiles exist, and what they take

`GET /api/v1/tool-settings/guardrails/profiles` lists the profiles the deployment's
guardrails service has actually built, with the `validate_kwargs` each one
accepts. The dashboard's guardrail form is built from it.

The profiles come from the service's own `GET /profiles`, which reports each
profile's name and the `any-guardrail` class it was built from; the parameters
come from
[any-guardrail's parameter registry](https://github.com/mozilla-ai/any-guardrail),
keyed by that class. Otari keeps neither list. Only `validate` parameters
appear: a guardrail's constructor arguments are fixed by the operator's
`service.yaml` when the service boots, and `POST /validate` takes nothing else.

```json
{
  "available": true,
  "profiles": [
    {
      "profile": "prompt-injection",
      "guardrail": "injec_guard",
      "model_id": "leolee99/InjecGuard",
      "parameters_known": true,
      "parameters": []
    }
  ]
}
```

A service that is unconfigured, unreachable, or older than its `/profiles`
endpoint answers `"available": false` with a reason rather than an error, and
the dashboard falls back to naming a profile by hand. The same fallback covers
an entry that points at an endpoint of its own: only `guardrails_url` is read
here, because a URL taken from an entry would be one a caller chose.

Any API key and any signed-in user can read this list, because a profile name
is what a caller puts in a request. Each gateway worker keeps the answer for
30 seconds, so many reads cause at most one request to the guardrails service
in that time. A changed `guardrails_url` is read at once.

## The guardrail catalog

`GET /api/v1/tool-settings/guardrails/catalog` lists the guardrails Otari can
build and call without a service in front of them, with the constructor and
per-call arguments each one takes. It is the picker for a definition, as the
profiles read is the picker for a service's profile.

The catalog reaches no service, so unlike the profiles read it has no
unavailable state. Any signed-in user and any API key can read it. What it
answers is a property of the installed library, identical on every deployment
of the same build, and a parameter's environment variable is named without
saying whether it is set.

It lists only the [any-guardrail](https://github.com/mozilla-ai/any-guardrail)
guardrails that call a hosted API. A guardrail that runs model weights is left
out, even if it also offers a hosted API, because a saved definition has no way
to choose that API.

## What a guardrails service answers

The contract a service behind `guardrails_url`, or behind a mandate's or a
request's `url`, implements:

```
POST /validate  {"profile": "...", "input_text": "...", "validate_kwargs": {...}}
→ {"profile": "...", "result": {"valid": false, "explanation": null, "score": 0.997}}
```

Each of those addresses is the service's base: Otari adds `/validate` to it and
calls the result, so a base address carries no query string and no fragment.
A `url` that already ends in `/validate` is called at
`/validate/validate`, and the check reports the status the service answers
there, most often `http_404`.

`valid: false` means the input was flagged, and `valid: null` is an inconclusive
verdict. `categories` is optional and names what fired:

```
→ {"profile": "...", "result": {"valid": false, "explanation": null, "score": 0.997, "categories": ["prompt_attack"]}}
```

Each item is a name, or an object with `name` and `triggered` as any-guardrail
serializes a category, of which only those with `triggered: true` count. A
result without `categories` names nothing. Otari keeps only the names a
[refusal](#modes) may show.

`GET /profiles` is optional: it lists
`[{"name": "prompt-injection", "guardrail_name": "injec_guard"}]` so the
dashboard can offer the profile in a picker.
[Host your own guardrail](host-your-own-guardrail.md) walks through writing one.

## How long a service has to answer

Otari waits up to 30 seconds for a service behind `guardrails_url`, a mandate's
`url` or a request's `url`. The limit applies to each phase of the call
(connecting, sending the request, waiting for the answer and getting a
connection from the pool) rather than to the whole of it. Nothing retries, with
one exception: a mandate's or a request's `url` whose host resolves to more than
one address is tried at the next address when a connection cannot be opened.
Each attempt has its own 30 seconds to connect, so connecting can take longer
for such a host.

A call that runs out of time is a check that could not run, the same as a
service that is down: it takes the paths in
[When the guardrails service is unreachable](#when-the-guardrails-service-is-unreachable).
A `block` entry with `on_unavailable: block` refuses input with a `502` and an
answer with a `403`, withholds a tool result and skips a tool call. A `block`
entry with `on_unavailable: monitor`, and any `monitor` entry, lets the text
through, and the `Otari-Guardrails` entry reads `"valid": null`, `"status":
"unavailable"` and `"reason": "evaluation_failed"`. That differs from
`inconclusive`, which is a service that answered in time with `valid: null`.
The guardrail runs before the provider is called, so the wait is added to the
request. A check that timed out is asked again on the next request.

A [definition](#definition-fields) Otari runs itself has 10 seconds per check,
and its timeout takes the same paths.
