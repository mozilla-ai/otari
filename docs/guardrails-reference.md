# Inference Guardrails reference

The fields, headers, states and endpoints of inference guardrails. The
[Inference Guardrails guide](guardrails.md) covers adding a guardrail to a
request and mandating one across an organization, and explains how the layers
compose.

Contents:

- [The `guardrails` request field](#the-guardrails-request-field)
- [Modes](#modes)
- [Reading the `Otari-Guardrails` header](#reading-the-otari-guardrails-header)
- [When the guardrails service is unreachable](#when-the-guardrails-service-is-unreachable)
- [Mandate fields](#mandate-fields)
- [Definition fields](#definition-fields)
- [Telling whether a definition is running](#telling-whether-a-definition-is-running)
- [Which profiles exist, and what they take](#which-profiles-exist-and-what-they-take)
- [The guardrail catalog](#the-guardrail-catalog)
- [What a guardrails service answers](#what-a-guardrails-service-answers)

## The `guardrails` request field

A top-level list on `/api/v1/chat/completions`, `/api/v1/messages` and
`/api/v1/responses`. Each entry carries:

| Field | Meaning |
| --- | --- |
| `profile` | Required. The profile to run, as [the profiles read](#which-profiles-exist-and-what-they-take) names it. |
| `mode` | `monitor` (default) or `block`. See [Modes](#modes). |
| `on_unavailable` | `block` (default) or `monitor`. See [When the guardrails service is unreachable](#when-the-guardrails-service-is-unreachable). |
| `validate_kwargs` | Extra arguments for the check, merged on top of the profile's own. |
| `url` | A guardrails service to send this check to instead of the deployment's. A URL that fails Otari's address safety check is a `400`. |
| `on` | The directions to check. Defaults to `["input"]`; `output` is accepted but not enforced yet. |

## Modes

| Mode | Behavior |
| --- | --- |
| `monitor` (default) | Forwards to the provider and surfaces the verdict on the `Otari-Guardrails` response header. |
| `block` | Returns `403` (`guardrail_violation`) and never calls the provider when the input is flagged. |

## Reading the `Otari-Guardrails` header

A served non-streaming request carries one entry per input guardrail. A
streamed response does not carry the header yet, though a `block` guardrail
still refuses the request before any bytes are sent.

```json
[{"profile": "prompt-injection", "mode": "monitor", "valid": null, "score": null, "status": "unavailable", "reason": "not_configured"}]
```

`valid: null` alone cannot tell a check that did not run from one that ran and
gave no verdict, so `status` says which:

| `status` | Meaning |
| --- | --- |
| `evaluated` | The check ran. `valid` is `true` or `false`. |
| `inconclusive` | The check ran and the guardrail returned no verdict. |
| `unavailable` | The check could not run, and the request was served anyway. `reason` says why. |

`reason` appears only on an `unavailable` entry. `not_configured` means the
deployment has no guardrails service for the profile. `evaluation_failed`
covers everything else: the service is down, does not serve the profile, or
answered something malformed. The detail goes to the gateway's log, not to the
caller. The guardrail's own explanation stays out of the header too, because it
is free text of any length.

## When the guardrails service is unreachable

A `block` guardrail that cannot be evaluated at all (service down, no URL
configured, malformed response) **fails closed**: the request is rejected with a
`502` rather than forwarded unchecked. A `monitor` guardrail fails open, since it
was never enforcing.

A mandated entry whose endpoint fails its safety check counts as unevaluable
too, and takes the same two paths. That covers a host that has stopped
resolving, so an organization's endpoint going away is an outage of that entry
rather than a refusal of every request scoped to it. A `url` you send in the
request body is different: it is yours to fix, so a URL that fails the check is
a `400` naming what was wrong with it.

`on_unavailable` sets this per entry:

| `on_unavailable` | A `block` entry that cannot be evaluated |
| --- | --- |
| `block` (default) | Refuses the request with `502`. |
| `monitor` | Serves the request and records the check as `unavailable`, trading enforcement for availability. |

## Mandate fields

Entries under `/api/v1/organizations/me/guardrails` (master key or a dashboard
session, and an organization owner or admin) carry:

| Field | Meaning |
| --- | --- |
| `profile` | The profile on the guardrails service. One entry per profile per organization. |
| `mode`, `on_unavailable` | The same two settings a request-body entry has, with the same meanings. |
| `url` | An endpoint of the organization's own. Omit it to use the deployment's `guardrails_url`. |
| `definition_id` | One of the organization's own guardrail definitions, for Otari to build and run the check itself. Exclusive with `url` and `credential`, which name a service instead. An explicit `null` clears it. |
| `credential` | Sent to that endpoint as `Authorization: Bearer`. Requires `url`, which must then be `https`, so the credential is never sent to the deployment URL, which may be a plain-http sidecar. Encrypted at rest, never returned. |
| `validate_kwargs` | Forwarded to the guardrails service `/validate` call. A parameter whose name looks credential-shaped (it contains `key`, `secret`, `token`, `password`, `authorization` or `credential`) is read back as `***` rather than its stored value. Sending `***` back keeps what is stored, so editing the rest of an entry does not overwrite the parameter you were never shown. |
| `enabled` | `false` stops this mandate everywhere without discarding the entry. |
| `applies_to_all_workspaces` | `true` runs it in every workspace, including any created later. |
| `workspace_ids` | The workspaces it runs in, when it does not apply to all of them. |

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

`create_kwargs` comes back in clear deliberately: the credentials were already
taken out of it by flag, and a form has to round-trip an endpoint or a project
id. Sending a `***` back keeps the value stored under that name, a new value
rotates it, and a credential left out of a sent map is cleared, so editing the
endpoint of an entry does not overwrite the key you were never shown. Omitting
`create_kwargs` altogether leaves both columns untouched and reads neither,
which is what lets an admin on a deployment whose `OTARI_SECRET_KEY` has moved
still turn the guardrail off and repair it by typing the credential again. Such
a row reports `secrets_decryptable: false` and an empty `create_secrets` rather
than failing the whole listing.

A write saves the row and then builds it on the worker that served it, so a
guardrail this deployment cannot construct is still stored, and the response
says what happened.

### Which definitions are refused

What may be defined is the catalog's answer and not a list Otari keeps, so the
set the form offers and the set the store accepts cannot disagree. A definition
is refused when its guardrail is not one this deployment can build, when an
argument is not one that guardrail declares, when an argument is a live object
rather than configuration (Bedrock's `boto3_session` and watsonx's `api_client`
are the two, and the message names the arguments to use instead), and when a
required argument that no environment variable can supply is missing. Whether a
variable is *set* is never consulted: the process that writes the row is not
always the process that builds the guardrail, and the row may outlive both.

An argument whose value carries a scheme is checked as an address too, because
a later step builds the guardrail from these arguments and dials it. Seven of
the eight definable guardrails take one, spelled `endpoint`, `base_url` or
`url`, and the check reads the value rather than the argument's name, so a
spelling any-guardrail adds next is covered without an Otari release. It
refuses plain `http`, an address with no host, and a host that resolves inside
the network Otari runs in, and it looks inside a nested argument such as
Alinia's `detection_config`. The check runs at the write and not on every
request: the guardrail owns the socket it dials, so a later lookup would report
something Otari could not act on. Two things follow. An endpoint that arrives
from an environment variable (`ALINIA_ENDPOINT`, `CONTENT_SAFETY_ENDPOINT`,
`WATSONX_URL`) is never checked, the same carve-out the required-argument rule
makes, and that address is the operator's own rather than an organization
admin's. And an operator who has set `OTARI_MCP_ALLOW_PRIVATE_HOSTS=true` has
turned the private-address half of this check off as well: one flag covers
every address this gateway is asked to dial.

Two guardrails the catalog lists cannot be defined this way. `bedrock_guardrails`
needs both AWS keys, because without them boto3 falls back to the instance role
of the host Otari runs on, which is the operator's identity rather than the
organization's. And `any_llm` is refused outright: it takes no credential of its
own, so it would judge text by calling an LLM on whatever key the deployment's
environment holds, with nothing metering the call and nothing refunding it.

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
failure, and `failed` as the one to act on.

This is the field that keeps `enabled: true` honest. Without it a definition
with a mistyped credential saves, reports nothing wrong, and refuses every
request in the scope of any mandate that names it.

## Which profiles exist, and what they take

`GET /api/v1/tool-settings/guardrails/profiles` lists the profiles the deployment's
guardrails service has actually built, with the `validate_kwargs` each one
accepts. It drives the dashboard's guardrail form, so an entry is configured by
picking a profile and filling in typed fields rather than by naming a profile
from memory and hand-writing a dict.

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
[Which guardrails Otari can run itself](guardrails.md#which-guardrails-otari-can-run-itself)
explains what it leaves out.

## What a guardrails service answers

The contract a service behind `guardrails_url`, or behind a mandate's or a
request's `url`, implements:

```
POST /validate  {"profile": "...", "input_text": "...", "validate_kwargs": {...}}
→ {"profile": "...", "result": {"valid": false, "explanation": null, "score": 0.997}}
```

`valid: false` means the input was flagged, and `valid: null` is an inconclusive
verdict. `GET /profiles` is optional: it lists
`[{"name": "prompt-injection", "guardrail_name": "injec_guard"}]` so the
dashboard can offer the profile in a picker.
