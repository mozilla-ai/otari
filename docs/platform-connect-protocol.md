# Platform connect protocol

**Contract version: 1.**

**Connect otari.ai** in a self-hosted gateway's dashboard gives the gateway an
otari.ai API key without the operator copying one. This page specifies the
contract between the two sides: the gateway, which asks, and otari.ai (the
**platform**), which issues the key once a signed-in person approves. It is a
device authorization in the shape of
[RFC 8628](https://datatracker.ietf.org/doc/html/rfc8628), reduced to what this
exchange needs.

`{platform}` is the gateway's `platform_url` setting (`OTARI_PLATFORM_URL`),
`https://api.otari.ai` by default. For what the gateway does with the key, see
[Connecting otari.ai](hosted-providers.md#connecting-otariai).

## Roles

| Role | Who | Responsibility |
|---|---|---|
| Gateway | A self-hosted Otari | Requests a code, shows it, polls for the key, stores the key |
| Platform | otari.ai | Issues codes, shows the approval page, issues one key per approved code |
| Operator | A person signed in to both | Starts the attempt in the gateway's dashboard, approves it on the platform |

## Transport

- Both endpoints are `POST` with a JSON body, and answer JSON.
- Neither takes a credential. The gateway sends no `Authorization` header, no
  API key and no cookie.
- The gateway requires `https`, except for a platform on `localhost`,
  `127.0.0.1` or `::1`, which may be `http` for development.
- The gateway follows no redirect. A `3xx` answer is a failure.
- A body field this page does not name is ignored by the reader.

## Request a code

```
POST {platform}/api/v1/connect/device
```

Request body:

| Field | Type | Meaning |
|---|---|---|
| `gateway_name` | string, at most 100 characters | What the approval page and the key call this gateway. The gateway sends the host of its `public_base_url`, or its machine's hostname where that is not set. |

`200` response:

| Field | Type | Meaning |
|---|---|---|
| `device_code` | string | The secret the gateway exchanges for the key. Never shown to a person. |
| `user_code` | string | The short code the operator checks on the approval page, such as `BCDF-GHJK`. The gateway shows it as sent and neither validates nor reformats it. |
| `verification_uri` | string | The approval page. |
| `verification_uri_complete` | string | The approval page with `user_code` filled in, such as `https://otari.ai/dashboard/?region=us#/connect?code=BCDF-GHJK`. The gateway opens it as sent, query and fragment included. |
| `expires_in` | integer, seconds | How long the codes stay valid. |
| `interval` | integer, seconds | The shortest wait between two polls. |

`verification_uri_complete` is held to the transport rule above (`https`, or
`http` on a local host), because a browser opens it; nothing else about it is
read. The platform issues an `interval` of 5 seconds and codes that live 15
minutes.

Any other status is a failure, and so is a `200` missing a field: the gateway
starts no attempt and tells the operator otari.ai is unavailable, without
repeating the platform's body. That covers `429` (too many requests, with
`Retry-After`), `503`, and the `422` a malformed body gets, such as a
`gateway_name` over 100 characters, which the gateway cuts to 100 before
sending.

## Poll for the key

```
POST {platform}/api/v1/connect/token
```

Request body:

| Field | Type | Meaning |
|---|---|---|
| `device_code` | string | The `device_code` from the code request. |

`200` response, once the operator approved:

| Field | Type | Meaning |
|---|---|---|
| `api_key` | string | The API key issued to this gateway. |
| `api_base` | string | The inference root the key is for, such as `https://api.otari.ai/api/v1`. |
| `services` | object, optional | The services besides models the operator approved, such as `{"web_search": true, "code_execution": false}`. See [The approval](#the-approval). |

**The key is returned once.** A poll after the one that delivered it answers
`expired_token`, so the gateway stores the key in the step that receives it and
does not poll again. `api_key` has the form `otk_v1_<region>_...`, which the
gateway does not parse, and `api_base` is the region's host followed by
`/api/v1`. `api_base` is held to the transport rule above.

`400` response, with one field:

| `error` | Meaning | The gateway |
|---|---|---|
| `authorization_pending` | Nobody has approved or declined yet. | Polls again after `interval`. |
| `slow_down` | The gateway polled too soon (the platform answers this to polls less than 4 seconds apart). | Adds 5 seconds to `interval` for the rest of the attempt, up to 60 seconds, then polls again. The platform does not grow the interval itself. |
| `expired_token` | The codes expired before the key was collected. An approval does not extend their lifetime. | Ends the attempt as expired. |
| `access_denied` | The operator declined. | Ends the attempt as denied. |

A `429` (with a `{"detail": ...}` body) is read as `slow_down`: the gateway adds
5 seconds to `interval` and waits at least the seconds `Retry-After` names, up
to 300, before the next poll. The attempt stays pending.

Any other answer (another status, another `error`, a body that is not JSON) is
not an outcome. The gateway keeps the attempt pending and polls again after
`interval`, until its own count of `expires_in` runs out.

## The approval

The approval page lists what the gateway will run on the platform, all billed
to the platform's credits:

- **Models**, always.
- **Web search** and **Code execution**, each a choice the operator makes, on
  by default.

Approving a service turns it on for the workspace that holds the key, where it
was off, so the key can use it. A service the operator leaves unchecked keeps
the workspace's setting as it was: a key may still reach a service the
workspace already had on. Turning a service on takes an owner or admin of that
workspace; for anyone else the platform refuses the approval and the code stays
open.

The `services` field names what was approved, each key `true` or `false`. A
gateway that reads no `services` field, or a key it does not know, treats the
service as not approved. The field says what the operator chose, not whether
the workspace can pay for or reach the service now; the platform answers that
when the service is called.

## Limits the gateway applies

A code response outside these is a failure, and the gateway starts no attempt:
`device_code` up to 4096 characters, `user_code` up to 64,
`verification_uri_complete` up to 2048, and `expires_in` above 0. The gateway
reads an `interval` below 1 as 1 and one above 60 as 60, and counts an
`expires_in` above 24 hours as 24 hours. In a key response, `api_key` and
`api_base` are each up to 1024 characters.

## Polling

- The gateway waits `interval` seconds before the first poll and between polls.
- It polls only while its dashboard is reading the attempt's status. No
  background worker polls, so an attempt nobody is watching is not polled.
- It stops at the first `200`, `expired_token` or `access_denied`, when the
  operator cancels, and `expires_in` seconds after the code request.
- Canceling is local. The platform is not told, and an unapproved code expires
  on its own.

## What the gateway keeps

- One attempt at a time per deployment, stored so it survives a restart and is
  shared by every replica.
- `device_code` encrypted with `OTARI_SECRET_KEY`, and deleted when the attempt
  ends. It is never logged and never sent to the browser.
- `api_key` encrypted, as the credential of a hosted provider named `otari-ai`.
- `api_base` with a trailing `/api/v1` or `/v1` removed, as that provider's
  endpoint: the client the gateway dispatches with takes the origin and appends
  the API root itself.

## The model listing

After the key is issued, the gateway reads the platform's model listing
(`GET {api_base}/models`) with it, as it does for any provider:

- Each `id` is offered as the platform spells it (`openai/gpt-4o-mini`), and a
  request to the gateway names it `otari-ai:openai/gpt-4o-mini`.
- A model's `pricing` block (`input_price_per_million`,
  `output_price_per_million`, and the optional `cache_read_price_per_million`,
  `cache_write_price_per_million`, `pricing_tiers` and `unit`) is stored as the
  model's rate on the gateway. A model whose block is null, or lacks a numeric
  input or output rate, is priced from its vendor's public rate as an estimate,
  the vendor being the part of the `id` before the first `/`.

## What the platform may rely on

- Nothing reaches the platform from a gateway until an operator presses
  **Connect otari.ai**. The gateway fetches no catalog, price, asset or
  telemetry endpoint from it.
- After the key is issued, the gateway calls `api_base` with it as any client of
  the inference API does. It does not call the connect endpoints again until an
  operator connects again.
- The gateway does not revoke a key. Removing the `otari-ai` provider deletes
  the gateway's copy only.

## Versioning

A change that an existing gateway or platform would misread (a renamed field, a
new required field, a new `error` value that ends an attempt) raises the
contract version and is made to this page first. Adding an optional response
field does not.
