# Store agent traces

How a deployment you run captures, keeps and seals [agent traces](https://docs.otari.ai/guides/observability/traces).
What a trace records, and how a workspace chooses to keep what its sessions
were about, is on that page; this one covers the settings and keys behind it.

## Keeping trace content

A trace keeps no prompt, command or file path until the deployment allows it
with `OTARI_TRACE_CONTENT_ENABLED=true` (`trace_content_enabled`). Content is
stored encrypted, under `OTARI_SECRET_KEY` (the key that encrypts stored
provider credentials) by default or under an AWS KMS key, and is not kept at all
on a deployment with neither. Each workspace then turns it on for itself, as
[Keeping what a span was about](https://docs.otari.ai/guides/observability/traces#keeping-what-a-span-was-about)
describes. What tools returned is kept only where the deployment also sets
`OTARI_TRACE_CONTENT_TOOL_OUTPUT=true`.

A span's content reads back empty once no configured key opens it, so keep a
rotated-out key in `OTARI_SECRET_KEY`, or in
`trace_content_kms_previous_key_ids`, for as long as you keep traces.

At startup, where content is on, the gateway makes one data key and opens it
again, for a session no export names. A key it cannot use, such as a KMS key
whose policy or region is wrong, is logged as an error then, rather than only as
content quietly not kept (`gateway_trace_content_unsealed`). A deployment with
no key to seal with at all does not offer workspaces content, and nor does one
whose startup check failed, until the key source next makes a key.

## How content is sealed

Each span's content is encrypted with AES-256-GCM under a data key, and the
data key is stored beside it in wrapped form: wrapped with `OTARI_SECRET_KEY`,
or made by AWS KMS. The ciphertext is bound to its workspace, its session and
its span, so a value copied onto another span, another session or another
workspace reads back empty.

Each session has data keys of its own, so a key that leaked would open one
session. A data key serves its session for up to five minutes, so the key source
is asked once every few minutes per active session and replica rather than once
per span. A busy Claude Code session, around 14,000 spans an hour, costs KMS about
12 `GenerateDataKey` calls an hour, and reading that hour back about 12 `Decrypt`
calls, then none while they are cached. A batch fetches the keys it lacks
together, so a batch spanning many sessions waits for them at most 1.5 seconds
in all.

The key source's request quota is shared by every workspace, and one export can
name many sessions, so each workspace may ask a replica for 64 new data keys at
once and one a second after that. A session that starts past that allowance is
stored without its content (`gateway_trace_content_unsealed{reason="limited"}`),
and its next spans are sealed once the allowance refills. One second is room for
about 300 sessions active at once in a workspace, on each replica.

A cached data key is dropped five minutes after it was fetched, whether or not
the gateway is busy, and every cached key is dropped when the gateway stops.
Dropping a key removes the gateway's reference to it; Python does not overwrite
the memory it occupied, which stays as it was until it is reused.

When the key source cannot be reached, refuses the gateway, or takes more than
1.5 seconds to answer, spans are still stored, without their content, and the
gateway does not ask it again for 30 seconds; reads in those 30 seconds do not
extend them. Content that cannot be opened reads back empty. Both are counted, on
`gateway_trace_content_unsealed` and `gateway_trace_content_unopened`, by reason.

Content stored by earlier versions, sealed with `OTARI_SECRET_KEY` alone, still
reads back, and ages out with the session; nothing needs re-encrypting.

## Sealing content with AWS KMS

Set `trace_content_key_backend: aws_kms` and name a symmetric customer-managed
key. The gateway needs the `kms` extra (`boto3`) and AWS credentials from the
usual chain (an ECS task role, an instance profile, environment variables).

| Setting | Purpose |
| --- | --- |
| `trace_content_key_backend` | `secret_key` (the default), `aws_kms`, or `aws_kms_only`. |
| `trace_content_kms_key_id` | The key new content is sealed under: a key id, key ARN or alias. Required by `aws_kms` and `aws_kms_only`. Set beside `secret_key`, it still opens what it sealed, so a deployment can move back. |
| `trace_content_kms_previous_key_ids` | Earlier keys whose content still opens, separated by commas. |
| `trace_content_kms_region` | The key's region. Unset uses the AWS SDK's default. |
| `trace_content_kms_endpoint_url` | A KMS endpoint other than the region's, such as a VPC endpoint. |

On `aws_kms`, content sealed with `OTARI_SECRET_KEY` (before the deployment
moved to KMS) still opens, with no KMS call and so no CloudTrail record.
`aws_kms_only` leaves `OTARI_SECRET_KEY` off the keyring: such content reads back
empty, and nothing opens without the KMS key. Choose it where you need to say
that only your KMS key opens session content.

The gateway calls `kms:GenerateDataKey`, `kms:Decrypt` and `kms:DescribeKey`,
always with the encryption context `otari:purpose=trace-content`,
`otari:workspace=<workspace id>` and `otari:session=<trace id>`, so a policy can
allow only that purpose, and require every call to name a workspace and a
session:

```json
{
  "Effect": "Allow",
  "Action": ["kms:GenerateDataKey", "kms:Decrypt"],
  "Resource": "<key arn>",
  "Condition": {
    "StringEquals": {"kms:EncryptionContext:otari:purpose": "trace-content"},
    "ForAllValues:StringEquals": {"kms:EncryptionContextKeys": ["otari:purpose", "otari:workspace", "otari:session"]},
    "Null": {"kms:EncryptionContext:otari:workspace": "false", "kms:EncryptionContext:otari:session": "false"}
  }
}
```

plus `kms:DescribeKey` on the key and on each previous key listed, which the
gateway calls every few minutes to learn which keys it may open, so an alias
pointed at a new key is followed without a restart. Every `GenerateDataKey` and `Decrypt` lands in CloudTrail with
the workspace it was for. Those record each data key fetched, not each span
opened: the gateway opens a session's content with a cached key for five
minutes without asking KMS again, so CloudTrail shows when content was opened
for a session, and the share read records show who read which session.

Before relying on a key, check it from a development account with
`uv run python scripts/check_trace_content_kms.py --profile <profile> --region <region> --account <account id>`.
It creates throwaway keys, runs otari's own sealing code against them (KMS
refusing another session or workspace, rotation, an alias moved to a new key, the
batch deadline across many sessions, break-glass calling KMS afresh), and
schedules them for deletion. It makes real AWS calls, so it never runs in CI, and
it refuses to run when the profile resolves to any other account.

To rotate:

- **KMS automatic rotation** needs nothing: KMS keeps the earlier key material
  inside the same key.
- **Replacing the key**: set `trace_content_kms_key_id` to the new key and list
  the old one in `trace_content_kms_previous_key_ids` until traces older than the
  change have aged out. Content under a key that is not listed reads back empty.
- **Moving from `OTARI_SECRET_KEY` to KMS**: switch the backend. New content is
  sealed with KMS, and content sealed before opens with `OTARI_SECRET_KEY` for as
  long as it is set.

## Upgrading a deployment that keeps content

The upgrade turns every workspace's content off. During a rolling deploy,
replicas still on the earlier version neither read that switch nor take the
locks that erasure and turning content off rely on, so where
`trace_content_enabled` is on they go on storing content in every workspace
until they stop, and the upgraded replicas show it to its owner. Either:

- set `trace_content_enabled` to `false` for the deploy, and back to `true` once
  no earlier replica runs; or
- stop every earlier replica before the first upgraded one starts.

Content an earlier replica stored after the upgrade began can be removed with
`DELETE /api/v1/workspaces/{workspace_id}/trace-content`.

## Settings

| Setting | Default | |
| --- | --- | --- |
| `trace_capture_enabled` | `true` | Record traces and serve the read API. Requires restart. |
| `trace_retention_days` | `30` | Delete a trace once it has had no activity for this long. |
| `trace_session_max_age_days` | `90` | Delete a session's trace this long after it started, even while it is active, so a reused session id starts a new trace. |
| `trace_content_enabled` | `false` | Let workspaces keep what an instrumented agent says a span was about; see [Keeping what a span was about](https://docs.otari.ai/guides/observability/traces#keeping-what-a-span-was-about). |
| `trace_content_tool_output` | `false` | Also keep what each tool returned on a routed request that keeps content, unless the workspace turned it off. |
| `trace_platform_break_glass_enabled` | `false` | Let the deployment's operator open any organization's session content, recorded with a reason and a reference. |

Retention runs whether or not traces are captured, so a deployment that stops
capturing still deletes what it captured on this schedule. Erasing a user deletes
their traces on every replica: spans another replica took in for them before the
erasure, and still holds, are never stored. To enforce that, the gateway stores
nothing taken in more than 24 hours earlier, however long an upload or request
stalled, and keeps the erased user's id for about a day, then deletes it. The id
of a user never erased is deleted by the next hourly retention run.

Traces are written off the request path, within fixed limits: a trace never
slows or fails a request, and when the writer cannot keep up it drops whole
requests' spans rather than parts of them. One OTLP export adds at most
1,000 spans in at most 100 sessions; spans past either limit, or that arrive
while the writer's queue is full, are dropped, and the export and the usage it
carries are still accepted. The `gateway_trace_spans_dropped` metric counts what
was dropped, by reason.

The cost on a trace is a snapshot of the usage rows' cost. Usage stays the
billing record.
