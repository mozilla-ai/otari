# Store agent traces

How a deployment you run captures, keeps and seals [agent traces](https://docs.otari.ai/guides/observability/traces).
What a trace records, and how a workspace chooses to keep what its sessions
were about, is on that page; this one covers the settings and keys behind it.

## Keeping trace content

A trace keeps no prompt, command or file path unless the deployment allows it:

1. Set `OTARI_TRACE_CONTENT_ENABLED=true` (`trace_content_enabled`).
2. Make sure a sealing key exists: `OTARI_SECRET_KEY` (the default) or an
   [AWS KMS key](#sealing-content-with-aws-kms). With neither, no content is
   kept.
3. To also keep what tools returned, set `OTARI_TRACE_CONTENT_TOOL_OUTPUT=true`.

Each workspace then turns content on for itself; see
[Keeping what a span was about](https://docs.otari.ai/guides/observability/traces#keeping-what-a-span-was-about).

Content reads back empty once no configured key opens it. Keep a rotated-out
key in `OTARI_SECRET_KEY`, or in `trace_content_kms_previous_key_ids`, for as
long as you keep traces.

At startup, with content on, the gateway makes and opens one test data key. A
key it cannot use, such as a KMS key with the wrong policy or region, is logged
as an error then. Until the key source works, workspaces are not offered
content.

## How content is sealed

- Each span's content is encrypted with AES-256-GCM under a data key, stored
  beside it wrapped with `OTARI_SECRET_KEY` or made by AWS KMS.
- The ciphertext is bound to its workspace, session and span; copied anywhere
  else, it reads back empty.
- Each session has its own data keys, so a leaked key opens one session. A
  data key is cached for five minutes (and dropped when the gateway stops), so
  the key source is asked every few minutes per active session and replica,
  not per span. A busy Claude Code session (about 14,000 spans an hour) costs
  about 12 KMS `GenerateDataKey` calls an hour, and about 12 `Decrypt` calls to
  read that hour back.
- Each workspace may ask a replica for 64 new data keys at once, then one a
  second. A session past that allowance is stored without content
  (`gateway_trace_content_unsealed{reason="limited"}`) until the allowance
  refills: room for about 300 sessions active at once per workspace and
  replica.
- When the key source is unreachable, refuses, or takes more than 1.5 seconds,
  spans are stored without content and the source is not asked again for 30
  seconds. A batch spanning many sessions waits at most 1.5 seconds in all.
- Content that cannot be opened reads back empty. Both cases are counted by
  reason on `gateway_trace_content_unsealed` and
  `gateway_trace_content_unopened`.

Content sealed by earlier versions with `OTARI_SECRET_KEY` alone still reads
back; nothing needs re-encrypting.

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

On `aws_kms`, content sealed with `OTARI_SECRET_KEY` before the move still
opens, with no KMS call and no CloudTrail record. Use `aws_kms_only` where only
your KMS key may open session content: content sealed with `OTARI_SECRET_KEY`
then reads back empty.

The gateway calls `kms:GenerateDataKey` and `kms:Decrypt` with the encryption
context `otari:purpose=trace-content`,
`otari:workspace=<workspace id>` and `otari:session=<trace id>`. A policy can
allow only that purpose and require every call to name a workspace and session:

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

Also allow `kms:DescribeKey` on the key and on each previous key, without those
conditions, since it carries no encryption context. The gateway
calls it every few minutes, so an alias moved to a new key is followed without
a restart.

CloudTrail records each `GenerateDataKey` and `Decrypt` with its workspace.
Because data keys are cached for five minutes, it shows when a session's
content was opened, not each span read; the share read records show who read
which session.

Before relying on a key, test it from a development account:

```bash
uv run python scripts/check_trace_content_kms.py --profile <profile> --region <region> --account <account id>
```

It creates throwaway keys, runs Otari's sealing code against them (isolation,
rotation, a moved alias, the batch deadline, break-glass), and schedules them
for deletion. It makes real AWS calls and refuses to run if the profile
resolves to a different account.

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
replicas still on the earlier version ignore that switch and keep storing
content in every workspace while `trace_content_enabled` is on. Either:

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

Retention runs even when capture is off, so captured traces are still deleted
on schedule.

Erasing a user deletes their traces on every replica, including spans another
replica still holds. To enforce that, the gateway stores nothing taken in more
than 24 hours earlier, and keeps an erased user's id for about a day.

Traces are written off the request path and never slow or fail a request:

- One OTLP export adds at most 1,000 spans in at most 100 sessions.
- Spans past either limit, or arriving while the writer's queue is full, are
  dropped (whole requests at a time); the export and its usage are still
  accepted.
- `gateway_trace_spans_dropped` counts drops by reason.

A trace's cost is a snapshot of its usage rows' cost; usage stays the billing
record.
