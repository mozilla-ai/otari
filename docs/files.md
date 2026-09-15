# File uploads & document understanding

Frontier models read PDFs, office documents, and images natively. Most local /
open-source models can't; many are text-only. Otari closes that gap: you can attach a file and a text-only local model can still understand it,
because Otari extracts the file to text (and captions images) before the
model ever sees the request.

This works the same way Otari's other "frontier capabilities" do: it
inspects the request, decides what to do per attachment, and only does work the
target model actually needs.

> Standalone mode only. Hybrid mode routes to frontier providers that already
> understand documents/images, so attachments pass through untouched.



## Uploading a file

```bash
curl -X POST http://localhost:8000/api/v1/files \
  -H "Otari-Key: <your-api-key>" \
  -F purpose=user_data \
  -F file=@report.pdf
# -> {"id": "file-abc123", "object": "file", "bytes": 84213, "filename": "report.pdf", ...}
```

Then reference it from a chat request. The example below uses the OpenAI chat
format; uploaded files also work with Anthropic `document` blocks and Responses
`input_file` items:

```bash
curl http://localhost:8000/api/v1/chat/completions \
  -H "Otari-Key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama:llama3",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "Summarize the attached report." },
          { "type": "file", "file": { "file_id": "file-abc123" } }
        ]
      }
    ]
  }'
```

Before you use a local or self-hosted model here (for example
`ollama:llama3`), make sure Otari is configured for that backend under
`providers`: set its route (`api_base`, when needed) and any backend
credentials it expects (such as an API key or token). In standalone mode,
Otari also requires pricing for that model key by default: add pricing, enable
`default_pricing` if it covers the model, or set `require_pricing: false` for
an intentionally unpriced backend.

You can also inline a file as a base64 `data:` URL (`file.file_data`) or send an
`image_url` block, with or without uploading first.

### Who can see an uploaded file

A file belongs to the user who uploaded it and to the workspace the uploading API
key belongs to. Both have to match for a request to reach it, on the listing, the
metadata, the download, the delete, and the `file_id` references resolved out of
a chat message. So a user who holds keys in two workspaces reaches each
workspace's files only through that workspace's key, and anything they cannot see
answers 404 rather than 403, which is what keeps a foreign id indistinguishable
from a missing one.

The workspace comes off the key rather than a header, because a caller controls
its headers and not which key it holds. The master key is the exception: it is the
operator acting deployment-wide and sees every workspace, narrowable with
`GET /api/v1/files?workspace_id=<id>`. A master-key upload lands in the deployment's
default workspace.

## What Otari does per attachment

For each file/image block it resolves the **target model's** capabilities, then:


| Target model                                          | Documents                                  | Images                                                        |
| ----------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| **Natively capable** (e.g. Anthropic, OpenAI, GPT-4o) | forwarded unchanged                        | forwarded unchanged                                           |
| **Text-only** (most local models)                     | extracted to text (markitdown) and inlined | captioned by a vision model / OCR, or dropped with a log line |


Scanned/image-only PDFs (no extractable text) are rasterized page-by-page and
sent through the image path.

### Capability resolution

Otari must know whether the target model is natively multimodal. It uses,
in order:

1. `model_capabilities` **config override**: authoritative.
2. **any-llm provider metadata**: trusted only for hosted providers.
3. **Default: extract**, safe, since a needless extraction still yields a
  correct answer while a wrong passthrough silently drops the file.

> any-llm's capability flags are set per provider *class*, so they over-report
> for text-only models served behind OpenAI-compatible servers (vLLM, llama.cpp,
> Ollama). For those, set a `model_capabilities` override to enable native
> passthrough where the served model truly supports it.



## Configuration

See [config.example.yml](../config.example.yml) for the full list. Key knobs:

- `files_enabled`, `files_backend`, `files_local_dir`, `files_max_bytes`,
`files_retention_hours`: upload storage.
- `file_understanding_enabled`: master switch for content normalization.
- `vision_strategy` (`describe` | `ocr` | `off`) and `vision_describe_model`:
how images are handled for text-only models. The describe model may be a local
vision model (e.g. `ollama:qwen2-vl`) to keep captioning free.
- `model_capabilities`: per-model `supports_image` / `supports_pdf` overrides.



## Dependencies

Text/office/PDF extraction uses [markitdown](https://github.com/microsoft/markitdown)
(MIT); scanned-PDF rasterization uses [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)
(Apache-2.0). Both are permissively licensed, deliberately avoiding AGPL PDF
libraries since Otari is a network service. OCR is optional; install the
`ocr` extra (`pip install gateway[ocr]`) to enable it.

## Hybrid Anthropic Files (opt-in)

Hybrid gateways can forward the Anthropic GA Files API through any-llm while
keeping bytes at Anthropic. Enable `files_provider_native_enabled` only after the
control plane contributes the Files protocol and the deployed any-llm version
contains its Files interface. The initial implementation targets any-llm 1.28.
The gateway returns a fixed 502 if that interface or the control-plane protocol
is unavailable. The default remains disabled.

Use the official Anthropic SDK's GA `files` resource, not `beta.files`:

```python
from anthropic import Anthropic

client = Anthropic(
    auth_token="YOUR_OTARI_WORKSPACE_API_KEY",
    base_url="https://gateway.example/api/",
)
with open("input.csv", "rb") as source:
    uploaded = client.files.upload(file=("input.csv", source, "text/csv"))

message = client.messages.create(
    model="anthropic:YOUR_AUTHORIZED_CLAUDE_MODEL",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "container_upload", "file_id": uploaded.id},
            {"type": "text", "text": "Analyze this CSV."},
        ],
    }],
    tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
)
client.files.delete(uploaded.id)
```

The SDK appends `/v1/files`, so the direct deployment base URL ends in `/api/`.
Files requests require `anthropic-version`, which the SDK supplies. Every Files
verb rejects the legacy `files-api-2025-04-14` beta. Listings use `page`, `limit`,
and `ids[]`, with `data` and `next_page` responses. `ids[]` cannot be combined
with pagination. Legacy `after_id`, `before_id`, and `order` are rejected.

Files belong to the API key's uploader and workspace. Sharing a workspace does
not grant another user access. Listings come from those scoped bindings, never
from an account-wide Anthropic listing. Unknown, foreign, expired, and deleted
IDs are indistinguishable. Uploaded inputs are not downloadable when Anthropic
marks them `downloadable: false`; eligible generated outputs can be downloaded
with `client.files.download(file_id)`.

Every structured reference in Messages history is checked before dispatch.
The authorized model plan must include the binding's exact Anthropic account
generation. File-bearing requests have no account or provider fallback. Chat
Completions and Responses reject provider file references; use Messages.
Managed credentials still reject caller-selected container reuse.

### Limits and cleanup

| Setting | Default / requirement |
| --- | --- |
| `files_max_bytes` | 512 MiB per file |
| `files_transfer_timeout_seconds` | 300 seconds, covering receipt and upload |
| `files_idle_timeout_seconds` | 30 seconds |
| `files_rate_limit_rpm` | 60 operations per uploader/workspace, enforced in the control plane |
| `files_retention_hours` | Hybrid default 168; provider range 1–2160 hours |
| `files_max_count`, `files_max_outstanding_bytes` | Explicit positive control-plane quotas required |
| `files_temporary_capacity_bytes` | 2 GiB shared admission ceiling across local workers |
| `files_operation_timeout_seconds` | 600 seconds |
| `files_diagnostic_retention_days` | 30 days for unbound operation diagnostics |

Uploads spool to private, request-scoped temporary files. Multipart receipt
finishes before provider upload starts. The initial spool admission mechanism
requires a POSIX filesystem and coordinates workers under the same operating
system user. Use an ephemeral, quota-limited temporary volume; reservations are
reclaimed after process termination, and rolled-over file buffers are unlinked
temporary files. No durable gateway file store is used.

Deletion revokes local access before contacting Anthropic. Failed deletions stay
in a durable cleanup queue. Gateways claim fenced, five-minute leases of up to
20 files; failures back off from one minute to six hours. Replacing, removing,
or restoring a retired credential waits for required cleanup. Workspace-key
disabling and user/workspace deletion also revoke affected bindings.

An upload or generated ID is withheld until its binding commits. Uploads are
never retried after an uncertain provider outcome. A crash or lost response can
leave an inaccessible upstream orphan. Uploaded bytes receive finite provider
retention. Generated outputs have **no guaranteed provider retention** unless
Anthropic reports it; local expiry alone cannot delete an unknown upstream ID.

### Release verification

The core contract has been exercised with the merged any-llm Files implementation
at `2524c196c4c8cbeb8698a9e0b6f90d73aa659a9d` and Anthropic Python SDK 0.125.0.
The published any-llm 1.28 dependency pin and lockfile update remain a release
gate. Before hosted enablement, verify the composed hosted adapter, generated
output expiry, and the Octonous workflow without managed container reuse.
The canonical server contract is in [Hybrid mode protocol](hybrid-mode-protocol.md#provider-native-files).
