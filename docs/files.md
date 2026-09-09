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
curl -X POST http://localhost:8000/v1/files \
  -H "Otari-Key: <your-api-key>" \
  -F purpose=user_data \
  -F file=@report.pdf
# -> {"id": "file-abc123", "object": "file", "bytes": 84213, "filename": "report.pdf", ...}
```

Then reference it from a chat request. The example below uses the OpenAI chat
format; uploaded files also work with Anthropic `document` blocks and Responses
`input_file` items:

```bash
curl http://localhost:8000/v1/chat/completions \
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
`image_url` block, with or without uploading first. On the Responses API a
`input_file` or `input_image` item may sit directly in `input` as well as inside
a message.

### Using the OpenAI or Anthropic SDK

The five routes (`POST`/`GET /v1/files`, `GET`/`DELETE /v1/files/{id}`,
`GET /v1/files/{id}/content`) share their paths and verbs with both vendors'
Files APIs, so either official SDK works against Otari with only its base URL
changed. The response shape follows the caller: a request carrying Anthropic's
`anthropic-version` header, which its SDK sends on every call, gets Anthropic's
`FileMetadata` (`type`, `size_bytes`, `mime_type`, `downloadable`, an RFC 3339
`created_at`); everything else gets the OpenAI file object (`object`, `bytes`,
`purpose`, an epoch `created_at`).

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:8000", api_key="<your-api-key>")
meta = client.beta.files.upload(file=("report.pdf", open("report.pdf", "rb"), "application/pdf"))
client.beta.files.download(meta.id)  # Otari serves every stored file's bytes back
```

Listings are cursor-paged: `limit` (default 100, at most 1000), `after`
(OpenAI) or `after_id` (Anthropic) naming the last file of the previous page,
`order` (`desc` by default), and `has_more`, `first_id`, `last_id` on the page.

## Files and code execution

When a request declares the `otari_code_execution` tool, every uploaded file it
references is also seeded into the sandbox session's working directory under its
own filename, so the code the model writes can open it. An Anthropic
`container_upload` block (`{"type": "container_upload", "file_id": "..."}`) is
for the sandbox only: the model is told the file is there and never sees its
contents. A `document`, `file`, or `input_file` block with a `file_id` is both
shown to the model (extracted or passed through as usual) and seeded. Without a
sandbox in the request, a `container_upload` block is read as a document.

A file the code writes into the working directory comes back as a new stored
file owned by the same user and workspace, with purpose `code_execution_output`.
The model sees it in the tool result as `chart.png (file_id: file-...)` and is
asked to pass that id on, and the caller downloads it with
`GET /v1/files/{id}/content`. Both directions need a sandbox backend that
implements the protocol's optional `PutFile` and `GetFile` operations; a seed the
backend refuses fails the request rather than running code over a missing input,
while an output that cannot be fetched is named without an id and the run stands.

> The reference `otari-sandbox-container` implements the file operations but
> does not yet populate the result block's file-reference list, so with it
> inputs are seeded and outputs are not collected until that lands.

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
`GET /v1/files?workspace_id=<id>`. A master-key upload lands in the deployment's
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
`files_retention_hours`: upload storage. An expired file answers 404 at once,
and the background sweep (`files_sweep_interval_sec`, hourly by default, `0` to
disable) then reclaims its bytes and row along with those of deleted files.
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
