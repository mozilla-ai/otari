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
`image_url` block, with or without uploading first. On the Responses API a
`input_file` or `input_image` item may sit directly in `input` as well as inside
a message.

### Using the OpenAI or Anthropic SDK

The five routes (`POST`/`GET /v1/files`, `GET`/`DELETE /v1/files/{id}`,
`GET /v1/files/{id}/content`) share their paths and verbs with both vendors'
Files APIs, so either official SDK works against Otari with only its base URL
changed. The response shape follows the caller: a request carrying Anthropic's
`anthropic-version` header, which its SDK sends on every call, gets the
`FileMetadata` of Anthropic's GA Files API (`type`, `size_bytes`, `mime_type`,
`downloadable`, and RFC 3339 `created_at` and `expires_at`, with `expires_at`
`null` for a file kept indefinitely); everything else gets the OpenAI file
object (`object`, `bytes`, `purpose`, an epoch `created_at`).

Otari serves Anthropic's GA shapes only. A request whose `anthropic-beta`
header includes `files-api-2025-04-14` gets a 400, because that beta answers in
different shapes. Anthropic's Python SDK before 1.2.0, and earlier releases of
its other SDKs, send that header from `client.beta.files`, so call
`client.files` instead (see Anthropic's
[migration notes](https://platform.claude.com/docs/en/build-with-claude/files#migrate-from-files-api-2025-04-14)).

Mind the base URL: Anthropic's SDK appends `/v1` itself, so it takes
`http://localhost:8000/api`, while an OpenAI-compatible client takes
`http://localhost:8000/api/v1` (see the
[API reference](api-reference.md)).

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:8000/api", api_key="<your-api-key>")
meta = client.files.upload(file=("report.pdf", open("report.pdf", "rb"), "application/pdf"))
client.files.download(meta.id)  # Otari serves every stored file's bytes back
```

Listings are cursor-paged, and each flavor pages with its own vendor's cursor.
Both take `limit` (default 100, at most 1000).

- OpenAI: `after` names the last file of the previous page, `order` is `desc`
  by default, and the page carries `has_more`, `first_id` and `last_id`.
- Anthropic: the page is `{data, next_page}`, and `next_page` goes back as
  `page` to get the next one. To read up to 100 known files in one page, name
  them with `ids[]`, which cannot be combined with `page` or `limit`; a file you
  cannot see is left out. `after_id` and `before_id` get a 400.

Unlike Anthropic, Otari leaves an expired file out of a listing.

## Files and code execution

When a request's code runs on Otari's sandbox, because it declared the
`otari_code_execution` tool or because the
[executor](tools.md#code-execution-executor) brought a provider's own
declaration here, every uploaded file it references is also seeded into the
sandbox session's working directory, so the code the model writes can open it.
The file keeps its own filename, reduced to its last path segment; a second
upload with the same name is suffixed (`data.csv`, then `data-2.csv`), and the
marker the model is given carries the name the file actually has. An Anthropic
`container_upload` block (`{"type": "container_upload", "file_id": "..."}`) is
for the sandbox only: the model is told the file is there and never sees its
contents. A `document`, `file`, or `input_file` block with a `file_id` is both
shown to the model (extracted or passed through as usual) and seeded. Without a
sandbox in the request, a `container_upload` block is read as a document. That
is also what happens when the executor leaves a provider's declaration with the
provider: whether a file is staged follows who runs the code, decided once from
the workspace pin, the header and the deployment default.

A file the code writes into the working directory comes back as a new stored
file owned by the same user and workspace, with purpose `code_execution_output`.
Otari finds it two ways and unions them: the result block's own list of produced
files, and a listing of the workspace after each call compared with the one
before, so a backend that leaves the block's list empty (the reference container
does) still has its files collected. A seeded input the code rewrote counts as
produced.
The model sees it in the tool result as `chart.png (file_id: file-...)` and is
asked to pass that id on, and the caller downloads it with
`GET /v1/files/{id}/content`. A caller who declared Anthropic's own code tool
also gets the id in the `code_execution_tool_result` block's
`code_execution_output` entries, where Anthropic's SDK looks for it. Both directions need a sandbox backend that
implements the protocol's optional `PutFile` and `GetFile` operations, and
collecting a file the block does not name needs `ListFiles` as well; a seed the
backend refuses fails the request rather than running code over a missing input,
while an output that cannot be fetched is named without an id and the run stands.

### A file the provider's own sandbox produced

A declaration the [executor](tools.md#code-execution-executor) leaves with the
provider runs in the provider's container, and the provider names each file the
code writes by an ID of its own. When the reply arrives, Otari copies each of
those files into its store and keeps the provider's ID as the file's ID. The
copy belongs to the user and workspace the request is billed to, like an
upload. So `GET /v1/files/{id}/content` serves a chart whichever sandbox drew
it, and still serves it after the provider has discarded its container. OpenAI
discards a container 20 minutes after its last use.

The ID stays the provider's because a client that sends the turn back carries a
container reference, and a rewritten ID would name a file the provider never
issued. The copy is a stored file like any other: a listing shows its size, and
a later request can name it in a `file_id` block.

Otari copies a file before the caller sees its ID: before the reply returns,
or, on a stream, before the event that names the file is sent. The cost is
time: the reply, or the stream, waits for the download, for at most 60 seconds
in all. One reply copies at most `files_output_max_files` files and
`files_output_max_bytes` in total, the caps in the next paragraph.

Some files are not copied: one past a cap or past the time limit, one the
provider will not serve, and any file from a provider other than Anthropic or
OpenAI. Such a file's ID still appears in the reply, and Otari answers 404 for
it.

One call may store at most `files_output_max_files` files and
`files_output_max_bytes` in total (20 files and 64 MB by default, the latter also
bounded by `files_max_bytes`). What a run writes is untrusted, so a file past
either cap is named in the tool result without an id rather than stored. A
produced file is streamed from the sandbox into the store and never held whole.

> The reference `otari-sandbox-container` leaves the result block's
> file-reference list empty, so with it collection depends on `ListFiles`, which
> it implements. A backend that neither names nor lists a file does not have it
> collected.

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

- `files_enabled`, `files_backend`, `files_max_bytes`, `files_retention_hours`:
upload storage (see [Storage backends](#storage-backends)).
`files_output_max_files` and `files_output_max_bytes` bound what one
code-execution call may store from its sandbox, and what one reply may copy
from a provider's (see above). An expired file answers 404 at once, and the
background sweep (`files_sweep_interval_sec`, hourly by default, `0` to
disable) then reclaims its bytes and row along with those of deleted files.
- `file_understanding_enabled`: master switch for content normalization.
- `vision_strategy` (`describe` | `ocr` | `off`) and `vision_describe_model`:
how images are handled for text-only models. The describe model may be a local
vision model (e.g. `ollama:qwen2-vl`) to keep captioning free.
- `model_capabilities`: per-model `supports_image` / `supports_pdf` overrides.

**Upgrading from 0.6.** Two things change. `GET /api/v1/files` returns at most
100 files per page where it returned every file, so a client with more than
100 files must follow `has_more`. And the sweep now runs by default, so a
deleted file, or one past `files_retention_hours`, loses its row and bytes
within an hour where cleanup was the operator's task. Set
`files_sweep_interval_sec: 0` to keep it that way.

If recording sandbox or provider output fails during commit, Otari retains the
bytes because the metadata may already have committed. If the commit actually
failed, this can leave an orphan blob. The sweep only follows database rows and
cannot reclaim blobs without metadata; those require manual reconciliation.

### Storage backends

`files_backend` chooses where the file bytes live. Their metadata stays in the
database either way.

- `local`, the default, writes under `files_local_dir`. Use it for development
  and for a single node.
- `s3` writes to `files_s3_bucket` through boto3. Use it for AWS S3 and for
  S3-compatible stores, such as the SeaweedFS below; `files_s3_endpoint_url`
  names a store that is not AWS. Credentials come from boto3's standard chain:
  environment variables, `~/.aws/credentials`, or an IAM role.
- `fsspec` writes under `files_url` through
  [fsspec](https://filesystem-spec.readthedocs.io). Use it for the stores the
  other two do not reach, such as GCS (`gcs://bucket/prefix`), Azure
  (`abfs://container/prefix`) or SFTP (`sftp://host/path`).
  `files_storage_options` holds the keyword arguments of the protocol's
  implementation. It is an optional extra, like `s3` (`uv sync --extra fsspec`
  in a checkout); install the implementation package for the protocol as well
  (`gcsfs`, `adlfs`, `s3fs`, `paramiko`). Most read their standard credential
  environment variables on their own.

`fsspec` also accepts `s3://`, but prefer `s3` for an S3 store: the published
image includes boto3 and does not include `s3fs`.

### Object storage with Docker Compose

The `object-storage` Compose profile runs
[SeaweedFS](https://github.com/seaweedfs/seaweedfs), an S3-compatible store,
with an `otari-files` bucket. Set its credentials in `.env` beside
`docker-compose.yml`. Compose gives Otari the same pair, as
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`:

```dotenv
OTARI_S3_ACCESS_KEY=otari
OTARI_S3_SECRET_KEY=<a long random secret>
```

```bash
docker compose --profile object-storage up -d
```

Then point `config.yml` at the bucket:

```yaml
files_backend: s3
files_s3_bucket: otari-files
files_s3_endpoint_url: http://seaweedfs:8333
```

A gateway that runs on the host rather than in Compose uses
`http://localhost:8333`, with the pair exported as `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`. With an image that adds `s3fs`, `fsspec` reaches the
same bucket with `files_url: s3://otari-files` and
`files_storage_options: { endpoint_url: "http://seaweedfs:8333" }`.

Three things to know:

- The AWS variables reach every AWS client in Otari, so Bedrock also uses
  them when its provider entry sets no credentials of its own.
- SeaweedFS's filer API answers on port 8888 without credentials, so the store
  sits on an `object-storage` network that only the otari service joins. Keep
  every other service off that network.
- `OTARI_MCP_ALLOW_PRIVATE_HOSTS` and `OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS`
  let Otari call private addresses on a request's behalf, and the store is one
  of them. Leave both off in this setup.



## Dependencies

Text/office/PDF extraction uses [markitdown](https://github.com/microsoft/markitdown)
(MIT); scanned-PDF rasterization uses [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)
(Apache-2.0). Both are permissively licensed, deliberately avoiding AGPL PDF
libraries since Otari is a network service. OCR is optional; install the
`ocr` extra (`uv sync --extra ocr` in a checkout) to enable it.
