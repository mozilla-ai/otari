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
`image_url` block, with or without uploading first.

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
