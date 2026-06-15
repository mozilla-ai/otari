# SDK client codegen

Generates a typed **client core** for each language SDK from the gateway's
OpenAPI spec, using [OpenAPI Generator](https://openapi-generator.tech/).

The shipped design: the generator emits a typed core covering *every* endpoint
(inference *and* control plane), and each SDK wraps that core with a thin,
hand-written shell. The cross-repo workflow runs the generator in `--mode full`.

## Architecture: generated core + hand-written shell

`generate.py --mode full` produces a typed core for the whole API surface. The
gateway spec leaves the inference surface loosely typed (chat `messages` are
untyped dicts, several inference responses use `response_model=None`), so
`enrich_spec` first injects the real typed completion schemas from `any-llm`
(which the gateway already depends on) before generation. The result: a chat
method that accepts typed messages and returns a typed `ChatCompletion`,
typed `messages` / `rerank` / `embeddings` responses, and the fully typed
control-plane endpoints (keys, users, budgets, pricing, usage).

Each SDK then **hand-writes a thin shell** over this generated core for the
things OpenAPI Generator cannot emit:

- **Streaming (SSE)** — the generator emits no server-sent-events consumption
  code, so the stream iterator is hand-written over the generated client.
- **Typed error mapping** — turning HTTP error responses into the SDK's typed
  exception hierarchy.
- **Ergonomic methods and auth modes** — friendly method names and the SDK's
  auth ergonomics on top of the raw generated calls.

The shell is **not** regenerated; only the core is. Audio and images are left
opaque in the core: they return binary/file payloads that do not map onto a
JSON response schema.

## Spec version stamping

Compatibility between an SDK and the gateway is expressed through the spec
version, not matching package numbers (each SDK is versioned independently). So
every generated core carries the gateway/spec version it was generated from, as a
language-native marker file the SDK's hand-written shell reads to surface it
(`__spec_version__` / `SPEC_VERSION` / equivalent):

| Language   | Marker file        | Symbol                            |
|------------|--------------------|-----------------------------------|
| python     | `_spec_version.py` | `__spec_version__`                |
| typescript | `specVersion.ts`   | `SPEC_VERSION`                    |
| go         | `spec_version.go`  | `SpecVersion` (core's package)    |
| rust       | `spec_version.rs`  | `SPEC_VERSION` (`spec_version` mod) |

The version is the spec's `info.version` by default; the codegen workflow passes
the gateway release version (the tag without its leading `v`) via `--spec-version`
on a release so the stamp is the real release version rather than the `0.0.0-dev`
placeholder in the committed spec. See
[`docs/sdk-compatibility.md`](../../docs/sdk-compatibility.md) for the release
policy and the spec-version to minimum-SDK-version matrix.

## Endpoint-coverage drift gate

Because the core is generated but the shell is hand-written, a new gateway
endpoint can land in the spec without a corresponding shell method. Each SDK
repo runs an endpoint-coverage drift gate against the gateway's published spec
to catch this: if the generated core gains endpoints the SDK does not yet
surface, the gate fails, signalling that the SDK needs a regeneration plus shell
wiring for the new surface.

## Local usage

Requires a JRE and the OpenAPI Generator CLI:

```bash
npm install -g @openapitools/openapi-generator-cli
# full client core for every language:
uv run python scripts/sdk_codegen/generate.py --language all --mode full --out-dir dist
# one language, plus the enriched spec for inspection:
uv run python scripts/sdk_codegen/generate.py --language python --mode full \
  --write-spec dist/otari-full.openapi.json
```

Set `OPENAPI_GENERATOR_CLI` if the CLI is not named `openapi-generator-cli` on
PATH. The generator version is pinned in `openapitools.json`
(`generator-cli.version`); that file is the only mechanism the
`@openapitools/openapi-generator-cli` wrapper honors. Bump it deliberately and
review the resulting diff.

### Control-plane-only mode

`generate.py` also retains a `--mode control-plane` (the script's default) that
filters the spec to operations tagged `keys`, `users`, `budgets`, `pricing`,
`usage` (plus the transitive closure of the schemas they reference) and
generates only that subset into the `_control_plane` paths. This was the
original scope; the published SDKs and the codegen workflow now use
`--mode full`.

## Automation

`.github/workflows/otari-sdk-codegen.yml` runs `generate.py --mode full` on
release, on spec change to `main`, or manually, and opens a PR against each SDK
repo with the regenerated client core dropped into:

| Language   | SDK repo                      | Target path (`--mode full`) |
|------------|-------------------------------|-----------------------------|
| python     | `mozilla-ai/otari-sdk-python` | `src/otari/_client`         |
| typescript | `mozilla-ai/otari-sdk-ts`     | `src/_client`               |
| go         | `mozilla-ai/otari-sdk-go`     | `otari/client`              |
| rust       | `mozilla-ai/otari-sdk-rust`   | `src/_client`               |

The matrix in the workflow mirrors `FULL_TARGETS` in `generate.py`; keep them in
sync. Target paths match what each SDK's hand-written shell imports from.

`.github/workflows/otari-sdk-codegen-check.yml` runs the same generate step on
pull requests that touch the spec or codegen tooling (but opens no PRs and needs
no token), so a spec change that breaks generation for any language is caught on
the PR rather than failing the post-merge codegen workflow. Keep its toolchain
and matrix in sync with `otari-sdk-codegen.yml`.

**Required secret:** `SDK_CODEGEN_TOKEN`, a fine-grained PAT or GitHub App token
with `Contents:write` and `Pull-requests:write` on the four SDK repos. The default
`GITHUB_TOKEN` cannot push to other repositories.

## Post-processing

Raw generator output needs small fix-ups (applied by `postprocess`/`normalize`
in `generate.py`, so no per-repo hand-edits to generated code are required):

- **rust**: the generator emits a standalone crate, but the SDK inlines it as a
  private `src/_client` module so the crate publishes to crates.io as a single
  unit (a path dependency on a separate `otari-client` crate is unpublishable;
  see mozilla-ai/otari-sdk-rust#37). `_rust_inline_module` reduces the crate to a
  module tree: `src/lib.rs` becomes `mod.rs` with the crate-root attributes
  replaced by a lint-exemption header (the module declarations are kept), the
  rest of `src/` is hoisted up, the crate scaffolding (`Cargo.toml`, docs,
  generator metadata) is dropped, and every `.rs` file is run through `rustfmt`
  (the module is now reached by the SDK's `cargo fmt --all -- --check`).

  Because the emitted `Cargo.toml` is dropped, the generated core's
  dependencies live in the SDK crate's `Cargo.toml` (`serde_with`, `url`,
  `uuid`, the `chrono` `serde` feature, the `reqwest` `multipart` feature). If a
  regeneration starts using a new crate, add that dependency to the SDK
  `Cargo.toml` by hand; the endpoint-coverage drift gate will not catch a
  missing dependency.
- **typescript** — the `mapValues` helper the models import is appended to
  `runtime.ts` (this generator version omits it).
- **go** — the generated `test/` dir is dropped (its unfilled
  `GIT_USER_ID`/`GIT_REPO_ID` import placeholders break `go vet`/`go test`) and
  the payload is run through `gofmt -w` (the generator emits un-gofmt'd Go, which
  fails the SDK repo's `gofmt -l` check).
- **python** — output is collapsed to the package directory so the workflow drops
  `otari._client` directly into the SDK.

The spec is also sanitized before generation (`sanitize_freeform_object_arrays`):
any-llm types Anthropic's `system` as `str | list[dict] | None`, whose array
variant has inline free-form-object items. In a union with two or more non-null
variants the Go generator names the array field from that inline item type and
emits an invalid identifier (`ArrayOf*mapmapOfStringAny`), which fails `gofmt`.
The sanitizer points every such array at a shared `FreeFormObject` schema so the
variant name is deterministic across all languages.
