# Gateway integration tests

Integration tests spin up the FastAPI app against a real database (Postgres via
testcontainers, or `TEST_DATABASE_URL`) and exercise routes end to end. See the
root `AGENTS.md` for how to run them (`make test-integration`).

Most files here test the gateway in isolation. A small number encode a
**cross-SDK contract**: a behavior the gateway guarantees that the four language
SDKs (Python, TS, Rust, Go) depend on, and that each SDK re-asserts in its own
integration suite so the hand-mirrored shells cannot silently diverge. This file
documents those shared contracts.

## Cross-SDK control-plane conformance

Each SDK wraps the generated `_client` core (see
[`scripts/sdk_codegen/README.md`](../../scripts/sdk_codegen/README.md)) with a
thin, hand-written shell. The shell, not the generated core, is where typed
error mapping lives, and it is hand-mirrored in four repos with no shared
compiler enforcing that they stay aligned. The control-plane surface (keys,
users, budgets, pricing, usage) must honor the same contracts as the inference
surface.

### Error-mapping assertion (issue #226)

> A control-plane call made with an invalid master key surfaces the SDK's typed
> authentication error (the same type the inference path raises on a 401), not
> the raw generated or transport error type.

Optionally broaden to a couple more statuses (for example 404 and 429) to assert
the control-plane and inference paths share one error contract end to end. With
this in each SDK's CI, any SDK whose control-plane stops mapping errors fails the
contract, so the four shells cannot silently drift apart again.

This assertion has two halves:

- **SDK side (one per SDK repo):** the shell maps the control-plane HTTP error to
  the SDK's typed error hierarchy, exactly as the inference path already does.
  Tracked and fixed per SDK in `otari-sdk-python`, `otari-sdk-ts`,
  `otari-sdk-rust`, and `otari-sdk-go`.
- **Gateway side (this repo):** the server must actually return one uniform error
  contract across both surfaces, otherwise no shell could map them the same way.
  [`test_control_plane_error_contract.py`](test_control_plane_error_contract.py)
  pins it: an invalid credential yields a 401 with a JSON `{"detail": <str>}`
  body on both the inference path and every control-plane router, so the two
  paths are mappable to a single typed authentication error.

The gateway anchor and the per-SDK tests are the same assertion viewed from the
two ends of the wire; keep them phrased the same way when either changes.
