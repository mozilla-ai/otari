# SDK release coordination and compatibility

This page describes how the language SDKs (`otari-sdk-python`, `otari-sdk-ts`,
`otari-sdk-go`, `otari-sdk-rust`) are released relative to the gateway, and how to
tell which SDK version works with which gateway.

## Release policy

The SDKs are not version-locked to the gateway. The policy:

1. **A gateway release does not directly trigger SDK releases.** The codegen
   workflow opens a regeneration PR against each SDK repo on spec change; that PR
   stays the human review gate.
2. **The trigger for regeneration is a change to `docs/public/openapi.json`,** not
   the gateway release event. Most gateway releases do not touch the spec.
3. **Each SDK is versioned independently** with its own semver, not locked to the
   gateway version. A shell-only fix (streaming, error mapping) can warrant an SDK
   release with no gateway change, and a no-op spec change should not bump every
   SDK.
4. **Compatibility is expressed through the spec version,** not matching package
   numbers. The gateway/spec version is stamped into each generated core so the
   SDK can surface it, and this page documents the compatibility matrix.
5. **Release automation lives in each SDK repo,** triggered when a regeneration or
   shell PR merges to that repo's main.

The reason releases are decoupled: the SDKs are a generated typed core plus a
hand-written shell (streaming, typed error mapping, ergonomic methods). A
regenerated core is not automatically releasable; a new endpoint may need shell
wiring, which the endpoint-coverage drift gate exists to catch. Auto-publishing on
a gateway release would either ship an incomplete SDK or block the gateway release
on shell work across four repos.

## Spec version

Each SDK reports the gateway/spec version its generated core was built from. The
gateway stamps that version into the core during codegen
(`scripts/sdk_codegen/generate.py`); on a gateway release the codegen workflow
passes the release tag, so the stamp is the real release version rather than the
`0.0.0-dev` placeholder in the committed spec.

How each SDK surfaces it:

| Language   | Accessor                          |
|------------|-----------------------------------|
| python     | `otari.__spec_version__`          |
| typescript | `SPEC_VERSION`                    |
| go         | `otari.SpecVersion`               |
| rust       | `otari::SPEC_VERSION`             |

The exact accessor each SDK exports is finalized in that SDK's shell; the value is
the gateway/spec version the core was generated from.

## Compatibility matrix

A given SDK version targets the gateway/spec version stamped into its core. An SDK
works against any gateway whose spec version is greater than or equal to the
version the SDK was generated from, as long as the gateway has not removed an
endpoint or field the SDK relies on. New gateway endpoints are additive: an older
SDK simply does not surface them.

The matrix below maps each gateway/spec version to the minimum SDK version that
targets it. It is updated as the gateway and SDKs cut releases; until the first
tagged releases land, every component reports the `0.0.0-dev` placeholder.

| Gateway / spec version | otari-sdk-python | otari-sdk-ts | otari-sdk-go | otari-sdk-rust |
|------------------------|------------------|--------------|--------------|----------------|
| `0.0.0-dev` (unreleased) | unreleased     | unreleased   | unreleased   | unreleased     |

To read the matrix: find the gateway/spec version you run (its value is in the
spec's `info.version` and is reported on `/health`), then use at least the SDK
version in that row for your language.

## Related

- [`scripts/sdk_codegen/README.md`](https://github.com/mozilla-ai/otari/blob/main/scripts/sdk_codegen/README.md)
  for how the core is generated and where the spec version is stamped.
- The codegen workflow (`.github/workflows/gateway-sdk-codegen.yml`) and the
  staleness alert (`.github/workflows/sdk-regen-staleness.yml`).
