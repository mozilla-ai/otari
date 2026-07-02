# SDK compatibility

This page explains which SDK version works with which Otari version, and how
SDK releases relate to gateway releases.

## At a glance

- SDK versions do not match Otari versions.
- Each SDK has its own semver and release cycle.
- Compatibility is based on the Otari/OpenAPI spec version stamped into the
  SDK's generated core.
- A change to `docs/public/openapi.json` triggers SDK regeneration PRs. It does
  not publish SDKs by itself.

## Compatibility rule

Each SDK reports the Otari/spec version its generated core was built from.

| Language   | Accessor                 |
|------------|--------------------------|
| python     | `otari.__spec_version__` |
| typescript | `SPEC_VERSION`           |
| go         | `otari.SpecVersion`      |
| rust       | `otari::SPEC_VERSION`    |

- An SDK generated from spec version `X` should work with any Otari whose spec
  version is greater than or equal to `X`, as long as Otari has not removed an
  endpoint or field that SDK uses.
- New endpoints are additive. An older SDK simply does not expose them.

To find the running gateway version, check `/health/readiness`, or
`/openapi.json` (`info.version`) when docs are enabled.

## Compatibility matrix

Use the row for the Otari/spec version you run, then choose at least that SDK
version for your language.

| Otari / spec version       | otari-sdk-python | otari-sdk-ts | otari-sdk-go | otari-sdk-rust |
|----------------------------|------------------|--------------|--------------|----------------|
| `0.0.0-dev` (unreleased)   | unreleased       | unreleased   | unreleased   | unreleased     |

Update this table when tagged SDK releases land. Until then, the gateway and
SDKs report `0.0.0-dev`.

## Why releases are independent

The SDKs ship as a generated typed core plus a hand-written shell. The shell
covers behavior the generator does not provide, including:

- streaming support
- typed error mapping
- ergonomic methods and auth helpers

Because of that split, an SDK may need shell work after a spec change, and an
SDK may also ship shell-only fixes with no Otari release.

## Release flow

1. A spec change regenerates the typed core and opens PRs in the SDK repos.
2. A maintainer reviews and merges the regeneration PR.
3. `release-please` opens or updates the SDK's release PR.
4. Merging the release PR tags the release and publishes the SDK. For Go, the
   tag itself is the consumable release via `go get`.

Otari and the SDKs therefore move on separate release tracks, linked by spec
version rather than matching package numbers.

## Related

- [`scripts/sdk_codegen/README.md`](https://github.com/mozilla-ai/otari/blob/main/scripts/sdk_codegen/README.md)
  for codegen details and spec version stamping
- `.github/workflows/otari-sdk-codegen.yml`
- `.github/workflows/sdk-regen-staleness.yml`
