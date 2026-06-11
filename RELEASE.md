# Releasing

This document covers releasing Otari. The language SDKs release
separately on their own tracks; see [SDK releases](#sdk-releases) below.

## Releasing Otari

Otari is distributed as a Docker image (`docker.io/mzdotai/otari`). A
release is cut manually by publishing a GitHub Release with a semver tag; there is
no release-please on Otari.

### Steps

1. Update `CHANGELOG.md`: move the `[Unreleased]` entries into a new versioned
   section (this project follows [Keep a Changelog](https://keepachangelog.com/)).
   Open this as a normal PR and merge it to `main`.
2. Publish a GitHub Release with a semver tag (for example `v0.4.0`). Creating the
   Release creates the tag.

Two workflows react to the published Release:

- **`otari-docker.yml`** builds and pushes the multi-arch image to Docker Hub,
  tagged `{{version}}` (e.g. `0.4.0`), `{{major}}.{{minor}}` (e.g. `0.4`), and the
  short commit SHA. The release tag is baked in as the `OTARI_VERSION` build
  arg, so the running Otari reports it on `/health` and in the OpenAPI
  `info.version`.
- **`otari-sdk-codegen.yml`** regenerates each SDK's typed core, stamps the
  release version into the core, and opens a regeneration PR on each SDK repo.

### Continuous (non-release) builds

Every push to `main` that touches the service also builds and pushes a Docker
image tagged `latest` and the short SHA, with `OTARI_VERSION` set to the commit
SHA. These are not releases; only a published GitHub Release produces a
semver-tagged image.

### Prerequisites (repository secrets)

- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`, used by `otari-docker.yml` to push
  the image.
- `SDK_CODEGEN_TOKEN`, used by `otari-sdk-codegen.yml` to open regeneration PRs
  on the SDK repos.

## SDK releases

The four language SDKs (`otari-sdk-python`, `otari-sdk-ts`, `otari-sdk-go`,
`otari-sdk-rust`) are versioned independently of Otari and release on their
own tracks with [release-please](https://github.com/googleapis/release-please). A
Otari release does not release any SDK; it only opens the regeneration PRs
described above. A maintainer reviews and merges each, after which that SDK's own
release-please flow opens a release PR, and merging it cuts and publishes that
SDK's release.

For the full model (the spec-version compatibility scheme, the two-merge
lifecycle, and the per-SDK registries and secrets) see
[`docs/sdk-compatibility.md`](docs/sdk-compatibility.md) and each SDK repo's
`RELEASE.md`.
