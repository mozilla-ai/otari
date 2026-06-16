# Releasing

This document covers releasing Otari. The language SDKs release
separately on their own tracks; see [SDK releases](#sdk-releases) below.

## Releasing Otari

Otari is distributed as a Docker image (`docker.io/mzdotai/otari`). There is no
package publish and no version file to bump: the runtime version comes from the
git tag, baked in as the `OTARI_VERSION` build arg (see `src/gateway/version.py`).

`CHANGELOG.md` and the GitHub Release body are generated from Conventional
Commits by [git-cliff](https://git-cliff.org/) (config in
[`cliff.toml`](cliff.toml)), at release time, not on every PR. Generating once
per release (rather than regenerating and re-committing the file on every PR)
keeps the changelog out of day-to-day diffs, so feature PRs never collide on it
and nobody hand-curates release notes. Correctness comes from the commit
messages: the only per-PR requirement is a conventional PR title, which the
**Otari PR Title Check** enforces. See
[Changelog visibility](#changelog-visibility) for which commit types appear.

### Steps

The release runs in two halves so the changelog is reviewable before the tag:

1. **Open the release PR.** Trigger the **Otari Release (open PR)**
   (`otari-release.yml`) workflow from the Actions UI with the target version
   (for example `0.4.0`). It regenerates `CHANGELOG.md` for `v0.4.0` and opens a
   `release/v0.4.0` PR labeled `release` with the rendered notes in the body.
2. **Merge it.** Review the changelog diff and squash-merge the PR.
   **Otari Release (tag + publish)** (`otari-tag-release.yml`) then tags the
   squash commit `v0.4.0` and publishes the GitHub Release with the git-cliff
   notes. Creating the Release is what creates the tag.

For a local preview of what the next release notes will look like, run
`make changelog` (set `GITHUB_TOKEN` to resolve PR and author links).

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

### Changelog visibility

`CHANGELOG.md` and the GitHub Release body are generated from the Conventional
Commit messages on `main`. Otari squash-merges PRs (squash title = PR title), so
the PR title is what git-cliff parses; the **Otari PR Title Check**
(`otari-pr-title.yml`) enforces a conventional title on every PR.

User-visible prefixes appear in release notes:

- `feat:` new user-visible behavior
- `fix:` bug fixes that affect users
- `perf:` performance improvements
- `security:` security fixes
- `revert:` reverts of previously released changes

Routine maintenance is intentionally hidden: `chore:` (including `chore(deps):`
and `chore: release`), `build:`, `ci:`, `docs:`, `style:`, `refactor:`, `test:`.
Scope visibility via the prefix: `feat(web): ...` / `fix(web): ...` show up;
`refactor(web):` / `chore(web):` / `test(web):` stay out.

A non-conventional title is not silently dropped: `cliff.toml`'s catch-all parser
routes anything without a recognized prefix into a generic "Other" group, and the
release workflow fails if git-cliff still flags a parse-error skip. The PR Title
Check refuses the merge before that can happen, so "Other" should only ever catch
direct pushes to `main`.

### Prerequisites (repository secrets)

- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`, used by `otari-docker.yml` to push
  the image.
- `SDK_CODEGEN_TOKEN`, used by `otari-sdk-codegen.yml` to open regeneration PRs
  on the SDK repos.
- `RELEASE_TOKEN`, used by `otari-release.yml` and `otari-tag-release.yml`. A PAT
  or GitHub App token with `contents: write` + `pull-requests: write` on this
  repo. Required because the default `GITHUB_TOKEN` cannot start downstream
  workflows: a PR it opens would not run CI, and a Release it publishes would not
  trigger `otari-docker.yml`. Until this secret exists, only the release
  workflows are blocked; normal development is unaffected.

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
