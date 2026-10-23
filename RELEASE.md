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
   A release that contains a breaking change needs a larger version; see
   [Breaking changes](#breaking-changes).
2. **Merge it.** Review the changelog diff and squash-merge the PR.
   **Otari Release (tag + publish)** (`otari-tag-release.yml`) then tags the
   squash commit `v0.4.0`, opens a draft GitHub Release with the git-cliff
   notes, attaches the Homebrew assets (`otari-homebrew-build.yml`: the
   `otari-agent` sdist, its pinned requirements and `Formula/otari.rb`), and
   publishes it. Releases are immutable once published, so a Release's assets
   and tag cannot change afterwards, which is why the assets are attached
   while it is a draft. If a run fails before publishing, re-run it: it reuses
   the tag and the draft.

The same workflow then sends an `otari-release-published` repository dispatch
to `mozilla-ai/otari-ai`. The event carries the tag and the merge commit SHA.
It is sent after the Release is published. The image workflow runs on its own,
so the event can arrive before or after the image is pushed.

For a local preview of what the next release notes will look like, run
`make changelog` (set `GITHUB_TOKEN` to resolve PR and author links).

Five workflows react to the published Release:

- **`otari-docker.yml`** builds and pushes the multi-arch image to Docker Hub,
  tagged `{{version}}` (e.g. `0.4.0`), `{{major}}.{{minor}}` (e.g. `0.4`), and the
  short commit SHA. The release tag is baked in as the `OTARI_VERSION` build
  arg, so the running Otari reports it on `/health` and in the OpenAPI
  `info.version`.
- **`otari-sdk-codegen.yml`** regenerates each SDK's typed core, stamps the
  release version into the core, and opens a regeneration PR on each SDK repo.
- **`otari-homebrew.yml`** ships the agent-side CLI (`cli/`, distribution
  `otari-agent`) to `mozilla-ai/homebrew-tap`. The formula it ships was
  attached to the Release before publishing: `otari-homebrew-build.yml` builds
  the sdist, whose version setuptools-scm takes from the tag being released,
  and renders `Formula/otari.rb` from `packaging/homebrew/otari.rb.tmpl` and
  the lock (`scripts/homebrew_formula.py`). This workflow installs and
  `brew test`s that formula on a macOS runner, then commits it to the tap, so
  `brew install mozilla-ai/tap/otari` follows every release.
- **`otari-release-labels.yml`** labels every PR merged since the previous
  release with the release tag (for example `v0.4.0`), creating the label, so
  each PR shows the release it shipped in. A final release counts from the
  previous final release, so a PR that first shipped in a pre-release also gets
  the final version's label. Dispatch it with a tag to label an earlier release.
- **`otari-docs.yml`** builds the GitBook site from `docs/` at the release tag
  (`scripts/prepare_gitbook_site.py`) and commits it to the `gitbook-docs`
  branch, which GitBook syncs from. A pre-release does not publish. The site
  shows the highest release version only, so a release with a lower version,
  such as a patch to an older minor version, fails to publish. That failure is
  expected and needs no action. Dispatch the workflow to publish the highest
  version's docs again.
  The site includes the OpenAPI spec at `api/openapi.json` and a generated
  page per API tag. Those pages render from the spec registered in the
  GitBook organization as `otari-openapi-spec`, by the URL
  `https://raw.githubusercontent.com/mozilla-ai/otari/gitbook-docs/api/openapi.json`,
  so a renamed spec or a moved file breaks them.

One step stays manual. The Railway template (`deploy/railway/`) is a
platform object that no workflow writes. After a release that changes the
config a deploy needs (a new required variable, a new healthcheck path), or
that should move new deploys to the new image tag, edit the live template on
the mozilla-ai Railway account and update `deploy/railway/template.json` in a
PR. When `deploy/railway/listing.md` changed, paste it into the template's
README field. Then run `make railway-template-check`; it exits non-zero while
the two disagree. The **Otari Railway Template** workflow
(`otari-railway-template.yml`) runs the same check on such a PR and weekly. See
[Maintaining the template](deploy/railway/README.md#maintaining-the-template).

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
`refactor(web):` / `chore(web):` / `test(web):` stay out. A breaking change is
never hidden: see [Breaking changes](#breaking-changes).

A non-conventional title is not silently dropped: `cliff.toml`'s catch-all parser
routes anything without a recognized prefix into a generic "Other" group, and the
release workflow fails if git-cliff still flags a parse-error skip. The PR Title
Check refuses the merge before that can happen, so "Other" should only ever catch
direct pushes to `main`.

### Breaking changes

A change is breaking when a deployment must do work of its own to upgrade. For
example, a new required method on a port breaks every adapter that a deployment
wrote against the old port.

Mark a breaking change with `!` in the PR title, after the type or the scope:
`feat(api)!: remove GET /v1/usage/summary.csv`. The squash title is what
git-cliff parses. git-cliff also reads a `BREAKING CHANGE:` footer, but a footer
reaches the squash commit only from a branch commit message, so do not rely on it.

The release notes put the marker **BREAKING:** at the start of each breaking
entry. The entry stays in the group of its type. A breaking commit of a hidden
type, such as `refactor(ports)!:`, appears under "Maintenance".

A release that contains a breaking change raises the minor version while Otari is
below 1.0 (`0.6.3` becomes `0.7.0`), and the major version from 1.0 on. The
**Otari Release (open PR)** workflow compares the requested version with the last
release tag. When the version is too small, it puts a warning at the top of the
release PR body. It does not stop the release.

### Prerequisites (environment secrets)

Every publishing credential is an environment secret, never a repository or
organization secret. A repository or organization secret is readable by any
workflow on any branch, so anyone who can push a branch, or holds a token that
can, could publish with it. An environment's deployment policy limits it to
`main` and, where a Release triggers the workflow, `v*` tags:

| Environment | Secrets | Allowed refs |
| --- | --- | --- |
| `docker-hub` | none (OIDC, see below) | `main`, `v*` tags |
| `homebrew-tap` | `RELEASE_APP_PRIVATE_KEY` | `main`, `v*` tags |
| `sdk-codegen` | `CODEGEN_APP_ID`, `CODEGEN_APP_KEY` | `main`, `v*` tags |
| `release` | `RELEASE_APP_PRIVATE_KEY` | `main` |
| `platform-dispatch` | `PLATFORM_DISPATCH_APP_PRIVATE_KEY` | `main` |

An organization secret with the same name would still be readable from any
branch, so this repository must not be among the repositories an organization
secret of that name is granted to.

- No Docker Hub credential. `otari-docker.yml` logs in to the `mzdotai`
  organization through a Docker Hub OIDC connection (Docker Home, Identity &
  auth, OIDC connections) whose subject claim is
  `repo:mozilla-ai/otari:environment:docker-hub`, so only a job in that
  environment can push. The connection's ID is the `docker-hub` environment
  variable `DOCKERHUB_OIDC_CONNECTIONID`; it is not a secret. The repository's
  description on Docker Hub is edited by hand, because updating it takes a
  personal token with delete scope.
- `CODEGEN_APP_ID` and `CODEGEN_APP_KEY`, the Client ID and private key of the
  codegen GitHub App. `otari-sdk-codegen.yml` uses them to open regeneration PRs
  on the SDK repos, and `sdk-regen-staleness.yml` uses them to read those PRs.
- `PLATFORM_DISPATCH_APP_PRIVATE_KEY`, used by `otari-tag-release.yml` to send
  events to `mozilla-ai/otari-ai`. It belongs to the `otari-notification`
  GitHub App, which must be installed on `mozilla-ai/otari-ai` with
  `contents: write`, the permission a repository dispatch needs. The workflow
  identifies the App with `PLATFORM_DISPATCH_APP_CLIENT_ID` (a repository
  variable, not a secret).
- `RELEASE_APP_CLIENT_ID` (a repository variable, not a secret) and
  `RELEASE_APP_PRIVATE_KEY`, used by `otari-release.yml`,
  `otari-tag-release.yml` and `otari-homebrew.yml`, which is why the key is in
  both the `release` and `homebrew-tap` environments. They identify the
  `otari-bot` GitHub App, org-owned and installed on this repo and on
  `mozilla-ai/homebrew-tap`, holding exactly `contents: write` +
  `pull requests: write`. `otari-homebrew.yml` narrows its token to the tap's
  contents to push `Formula/otari.rb`; without the installation the build and
  macOS jobs still run and the rendered formula sits on the Release as
  `otari.rb`, and only the push to the tap fails. Each run mints its own installation token with
  `actions/create-github-app-token`, narrowed further to the permissions that
  job uses and revoked when the job ends, so the release PR is authored by
  `otari-bot[bot]` rather than by whichever maintainer owned a personal access
  token. The private key itself does not expire, so it remains the one
  long-lived credential in the release path and is worth rotating on the same
  cadence as any other: what the App buys is a narrower blast radius and an
  identity that outlives any individual, not the removal of a standing secret.

  The default `GITHUB_TOKEN` cannot be used in its place, because events it
  creates start no workflows: a PR it opened would run neither the PR title
  check nor the template check, and a Release it published would not trigger
  `otari-docker.yml`, so no image would be built. An App installation token is a
  distinct identity and does start them. Until these exist, only the release
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

## Integration package releases

`pydantic-ai-otari` and `langchain-otari` live under `integrations/` and publish to
PyPI on their own versions, independent of Otari's. To release one:

1. Bump `version` in `integrations/<package>/pyproject.toml` in an ordinary PR,
   with a conventional title (`feat(integrations): ...` or `fix(integrations): ...`),
   and merge it.
2. Run **Otari Integrations Release** (`otari-integrations-release.yml`) from the
   Actions tab on `main`, choosing the package.
3. Approve the deployment when the run waits on the package's environment.

The workflow reads the version from the package's `pyproject.toml` on `main` and
refuses one whose tag already exists. After the approval it builds and publishes
the package, then tags the commit `<package>-vX.Y.Z`; a failed or rejected
publish leaves no tag. It creates no GitHub Release, so Otari's own release stays
the repository's latest, and the release-triggered workflows (Docker image, SDK
regeneration) do not run. Otari's `vX.Y.Z` tags and these `<package>-vX.Y.Z` tags
share the repository, so everything that reads Otari's version from Git matches
`v[0-9]*` only: `cliff.toml`, `otari-release.yml`, and the `describe_command` of
the workspace members' setuptools-scm.

Publishing uses PyPI trusted publishing, so no token is stored. Each project on
PyPI trusts owner `mozilla-ai`, repository `otari`, workflow
`otari-integrations-release.yml` and an environment of its own,
`pypi-<package>` (`pypi-pydantic-ai-otari`, `pypi-langchain-otari`). PyPI refuses
two pending publishers with the same configuration, so a shared environment
cannot be registered for both. Each environment on this repository requires a
reviewer's approval and can be deployed only from `main`.
