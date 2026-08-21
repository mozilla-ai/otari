---
name: pr-cycle
description: Take one or more GitHub issues to a ready PR end-to-end in mozilla-ai/otari: implement, self-review, open the PR, request Copilot and team review, wait for it, apply the fixes. Use when asked to "open a PR for issue X", "work through these issues", or run the implement/review/fix cycle (optionally fanned out across several issues in parallel).
---

# PR Cycle (otari)

Drive an issue from code to a reviewed, ready PR without hand-holding. One issue = one branch =
one PR. For several issues, fan out (see [Multi-issue orchestration](#multi-issue-orchestration)).

Compose with the sibling skills rather than repeating them: [`review`](../review/SKILL.md) for
the self-review, [`frontend-standards`](../frontend-standards/SKILL.md) before any `web/` edit
(its [testing.md](../frontend-standards/testing.md) is the dashboard's test guidance), and
[`backend-standards`](../backend-standards/SKILL.md) before any `src/gateway/` edit (its "Before
you finish" section carries the backend test bar). [AGENTS.md](../../../AGENTS.md) owns the
repo-wide facts this skill leans on: the two runtime modes, the test notes, and the generated
artifacts.

## The cycle (one PR)

1. **Branch** off `origin/main` (`git fetch origin main` first). Name it `<type>/<slug>`
   (`fix/…`, `perf/…`, `refactor/…`, `docs/…`). Never push to `main`.
2. **Understand the issue.** Read it and the exact files and lines it names, and verify current
   signatures before editing rather than trusting the issue's snippet. Read `AGENTS.md` plus the
   scoped one for the directory you are in (`web/`, `src/gateway/`). A backend edit respects the
   layer rules `scripts/check_architecture.py` enforces; a management route is standalone-only,
   so check `register_routers()` placement.
3. **Implement.** For a refactor, keep it strictly **behavior-preserving**: no change to a
   public API, a route, a query key, or an invalidation. Correctness over speed.
4. **Tests.** Happy path and error path, next to the behavior they cover (unit for pure logic,
   integration for route or database behavior). Mock the dashboard at the `apiFetch` boundary,
   not the hooks. Never weaken coverage, and never add a global pytest rerun policy: a genuinely
   flaky test carries `@pytest.mark.flaky(reruns=...)` and a stated reason.
5. **Checks** must be green before opening (see [Running checks](#running-checks)), and
   **regenerate whatever the change made stale** (see
   [Generated artifacts](#generated-artifacts-a-pr-can-owe)).
6. **Commit.** Conventional Commits. End every message with a `Co-Authored-By:` trailer naming
   the model that wrote it (`Co-Authored-By: <your model> <noreply@anthropic.com>`). Fill in your
   own identity; do not copy a model name out of an example or another commit.
7. **Open the PR** against `mozilla-ai/otari` with `Fixes #<n>`. The **title must be a
   Conventional Commit** (`otari-pr-title.yml` gates it; accepted types are `feat`, `fix`,
   `perf`, `security`, `revert`, `chore`, `build`, `ci`, `docs`, `style`, `refactor`, `test`),
   because the repo squash-merges and git-cliff parses that title into the changelog. Keep the
   template's `## PR Type`, `## Checklist` and `## AI Usage` sections: `pr-template-check.yml`
   fails and labels the PR `missing-template` if any of the three is absent. Fill in AI Usage
   honestly, including the AI-agent checkbox. No labels are required here. No em dashes in the
   description (repo prose rule). Default to opening **ready for review**; open a **draft** only
   if the user asked to see it first (confirm which if unsure).
8. **Self-review.** Invoke the [`review`](../review/SKILL.md) skill on your own PR before anyone
   else reads it. Apply what is valid and push; skip nits that fight the repo's conventions, and
   say why.
9. **Request reviewers** (see [Requesting reviewers](#requesting-reviewers)).
10. **Wait for the review, then fix** (see [Handling the review](#handling-the-review)). Leave
    the PR ready (or draft, per step 7). **Never merge unless told to.** `main` carries no branch
    protection, so nothing mechanical stops a merge, which is exactly why this one is a rule
    rather than a gate.

## Running checks

From the repo root:

- `make lint`: the architecture check, then Ruff. **Ruff alone is not equivalent.** A layer
  violation fails here with a clean `ruff check`.
- `make typecheck`: mypy.
- `make test`: `tests/unit` and `tests/integration`. `make test-unit` and `make test-integration`
  split it while iterating.

Integration tests need PostgreSQL: `TEST_DATABASE_URL` when set, otherwise a Testcontainers
`postgres:17`. With no Docker, point `TEST_DATABASE_URL` at any reachable instance; SQLite is not
a fallback, because teardown uses `DROP TABLE ... CASCADE`. Two tests make a real outbound call
and report a status mismatch with no network egress
(`test_error_detail_leakage.py::test_provider_error_does_not_leak_details`,
`test_streaming_error_event.py::test_streaming_creation_error_returns_http_error`): that is
environment noise, not a regression, so confirm the change against the rest of the suite.

A change to the app, the migrations, or dependency resolution also owes the OSS-edition smoke
gate: `uv run --frozen --no-dev python scripts/oss_edition_smoke.py`. It defaults to a throwaway
SQLite file, so it needs no Docker.

The dashboard has its own, which `make lint` does not touch: `pnpm --dir web run lint`,
`pnpm --dir web run typecheck`, `pnpm --dir web test`. Screenshot baselines are gitignored and
that suite runs on demand, so a PR that moves a page owes no PNGs; a PR that **adds** a page owes
a screenshot entry so the page is covered when the suite becomes a gate.

## Generated artifacts a PR can owe

- A route, a schema, **or a route docstring**: run `uv run python scripts/generate_openapi.py`,
  then `make postman`, and commit both. A single CI job runs `make openapi-check` and
  `make postman-check`, so regenerating only the spec still fails.
- `web/src/client/schema.ts` and `web/src/routeTree.gen.ts` are committed and drift-checked. The
  dashboard bundle (`src/gateway/static/dashboard/`) is not committed, so a `web/src` change
  sometimes leaves a file to commit and never leaves a bundle to commit.
- Never hand-edit `CHANGELOG.md`. The release workflows regenerate it from the squashed titles.

## Requesting reviewers

- **Copilot.** `gh pr edit --add-reviewer` fails on the bot login ("Could not resolve user"), so
  use the REST endpoint:
  ```bash
  gh api -X POST repos/mozilla-ai/otari/pulls/<n>/requested_reviewers \
    -f 'reviewers[]=copilot-pull-request-reviewer[bot]'
  ```
  It reviews against `.github/instructions/*.instructions.md`, matched by `applyTo` glob, so a
  `src/gateway/` diff draws the security and performance instructions and a `web/` diff draws the
  frontend ones. The skills are not loaded for it, which is why those files restate what they do.
- **Team.** `gh pr edit <n> --add-reviewer mozilla-ai/otari-team`. CODEOWNERS auto-requests that
  team only on the open-core guardrail paths (`ARCHITECTURE.md`, `scripts/check_architecture.py`,
  `.github/CODEOWNERS`), so every other PR needs the request made explicitly.

## Handling the review

1. **Wait** for the reviewer to finish. Copilot leaves `requested_reviewers` and appears under
   `/pulls/<n>/reviews` with state `COMMENTED` once it is done. Poll for that; it is not instant.
2. **Read the comments.** The summary body is in `/pulls/<n>/reviews`. **Inline comments can be
   missing from the REST `/pulls/<n>/comments` endpoint for bot reviews** even when the summary
   claims N comments, so fetch the threads over GraphQL:
   ```bash
   gh api graphql -f query='{ repository(owner:"mozilla-ai",name:"otari"){
     pullRequest(number:<n>){ reviewThreads(first:50){ nodes{
       isResolved isOutdated id
       comments(first:10){ nodes{ databaseId author{login} path line originalLine body } } } } } } }'
   ```
3. **Triage.** Apply every valid finding; skip a nit that contradicts an established convention
   here and say so. Verify against current source before "re-fixing" anything: GitHub re-anchors
   comments to HEAD, so an addressed comment can look like it came back.
4. **Push** the fixes. On each addressed thread, **reply** with one line on how it was addressed,
   then **resolve** it:
   ```bash
   # reply: databaseId comes from the query above
   gh api -X POST repos/mozilla-ai/otari/pulls/<n>/comments/<databaseId>/replies \
     -f body='Addressed in <sha>: <what changed>.'
   # resolve: GraphQL only, with the thread's node id (there is no REST endpoint)
   gh api graphql -f query='mutation{ resolveReviewThread(input:{threadId:"<threadId>"}){ thread{ isResolved } } }'
   ```
   A reply to a resolved thread does not reopen it. Keep replies terse and free of em dashes: it
   is a poor look to trip the prose rule in the same breath as answering a review.
5. Leave the PR ready (or draft, per the original decision).

## Multi-issue orchestration

To turn a batch of issues into PRs at once, run **one worktree-isolated agent per issue** (Agent
tool, `isolation: "worktree"`), each executing the cycle above.

- **Group into waves by file independence.** Issues touching disjoint files run in parallel;
  issues sharing a file are **sequenced**, with the later ones rebasing once the earlier lands.
- **Order within a wave:** small and low-risk first, large refactors later, so the rebase surface
  stays small.
- **Branch shape.** Squash and rebase merges are enabled and **merge commits are disabled**, so a
  branch collapses to one commit on `main` and its internal shape never lands. Still update with
  `git rebase origin/main` and `git push --force-with-lease` rather than merging `main` in: the
  PR diff and CI stay about your change.
- **Poll for reviews centrally**, not inside each agent, since an agent idling on Copilot burns
  its run for nothing. Once a wave's PRs are open, poll until each has a review, then route the
  fixes back to the original agents with `SendMessage` (their worktree and context are intact)
  rather than starting fresh ones.
- **Resource note.** Each agent running the integration suite boots its own Testcontainers
  Postgres, so stagger them or point them all at one `TEST_DATABASE_URL` instead of paying for a
  container per agent.

## Merging (when asked to merge)

- **Squash is the button** (merge commits are disabled). The PR title becomes the commit on
  `main` and the changelog line, so it has to be the sentence you want released.
- **Auto-merge is enabled** on the repo, so `gh pr merge --squash --auto` works while CI runs.
- **`main` is unprotected**: no required checks, no required approvals. Nothing stops a merge, so
  the bar is the one you hold yourself: green CI, the review addressed, and an explicit
  instruction to merge.
- **Branches are deleted on merge**, which makes the stacked-PR trap real. Merging parent A with
  its branch deleted **closes child B permanently**: B's base is gone and GitHub refuses to
  reopen a PR whose base branch was deleted (`422 "state cannot be changed"`), even if you
  recreate the branch. So `gh pr edit B --base main` **while A's branch still exists**, then merge
  A. If B already closed this way, the only recovery is a fresh PR from B's head branch, linking
  the old one for its review history.

## Non-negotiables

- Never push to `main`; never merge unless told; confirm before any other outward-facing action,
  including posting a review or commenting on somebody else's PR.
- Every PR: `Fixes #<n>`, a Conventional-Commit title, the three template sections, and commit
  messages carrying a `Co-Authored-By:` trailer for the model that wrote them.
- A behavior-preserving refactor changes structure, not observable behavior. That is the bar.
- `make lint` before declaring done, not `ruff check`: the architecture check runs first and is
  the half that catches a layer violation.
