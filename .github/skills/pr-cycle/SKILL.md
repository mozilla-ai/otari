---
name: pr-cycle
description: Take one or more GitHub issues to a ready PR end-to-end in mozilla-ai/otari: implement, self-review, open the PR, request review from the two bots and the team, wait for it, apply the fixes. Use when asked to "open a PR for issue X", "work through these issues", or run the implement/review/fix cycle (optionally fanned out across several issues in parallel).
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
    the PR ready (or draft, per step 7). **Never merge unless told to**, and note that `main` is
    protected as well (see [Merging](#merging-when-asked-to-merge)), so a merge needs a human
    approval on top of your instruction.

## Running checks

From the repo root:

- `make lint`: the architecture check, then Ruff. **Ruff alone is not equivalent.** A layer
  violation fails here with a clean `ruff check`.
- `make typecheck`: mypy.
- `make test`: `tests/unit` and `tests/integration`. `make test-unit` and `make test-integration`
  split it while iterating.

Integration tests need PostgreSQL: `TEST_DATABASE_URL` when set, otherwise a Testcontainers
`postgres:17`. Whichever it is, it is a *server* URL: each xdist worker creates a database of its
own on it (`postgres` becomes `postgres_gw0`, and so on) and drops it at the end of the session,
so the credentials need `CREATE DATABASE` and a `postgres` database to connect through. With no
Docker, point `TEST_DATABASE_URL` at any reachable instance; SQLite is not a fallback, because
none of that is available there. Two tests make a real outbound call and report a status mismatch
with no network egress
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

**Two bots review this repo, not one.** Requesting Copilot and reading back its comments is half
the surface: **CodeRabbit** reviews automatically, without being requested, and leaves inline
comments of its own. Poll for both before calling a review loop finished, and address CodeRabbit's
threads the same way (reply, then resolve). It tends to arrive first.

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

1. **Wait for BOTH bots to finish**, not whichever answers first. Each appears under
   `/pulls/<n>/reviews` with state `COMMENTED` once it is done, and Copilot also leaves
   `requested_reviewers`. Poll for both; neither is instant, and CodeRabbit usually arrives first.
   This is the step where a review gets called finished early: two CodeRabbit comments on a PR
   went unread because only Copilot was polled, and one of them had caught a real regression.
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
5. **Resolve your own self-review threads too.** `required_review_thread_resolution` counts
   every thread, including the ones you opened on your own diff in step 8. Explanatory comments
   that need no action still block the merge until resolved, which is a non-obvious cost of doing
   the self-review properly. Resolve them once they have been read, the same way as a reviewer's.
6. Leave the PR ready (or draft, per the original decision).

## Multi-issue orchestration

To turn a batch of issues into PRs at once, run **one worktree-isolated agent per issue** (Agent
tool, `isolation: "worktree"`), each executing the cycle above.

- **Group into waves by file independence.** Issues touching disjoint files run in parallel;
  issues sharing a file are **sequenced**, with the later ones rebasing once the earlier lands.
- **Order within a wave:** small and low-risk first, large refactors later, so the rebase surface
  stays small.
- **A stacked PR loses two protections, so stack deliberately.** Basing a PR on a topic branch
  instead of `main` buys a clean diff (only your own commits, not the parent's) and costs both of
  the things that would otherwise catch a mistake. `protect-main` applies to the default branch
  only, so the child reports `mergeStateStatus: CLEAN` and can be merged into its base with **no
  approval and no thread resolution**, silently folding two reviews into one. And
  `.coderabbit.yaml` sets `base_branches: ["main"]`, so **CodeRabbit skips the child entirely**
  (it posts a "Review skipped" comment saying so); trigger it with a `@coderabbitai review`
  comment. Neither is a reason not to stack, and both are reasons to say in the PR body that it
  is stacked and what has to happen before the parent merges.
- **Branch shape.** Squash and rebase merges are enabled and **merge commits are disabled**, so a
  branch collapses to one commit on `main` and its internal shape never lands. Still update with
  `git rebase origin/main` and `git push --force-with-lease` rather than merging `main` in: the
  PR diff and CI stay about your change.
- **Poll for reviews centrally**, not inside each agent, since an agent idling on a review burns
  its run for nothing, and poll for **both** bots (see
  [Requesting reviewers](#requesting-reviewers)). Once a wave's PRs are open, poll until each PR
  has a completed review from **both** of them, not just whichever arrived first, then route the
  fixes back to the original agents with `SendMessage` (their worktree and context are intact)
  rather than starting fresh ones.
- **Resource note.** Each agent running the integration suite boots its own Testcontainers
  Postgres. Stagger them rather than sharing one `TEST_DATABASE_URL`: two suites running at once
  against one server pick the same worker database names and drop each other's database out from
  under the run. Sharing a server is only safe if each agent gets a distinct database in the URL.

## Merging (when asked to merge)

- **Squash is the button** (merge commits are disabled). The PR title becomes the commit on
  `main` and the changelog line, so it has to be the sentence you want released.
- **Auto-merge is enabled** on the repo, so `gh pr merge --squash --auto` works while CI runs.
- **`main` is protected, by a ruleset rather than by legacy branch protection.** This matters
  because of how it looks from the API: `gh api repos/mozilla-ai/otari/branches/main/protection`
  returns **404 "Branch not protected"**, which is not evidence of no protection. It only means
  the rules are not the legacy kind. Read `gh api repos/mozilla-ai/otari/rulesets` instead, or
  `gh api repos/mozilla-ai/otari/branches/main -q .protected`, which reports `true`. An earlier
  version of this file claimed `main` was unprotected on the strength of that 404.

  The `protect-main` ruleset requires, on the default branch:
  - **one approving review**, and a bot's `COMMENTED` review does not satisfy it. Copilot and
    CodeRabbit both comment rather than approve, so a PR with both bots through and green CI
    still reports `mergeStateStatus: BLOCKED` and `reviewDecision: REVIEW_REQUIRED`.
  - **every review thread resolved** (`required_review_thread_resolution`). See the warning in
    [Handling the review](#handling-the-review) about your own self-review counting here.
  - **an extra approval for unattributed changes**
    (`require_extra_approval_for_unattributed_changes`). It keys on whether GitHub can attribute
    every commit in the PR to a user account, so a commit whose author email is not linked to one
    trips it. That matters here more than anywhere else in this file, because the subject is
    agents pushing commits: it presents as "green CI, one approval, still `BLOCKED`", which sends
    the reader back to the API. Avoid it by committing with an email linked to the account, rather
    than discovering it.
  - squash or rebase only, plus `deletion` and `non_fast_forward` rules on the branch.

  Repository admins are bypass actors, so an admin *can* merge through all of it. That makes the
  bar the one you hold yourself, unchanged: green CI, the review addressed, a human approval, and
  an explicit instruction to merge.
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
