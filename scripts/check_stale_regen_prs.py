#!/usr/bin/env python3
"""Alert when an SDK regeneration PR needs attention.

The gateway opens a regeneration PR against each SDK repo whenever the OpenAPI
spec changes (``.github/workflows/otari-sdk-codegen.yml``, branch
``sdk-codegen/client-core``). That PR is the human review gate: the generated
core can land in the spec without the matching hand-written shell wiring, so the
PR is meant to be reviewed and merged, not left open.

A PR is flagged for either of two reasons:

* **Failing checks.** Flagged immediately, at any age. A red regeneration PR
  cannot be merged, so age tells you nothing you did not already know.
* **Age.** Open longer than ``--max-age-days`` while otherwise mergeable.

Age alone used to be the only trigger, and it lagged badly: in #438 all four
PRs were red for over two weeks, and the alert reported "open 18 days" rather
than "these cannot merge, here is what broke". Checking the rollup surfaces the
actionable fact on the first run after a break.

Pure selection/rendering is unit-tested; the ``gh`` calls are thin wrappers.

Usage:
    python scripts/check_stale_regen_prs.py --max-age-days 7
    python scripts/check_stale_regen_prs.py --apply   # CI: sync the tracking issue
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

# The SDK repos the codegen workflow opens regeneration PRs against. Keep in sync
# with the matrix in .github/workflows/otari-sdk-codegen.yml.
DEFAULT_SDK_REPOS: tuple[str, ...] = (
    "mozilla-ai/otari-sdk-python",
    "mozilla-ai/otari-sdk-ts",
    "mozilla-ai/otari-sdk-go",
    "mozilla-ai/otari-sdk-rust",
)
# Head branch the codegen workflow pushes regeneration PRs from.
DEFAULT_BRANCH = "sdk-codegen/client-core"
DEFAULT_MAX_AGE_DAYS = 7
DEFAULT_TRACKING_REPO = "mozilla-ai/otari"
# Marks the single tracking issue this script owns, so reruns update it in place.
TRACKING_LABEL = "sdk-regen-stale"

# Check conclusions that mean the PR is genuinely broken. CANCELLED, SKIPPED and
# NEUTRAL are deliberately absent: a fail-fast matrix cancels its siblings when
# one leg fails, and reporting those as separate breakages just adds noise on top
# of the real failure, which is reported anyway.
FAILING_CONCLUSIONS = frozenset(
    {"FAILURE", "TIMED_OUT", "STARTUP_FAILURE", "ACTION_REQUIRED", "ERROR"}
)


def parse_iso8601(value: str) -> datetime:
    """Parse a GitHub ISO-8601 timestamp (``...Z``) into an aware UTC datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def pr_age_days(pr: dict[str, Any], now: datetime) -> float:
    """Age of ``pr`` in days from its ``createdAt`` to ``now``."""
    created = parse_iso8601(pr["createdAt"])
    return (now - created).total_seconds() / 86400.0


def check_state(pr: dict[str, Any]) -> str:
    """Roll a PR's checks up to ``failing``, ``pending``, ``passing`` or ``unknown``.

    ``unknown`` means no checks are reported at all, which is not treated as a
    problem: a repo may simply have no CI wired to that branch.
    """
    rollup = pr.get("statusCheckRollup") or []
    if not rollup:
        return "unknown"
    failing = False
    pending = False
    for entry in rollup:
        # CheckRun reports status/conclusion; the older StatusContext reports state.
        conclusion = (entry.get("conclusion") or "").upper()
        status = (entry.get("status") or "").upper()
        state = (entry.get("state") or "").upper()
        if conclusion in FAILING_CONCLUSIONS or state in FAILING_CONCLUSIONS:
            failing = True
        elif state == "PENDING" or (status and status != "COMPLETED"):
            pending = True
    if failing:
        return "failing"
    if pending:
        return "pending"
    return "passing"


def flag_reasons(pr: dict[str, Any], max_age_days: float, now: datetime) -> list[str]:
    """Why ``pr`` needs attention; empty when it does not.

    Both reasons can apply at once, and a red PR is flagged regardless of age.
    """
    reasons = []
    if check_state(pr) == "failing":
        reasons.append("checks failing")
    age = pr_age_days(pr, now)
    if age > max_age_days:
        reasons.append(f"open {age:.1f} days")
    return reasons


def select_flagged(
    prs: list[dict[str, Any]], max_age_days: float, now: datetime
) -> list[dict[str, Any]]:
    """Return the PRs needing attention, oldest-first, annotated with ``reasons``.

    A PR at the age threshold is not yet stale; it must exceed it.
    """
    flagged = [
        {**pr, "reasons": reasons}
        for pr in prs
        if (reasons := flag_reasons(pr, max_age_days, now))
    ]
    return sorted(flagged, key=lambda pr: pr["createdAt"])


def render_report(flagged: list[dict[str, Any]], max_age_days: float, now: datetime) -> str:
    """Render a markdown report of the regeneration PRs needing attention."""
    if not flagged:
        return "All SDK regeneration PRs are green and within the freshness window (or none are open)."
    red = [pr for pr in flagged if check_state(pr) == "failing"]
    lead = (
        "The following SDK regeneration PRs need attention, signalling an SDK lagging "
        f"the gateway spec. A PR is listed if its checks are failing (at any age) or it "
        f"has been open longer than {max_age_days:g} days."
    )
    if red:
        lead += (
            f" **{len(red)} of {len(flagged)} cannot merge until CI is fixed**; start there, "
            "since age is a symptom rather than the cause."
        )
    lines = [
        lead,
        "",
        "| Repo | PR | Age (days) | Checks | Why |",
        "| --- | --- | --- | --- | --- |",
    ]
    for pr in flagged:
        age = pr_age_days(pr, now)
        lines.append(
            f"| {pr['repo']} | [#{pr['number']}]({pr['url']}) | {age:.1f} | "
            f"{check_state(pr)} | {', '.join(pr['reasons'])} |"
        )
    return "\n".join(lines)


def _gh(args: list[str], token: str | None = None) -> str:
    """Run a ``gh`` command and return stdout, raising on failure.

    When ``token`` is given it overrides ``GH_TOKEN`` for that call only. The SDK
    repos and this repo can need different tokens (a cross-repo PAT reads the SDK
    PRs; this repo's own token manages the tracking issue), so each call selects
    its credential rather than relying on a single ambient one.
    """
    env = None
    if token:
        env = {**os.environ, "GH_TOKEN": token}
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True, env=env)
    return result.stdout


def fetch_open_regen_prs(repo: str, branch: str) -> list[dict[str, Any]]:
    """List the open regeneration PRs on ``repo`` for head ``branch``.

    Each returned PR is annotated with its ``repo`` for the report.
    """
    out = _gh(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            # statusCheckRollup drives the failing-checks trigger; without it a
            # red PR is only noticed once it also crosses the age threshold.
            "number,title,url,createdAt,statusCheckRollup",
        ]
    )
    prs: list[dict[str, Any]] = json.loads(out) if out.strip() else []
    for pr in prs:
        pr["repo"] = repo
    return prs


def _find_tracking_issue(tracking_repo: str, token: str | None) -> int | None:
    """Return the open tracking issue number for this script, if one exists."""
    out = _gh(
        [
            "issue",
            "list",
            "--repo",
            tracking_repo,
            "--label",
            TRACKING_LABEL,
            "--state",
            "open",
            "--json",
            "number",
        ],
        token=token,
    )
    issues: list[dict[str, Any]] = json.loads(out) if out.strip() else []
    return int(issues[0]["number"]) if issues else None


def sync_tracking_issue(
    tracking_repo: str, flagged: list[dict[str, Any]], body: str, token: str | None = None
) -> None:
    """Open/update the tracking issue while PRs are flagged; close it once none are.

    A single issue (identified by the ``sdk-regen-stale`` label) is reused across
    runs so the alert does not pile up duplicates. ``token`` authenticates the
    issue operations on ``tracking_repo``.
    """
    existing = _find_tracking_issue(tracking_repo, token)
    title = "SDK regeneration PRs need attention"
    if flagged:
        if existing is None:
            # --force upserts the label so `gh issue create --label` cannot fail on a
            # missing label in a fresh repo.
            _gh(["label", "create", TRACKING_LABEL, "--repo", tracking_repo, "--force",
                 "--color", "B60205", "--description", "An SDK regeneration PR is lagging the spec"],
                token=token)
            _gh(["issue", "create", "--repo", tracking_repo, "--title", title,
                 "--label", TRACKING_LABEL, "--body", body], token=token)
        else:
            _gh(["issue", "edit", str(existing), "--repo", tracking_repo, "--body", body], token=token)
    elif existing is not None:
        _gh(["issue", "comment", str(existing), "--repo", tracking_repo,
             "--body", "All regeneration PRs are green and merged or within the freshness window. Closing."],
            token=token)
        _gh(["issue", "close", str(existing), "--repo", tracking_repo], token=token)


def _write_step_summary(report: str) -> None:
    """Append the report to the GitHub Actions job summary, when running in CI."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Alert on SDK regeneration PRs needing attention.")
    parser.add_argument("--repos", nargs="+", default=list(DEFAULT_SDK_REPOS))
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--max-age-days", type=float, default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument("--tracking-repo", default=DEFAULT_TRACKING_REPO)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Open/update/close the tracking issue. Without it, only report.",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    all_prs: list[dict[str, Any]] = []
    for repo in args.repos:
        all_prs.extend(fetch_open_regen_prs(repo, args.branch))

    flagged = select_flagged(all_prs, args.max_age_days, now)
    report = render_report(flagged, args.max_age_days, now)
    print(report)
    _write_step_summary(report)

    if args.apply:
        # PR reads above used the ambient GH_TOKEN (the cross-repo PAT); the
        # tracking issue on this repo uses OTARI_GH_TOKEN when supplied.
        sync_tracking_issue(
            args.tracking_repo, flagged, report, token=os.environ.get("OTARI_GH_TOKEN")
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
