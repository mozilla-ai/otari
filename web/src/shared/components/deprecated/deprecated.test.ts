import { readdirSync, readFileSync, statSync } from "node:fs"
import { join } from "node:path"

import { describe, expect, it } from "vitest"

/**
 * The review block on a deprecated component, made mechanical.
 *
 * `web/design/DESIGN.md` has kept a prose table of components that must not be
 * used in new code, with "adding a new one is a review block" enforced by a
 * human noticing. This directory is what turned the table into a location, and
 * this file is what turns the location into a gate.
 *
 * A gate rather than a Biome rule, deliberately. `noRestrictedImports` would be
 * shorter, but every existing call site is a violation of it, so it would need a
 * `biome-ignore` in each of the five feature files and the ignore would be the
 * thing that has to be maintained. Naming the call sites here instead keeps them
 * in one list, next to what each of them should become, and reports a *new* one
 * with that replacement rather than with a rule name.
 *
 * Converting a call site is welcome: delete its row. The count only goes down.
 */

// Resolved from the Vitest root (web/) rather than import.meta.url, which the
// jsdom environment reports as an http URL. Same reason as foundation.test.ts.
const SRC = join(process.cwd(), "src")

/** What a deprecated module's callers should reach for instead. */
const REPLACEMENT: Record<string, string> = {
  PageHeader: "layout/PageIntro",
  StatCard: "metrics/KpiStrip + metrics/KpiCell",
  RowActions: "actions/RowActionRow",
  SettingsSection: "layout/SettingsGroup",
}

/**
 * Every file outside this directory that still imports from it, and nothing
 * else. Verified against the tree rather than transcribed from the docs, whose
 * own counts had drifted: DESIGN.md placed `StatCard` on "Overview, Usage" when
 * Usage had stopped using it, and buttons.md put `RowActions` on two call sites
 * when it was on one.
 */
const KNOWN: Record<string, readonly string[]> = {
  "features/budgets/OrganizationBudgetsPage.tsx": ["PageHeader"],
  "features/tools/McpServersPage.tsx": ["PageHeader"],
  "features/workspaces/WorkspaceMembersPage.tsx": ["PageHeader"],
  "features/account/PasskeysCard.tsx": ["RowActions"],
}

function* walk(dir: string): Generator<string> {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) {
      yield* walk(full)
    } else if (/\.tsx?$/.test(entry)) {
      yield full
    }
  }
}

/** file (relative to src) -> the deprecated components it imports. */
function callers(): Map<string, string[]> {
  const found = new Map<string, string[]>()
  for (const file of walk(SRC)) {
    const name = file.slice(SRC.length + 1)
    // A deprecated component may reach its own siblings; the rule is about
    // everything else.
    if (name.startsWith(join("shared", "components", "deprecated"))) continue
    const source = readFileSync(file, "utf8")
    const hits = [
      ...source.matchAll(/components\/deprecated\/([A-Za-z]+)"/g),
    ].map((m) => m[1])
    if (hits.length) found.set(name, [...new Set(hits)].sort())
  }
  return found
}

describe("deprecated components", () => {
  it("covers the source tree", () => {
    // A guard on the guard: a wrong root would find no callers and pass.
    expect([...walk(SRC)].length).toBeGreaterThan(50)
  })

  it("are imported only where they already were", () => {
    const actual = Object.fromEntries(
      [...callers().entries()].sort(([a], [b]) => a.localeCompare(b)),
    )
    const expected = Object.fromEntries(
      Object.entries(KNOWN)
        .map(([k, v]) => [k, [...v].sort()] as const)
        .sort(([a], [b]) => a.localeCompare(b)),
    )
    // The diff names the file and the component; REPLACEMENT above says what to
    // use instead of each.
    expect(actual).toEqual(expected)
  })

  it("name a replacement for every module in the directory", () => {
    const modules = readdirSync(join(SRC, "shared", "components", "deprecated"))
      .filter((f) => /\.tsx$/.test(f) && !/\.test\.tsx$/.test(f))
      .map((f) => f.replace(/\.tsx$/, ""))
      .sort()
    // A module added here without a replacement leaves the next person with a
    // ban and no alternative, which is how a deprecation stalls.
    expect(Object.keys(REPLACEMENT).sort()).toEqual(modules)
  })

  it("records SettingsSection as reachable from nowhere", () => {
    // Not a deprecation but dead code: it has no call site at all, and it
    // shadowed `layout/SettingsGroup` (7 call sites) while diverging from this
    // tree's `export function` convention. Kept here rather than deleted
    // because removing a component is a call for the maintainers, not a side
    // effect of moving files. Delete it and this test together.
    const users = [...callers().values()].flat()
    expect(users).not.toContain("SettingsSection")
  })
})
