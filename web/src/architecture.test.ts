import { execFileSync } from "node:child_process"
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs"
import { join } from "node:path"

import { afterEach, describe, expect, it } from "vitest"

// Resolved from the Vitest root (web/) rather than import.meta.url, which the
// jsdom environment reports as an http URL. Same reason as src/routes.test.ts.
const WEB = process.cwd()
const BIOME = join(WEB, "node_modules", "@biomejs", "biome", "bin", "biome")

// Probes are written into the real tree, because that is the only way to find out
// what the real biome.jsonc says about a path: its overrides key off `src/features/**`
// and `src/shared/**`, so a fixture parked anywhere else is answered by a different
// rule set than the one shipping. Each probe lives for the length of one assertion
// and the directory is gitignored, so a crashed run leaves nothing to commit.
const PROBE_DIR = "__boundary_probe__"
const probeRoots = new Set<string>()

interface Diagnostic {
  category: string
  message: string
}

/** Lint one throwaway module planted in `layer` and return what Biome said about it. */
function lintProbe(layer: string, source: string): Diagnostic[] {
  const dir = join(WEB, "src", layer, PROBE_DIR)
  const file = join("src", layer, PROBE_DIR, "probe.ts")
  probeRoots.add(dir)
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(WEB, file), source)

  let stdout: string
  try {
    // The JSON reporter is documented as experimental, which is survivable because
    // Biome is pinned to an exact version: a shape change arrives with a deliberate
    // bump, and lands as a failure here rather than as a boundary that stopped being
    // checked. stderr is captured rather than inherited so Biome's own summary
    // ("Some errors were emitted") does not read as a Vitest failure.
    stdout = execFileSync(
      process.execPath,
      [BIOME, "lint", "--reporter=json", file],
      {
        cwd: WEB,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
      },
    )
  } catch (error) {
    // Biome exits non-zero whenever it emitted a diagnostic, which is the case
    // every planted violation below is about. The report is still on stdout.
    stdout = (error as { stdout?: string }).stdout ?? ""
  }
  expect(stdout, "biome produced no report").not.toBe("")
  return (JSON.parse(stdout) as { diagnostics: Diagnostic[] }).diagnostics
}

function rejects(layer: string, source: string): string[] {
  return lintProbe(layer, source)
    .filter((d) => d.category === "lint/style/noRestrictedImports")
    .map((d) => d.message)
}

afterEach(() => {
  for (const root of probeRoots) {
    rmSync(root, { recursive: true, force: true })
  }
  probeRoots.clear()
})

// The layout in src/ is a dependency rule, not a filing convention: features are
// composed by the shell and never reach back into it, shared/ knows about no domain
// at all, and nothing in this repo reaches into an overlay's tree. Those three claims
// are what src/app, src/features and src/shared buy, and each one is worth precisely
// as much as its enforcement, hence: plant the violation, prove Biome rejects it.
//
// Every case here is a pair. A rule that fires on everything would pass the rejection
// halves while making the layout unusable, so each rejection is matched with the
// nearest import that must keep working.
describe("layer boundaries", () => {
  it("reject a feature importing the composition root", () => {
    const messages = rejects(
      "features",
      'import { router } from "@/app/router";\nexport const a = router;\n',
    )
    expect(messages).toHaveLength(1)
    expect(messages[0]).toMatch(/may not import the composition root/)
  })

  it("reject a feature reaching the composition root by relative path", () => {
    // The alias is the house style, but a boundary a `../../` walks around is
    // decoration. Both forms are listed in every group in biome.jsonc.
    const messages = rejects(
      "features",
      'import { router } from "../../app/router";\nexport const a = router;\n',
    )
    expect(messages).toHaveLength(1)
  })

  it("reject a feature importing a layer by its bare specifier", () => {
    // `@/app/**` matches nothing without a trailing segment, so a barrel
    // (`src/app/index.ts`) would reopen the boundary the day someone adds one.
    // Every group lists the bare form alongside the subpath form.
    expect(
      rejects(
        "features",
        'import { router } from "@/app";\nexport const a = router;\n',
      ),
    ).toHaveLength(1)
    expect(
      rejects(
        "features",
        'import { router } from "../../app";\nexport const a = router;\n',
      ),
    ).toHaveLength(1)
    expect(
      rejects(
        "shared",
        'import { UsagePage } from "@/features";\nexport const a = UsagePage;\n',
      ),
    ).toHaveLength(1)
    expect(
      rejects(
        "features",
        'import { nav } from "@/overlay";\nexport const a = nav;\n',
      ),
    ).toHaveLength(1)
  })

  it("allow a third-party specifier that merely has an app segment", () => {
    // Why the relative form is `**/../app/**` and not `**/app/**`: the broad
    // spelling also rejects a package path, which would read as a puzzling
    // failure with a message about a layer the import has nothing to do with.
    const messages = rejects(
      "features",
      'import { core } from "some-pkg/app/core";\nexport const a = core;\n',
    )
    expect(messages).toEqual([])
  })

  it("allow a feature importing shared code, the client, and another feature", () => {
    // Feature-to-feature is deliberate, not an oversight: the keys page picks models,
    // the budgets page picks users, and routing does both. What the layout forbids is
    // a feature depending on the shell that mounts it.
    const messages = rejects(
      "features",
      [
        'import { formatPct } from "@/shared/helpers/format";',
        'import type { User } from "@/client";',
        'import { UserComboBox } from "@/features/users/UserComboBox";',
        "export const a = [formatPct, UserComboBox] as const;",
        "export type B = User;",
      ].join("\n"),
    )
    expect(messages).toEqual([])
  })

  it("reject shared code importing a feature or the composition root", () => {
    const messages = rejects(
      "shared",
      [
        'import { UsagePage } from "@/features/usage/UsagePage";',
        'import { router } from "@/app/router";',
        "export const a = [UsagePage, router];",
      ].join("\n"),
    )
    expect(messages).toHaveLength(2)
    for (const message of messages) {
      expect(message).toMatch(
        /may not import src\/app, src\/features, or src\/routes/,
      )
    }
  })

  it("reject shared code importing a route", () => {
    // The step-in-between case: a route file names a feature's page, so this is
    // `shared -> features` with a hop, and the feature group alone would miss it.
    const messages = rejects(
      "shared",
      'import { Route } from "@/routes/usage";\nexport const a = Route;\n',
    )
    expect(messages).toHaveLength(1)
    expect(messages[0]).toMatch(
      /may not import src\/app, src\/features, or src\/routes/,
    )
  })

  it("allow shared code importing the generated client and its own layer", () => {
    const messages = rejects(
      "shared",
      [
        'import type { UsageTotals } from "@/client";',
        'import { apiFetch } from "@/shared/api/client";',
        "export const a = apiFetch;",
        "export type B = UsageTotals;",
      ].join("\n"),
    )
    expect(messages).toEqual([])
  })

  // The query keys and the bounded pagination walk were file-private to
  // `shared/api/hooks.ts` until it was split by domain, which is what made "a page
  // cannot hand-roll an invalidation the hooks own" structural rather than a
  // convention. Splitting had to export them so the domain modules could share
  // them, so the boundary moved out one directory and this is what holds it there.
  it.each(["app", "features", "routes"])(
    "reject %s importing shared/api's query keys",
    (layer) => {
      const messages = rejects(
        layer,
        'import { USAGE } from "@/shared/api/queryKeys";\nexport const a = USAGE;\n',
      )
      expect(messages).toHaveLength(1)
      expect(messages[0]).toMatch(/may import its own query keys/)
    },
  )

  it("reject a feature reaching the query keys by relative path", () => {
    const messages = rejects(
      "features",
      'import { USAGE } from "../../shared/api/queryKeys";\nexport const a = USAGE;\n',
    )
    expect(messages).toHaveLength(1)
  })

  it("reject shared code outside api/ importing the pagination walk", () => {
    const messages = rejects(
      "shared",
      'import { fetchAllPaged } from "@/shared/api/paging";\nexport const a = fetchAllPaged;\n',
    )
    expect(messages).toHaveLength(1)
  })

  it("allow shared/api importing its own query keys and pagination walk", () => {
    // The pair for the four above, and the reason the rule is a directory rather
    // than a file: the keys sit in one module precisely so the domain modules can
    // share them, so `shared/api` itself must keep reaching both.
    const messages = rejects(
      "shared/api",
      [
        'import { USAGE } from "@/shared/api/queryKeys";',
        'import { fetchAllPaged } from "@/shared/api/paging";',
        "export const a = [USAGE, fetchAllPaged];",
      ].join("\n"),
    )
    expect(messages).toEqual([])
  })

  // src/design-system is the one layer whose rule names every sibling rather than
  // a few, because what it protects is not a direction but a property: the
  // directory has to still compile with the rest of src/ deleted. These are the
  // probes for that, and they are what makes the claim in DESIGN.md's extraction
  // contract checkable rather than aspirational.
  it("reject the design system importing any other layer", () => {
    const messages = rejects(
      "design-system",
      [
        'import { formatUsd } from "@/shared/helpers/format";',
        'import { UsagePage } from "@/features/usage/UsagePage";',
        'import { router } from "@/app/router";',
        'import { Route } from "@/routes/usage";',
        'import type { User } from "@/client";',
        "export const a = [formatUsd, UsagePage, router, Route];",
        "export type B = User;",
      ].join("\n"),
    )
    expect(messages).toHaveLength(5)
    for (const message of messages) {
      expect(message).toMatch(/may not import the rest of src/)
    }
  })

  it("reject the design system reaching another layer by relative path", () => {
    // A `../` walk out of the directory is the form this boundary would actually
    // be broken by, since a component two levels down is already writing
    // relative paths to its own siblings.
    expect(
      rejects(
        "design-system",
        'import { formatUsd } from "../shared/helpers/format";\nexport const a = formatUsd;\n',
      ),
    ).toHaveLength(1)
    expect(
      rejects(
        "design-system",
        'import { apiFetch } from "../../shared/api/client";\nexport const a = apiFetch;\n',
      ),
    ).toHaveLength(1)
  })

  it("reject the design system importing a layer by its bare specifier", () => {
    expect(
      rejects(
        "design-system",
        'import { formatUsd } from "@/shared";\nexport const a = formatUsd;\n',
      ),
    ).toHaveLength(1)
    expect(
      rejects(
        "design-system",
        'import type { User } from "@/client";\nexport type B = User;\n',
      ),
    ).toHaveLength(1)
  })

  it("allow the design system its own modules and its third-party dependencies", () => {
    // The permitted set, spelled out: React, the two component libraries, the
    // icons, the chart library, the Markdown renderer, and itself. If this list
    // has to grow, that is a decision about what the package would depend on, so
    // it belongs in a diff rather than in a component's import header.
    const messages = rejects(
      "design-system",
      [
        'import { useState } from "react";',
        'import { Button } from "@heroui/react";',
        'import { Checkbox } from "react-aria-components";',
        'import { FiCopy } from "react-icons/fi";',
        'import { LineChart } from "recharts";',
        'import ReactMarkdown from "react-markdown";',
        'import remarkGfm from "remark-gfm";',
        'import { Section } from "@/design-system/layout/Section";',
        'import { Dot } from "../indicators/Dot";',
        "export const a = [useState, Button, Checkbox, FiCopy, LineChart, ReactMarkdown, remarkGfm, Section, Dot];",
      ].join("\n"),
    )
    expect(messages).toEqual([])
  })

  it.each(["app", "features", "shared"])(
    "allow %s importing the design system",
    (layer) => {
      // The dependency is one-way, not forbidden. Every layer composes the
      // primitives; none of them is visible from inside the package.
      const messages = rejects(
        layer,
        'import { Section } from "@/design-system/layout/Section";\nexport const a = Section;\n',
      )
      expect(messages).toEqual([])
    },
  )

  it.each(["app", "features", "shared", "design-system"])(
    "reject %s importing the overlay tree",
    (layer) => {
      // src/overlay does not exist here and is not meant to: an overlay is a separate
      // build that layers its own pages on top of Otari (see ARCHITECTURE.md). The rule
      // predates the tree because the first import of it is the one that ends Otari's
      // ability to build on its own, and it would be caught in someone else's repo.
      const messages = rejects(
        layer,
        'import { nav } from "@/overlay/nav";\nexport const a = nav;\n',
      )
      expect(messages).toHaveLength(1)
      expect(messages[0]).toMatch(/may import the overlay tree/)
    },
  )

  it("allow a test harness to reach the composition root", () => {
    // src/tests/ is deliberately outside the overrides. A harness exists to mount
    // what the app mounts, so it is the one place that has to reach `@/app`, and
    // it is how a feature's test gets the real providers without importing them.
    // Everything else about it is still checked: the overlay rule applies here too.
    expect(
      rejects(
        "tests",
        'import { Provider } from "@/app/provider";\nexport const a = Provider;\n',
      ),
    ).toEqual([])
    expect(
      rejects(
        "tests",
        'import { nav } from "@/overlay/nav";\nexport const a = nav;\n',
      ),
    ).toHaveLength(1)
  })

  it("allow the composition root to import features", () => {
    const messages = rejects(
      "app",
      'import { AuthProvider } from "@/features/auth/AuthContext";\nexport const a = AuthProvider;\n',
    )
    expect(messages).toEqual([])
  })
})

// The rules above are only a boundary if something runs them on the way in.
describe("the boundary check is wired up", () => {
  it("is a package script", () => {
    const pkg = JSON.parse(readFileSync(join(WEB, "package.json"), "utf8")) as {
      scripts: Record<string, string>
    }
    // `biome check` is the linter, the formatter, and the assists in one pass, so
    // it still runs the boundary rules. `--write` is deliberately not in it: that
    // one rewrites the tree instead of failing on it, and belongs to `lint:fix`.
    expect(pkg.scripts.lint).toContain("biome check")
    expect(pkg.scripts.lint).not.toContain("--write")
  })

  it("runs in the dashboard workflow", () => {
    const workflow = join(
      WEB,
      "..",
      ".github",
      "workflows",
      "otari-dashboard.yml",
    )
    // Skipped rather than failed in a checkout without the workflows (a sparse
    // clone, a vendored copy of web/): the assertion has nothing to say there.
    if (!existsSync(workflow)) return
    const runsLint = readFileSync(workflow, "utf8").includes("pnpm run lint")
    expect(runsLint, "otari-dashboard.yml does not run pnpm run lint").toBe(
      true,
    )
  })
})

// The layers are a closed set. A new top-level directory under src/ is a fourth
// layer nobody wrote a rule for, and the rules above would have nothing to say
// about it, so it is caught here instead of in review.
describe("the layout", () => {
  it("has exactly the layers the boundary rules cover", () => {
    const dirs = readdirSync(join(WEB, "src"), { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort()
    expect(dirs).toEqual([
      "app",
      "client",
      "design-system",
      "features",
      "routes",
      "shared",
      "styles",
      "tests",
    ])
  })
})

// The extraction contract has a second half the import probes above cannot see.
// `design-system/` compiles with the rest of `src/` deleted, which is what they
// prove; whether it still *looks* like itself is a question about CSS, and
// nothing failed when that drifted, so a deleted rule reached a reviewer as a
// component rendering unstyled rather than as a red test.
//
// The file read below is the design system's own stylesheet, not the
// application's, which is what makes the folder move real: a class a primitive
// wears has to be declared inside the directory that would ship. A rule put back
// in `globals.css` fails here, which is the point.
//
// What this proves and what it does not. It catches a rule deleted or renamed out
// from under a primitive, and a size added to a dialog's union with no rule to
// match. It does not catch a *second* rule for one of these classes added to
// `globals.css`, because several legitimately live there: the ghost-border
// family names `.otari-toolbar`, `.otari-table` and `.otari-bulk-bar` beside the
// feature places it also covers, and splitting that decision across two files
// would cost more than the check is worth.
describe("the design system's stylesheet dependency", () => {
  const DESIGN_SYSTEM = join(WEB, "src", "design-system")
  const STYLESHEET = join(DESIGN_SYSTEM, "design-system.css")

  /**
   * Every `.tsx` under `design-system/` that ships, so no story and no test.
   * A `.css` beside them is not a source file either.
   */
  function sourceFiles(dir: string): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const full = join(dir, entry.name)
      if (entry.isDirectory()) return sourceFiles(full)
      if (!entry.name.endsWith(".tsx")) return []
      if (entry.name.includes(".test.") || entry.name.includes(".stories."))
        return []
      return [full]
    })
  }

  // Comments come out first, and that is the whole reason this is not a bare
  // grep: the tree cites sibling issues as `otari-ai#2110` and explains retired
  // rules by name (`.otari-markdown`), and both read as a class to a pattern that
  // does not know what a comment is.
  function code(source: string): string {
    return source
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/^[ \t]*\/\/.*$/gm, "")
  }

  const CLASS = /otari-[a-z0-9]+(?:[-_][a-z0-9]+)*(?:__[a-z0-9-]+)?/g
  // `otari-dialog--${size}`: the modifier is chosen at render, so no literal
  // appears and only the prefix can be read off the source.
  const MODIFIER = /(otari-[a-z-]+--)\$\{/g

  function worn(): { literals: Set<string>; prefixes: Set<string> } {
    const literals = new Set<string>()
    const prefixes = new Set<string>()
    for (const file of sourceFiles(DESIGN_SYSTEM)) {
      const source = code(readFileSync(file, "utf8"))
      for (const match of source.matchAll(CLASS)) literals.add(match[0])
      for (const match of source.matchAll(MODIFIER)) prefixes.add(match[1])
    }
    // A modifier prefix also matches CLASS as its own base name. The base is
    // declared in its own right and asserted below, so drop the partial.
    for (const prefix of prefixes) literals.delete(prefix.slice(0, -2))
    return { literals, prefixes }
  }

  /**
   * Both declaration forms: a selector, and a Tailwind `@utility`.
   *
   * Comments come out of the stylesheet for the same reason they come out of the
   * source, and the case is not hypothetical: `.otari-markdown` is named twice
   * in `globals.css` explaining why the rules that used to carry it are gone,
   * and reading a file raw counts a class like that as declared. A primitive
   * wearing a class that survives only in prose is what this is meant to catch.
   */
  function declared(css = readFileSync(STYLESHEET, "utf8")): Set<string> {
    const rules = css.replace(/\/\*[\s\S]*?\*\//g, "")
    return new Set([
      ...[...rules.matchAll(/\.(otari-[A-Za-z0-9_-]+)/g)].map((m) => m[1]),
      ...[...rules.matchAll(/@utility\s+(otari-[A-Za-z0-9_-]+)/g)].map(
        (m) => m[1],
      ),
    ])
  }

  it("covers the design system", () => {
    // A guard on the guard: a wrong root would read no files, find no classes
    // and pass. Same shape as deprecated.test.ts's.
    expect(sourceFiles(DESIGN_SYSTEM).length).toBeGreaterThan(50)
    expect(worn().literals.size).toBeGreaterThan(10)
    // And a guard on the file. A path typo throws on the read, but a file
    // emptied by a bad merge does not, and would read as every rule at once
    // having been deleted somewhere else.
    expect(declared().size).toBeGreaterThan(10)
  })

  it("is reached from the stylesheet the application loads", () => {
    // The rules land only because `globals.css` imports them, and CSS drops an
    // `@import` that follows any other rule, silently, so the line's position is
    // as load-bearing as the line itself.
    const IMPORT = '@import "../design-system/design-system.css";'
    const globals = readFileSync(
      join(WEB, "src", "styles", "globals.css"),
      "utf8",
    )
    expect(globals).toContain(IMPORT)
    const above = globals
      .slice(0, globals.indexOf(IMPORT))
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .split("\n")
      .filter((line) => line.trim() !== "" && !line.startsWith("@import "))
    // Named rather than counted, so a failure says which rule got in the way.
    expect(above).toEqual([])
  })

  it("counts a declaration and not a mention of one", () => {
    // The fixture rather than the tree, so this keeps proving the stripping
    // after somebody deletes the `.otari-markdown` comments that motivated it.
    const css = [
      "/* .otari-ghost is gone; see .otari-real for what replaced it. */",
      ".otari-real { color: red; }",
      "@utility otari-util { color: blue; }",
      "/* @utility otari-phantom is not a declaration either. */",
    ].join("\n")
    expect([...declared(css)].sort()).toEqual(["otari-real", "otari-util"])
  })

  it("declares every class a primitive wears", () => {
    const available = declared()
    const missing = [...worn().literals]
      .filter((name) => !available.has(name))
      .sort()
    // Named rather than counted, so a failure says which rule to restore.
    expect(missing).toEqual([])
  })

  it("declares a rule for every size a dialog can render", () => {
    // The union is read from the component rather than repeated here: a fifth
    // size added to `DialogSize` with no rule to match is the case this exists
    // for, and a hardcoded list would pass straight through it.
    const sizesOf = (file: string, type: string): string[] => {
      const source = readFileSync(join(DESIGN_SYSTEM, file), "utf8")
      const union = new RegExp(`export type ${type} =([^\\n]*(?:\\n[^\\n]*)?)`)
      const declaration = union.exec(source)?.[1] ?? ""
      return [...declaration.matchAll(/"([a-z]+)"/g)].map((m) => m[1]).sort()
    }
    const available = declared()
    const cases: [string, string, string][] = [
      ["feedback/Dialog.tsx", "DialogSize", "otari-dialog--"],
      ["feedback/FormDialog.tsx", "FormDialogSize", "otari-form-dialog--"],
    ]
    for (const [file, type, prefix] of cases) {
      const sizes = sizesOf(file, type)
      expect(sizes.length).toBeGreaterThan(1)
      const undeclared = sizes.filter(
        (size) => !available.has(`${prefix}${size}`),
      )
      expect({ prefix, undeclared }).toEqual({ prefix, undeclared: [] })
    }
  })

  it("reads the prefixes the dialogs build at render", () => {
    // Pins the two that exist, so dropping a `${size}` interpolation (and with
    // it the whole modifier mechanism) fails here rather than silently shrinking
    // what the test above covers.
    expect([...worn().prefixes].sort()).toEqual([
      "otari-dialog--",
      "otari-form-dialog--",
    ])
  })
})
