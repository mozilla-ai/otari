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

  it.each(["app", "features", "shared"])(
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
    expect(pkg.scripts.lint).toContain("biome lint")
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
    const runsLint = readFileSync(workflow, "utf8").includes("npm run lint")
    expect(runsLint, "otari-dashboard.yml does not run npm run lint").toBe(true)
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
      "features",
      "routes",
      "shared",
      "styles",
      "tests",
    ])
  })
})
