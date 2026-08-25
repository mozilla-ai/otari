import { existsSync, readdirSync, readFileSync } from "node:fs"
import { dirname, join, relative, resolve, sep } from "node:path"

import { describe, expect, it } from "vitest"

// Resolved from the Vitest root (web/) rather than import.meta.url, which the
// jsdom environment reports as an http URL. Same reason as src/routes.test.ts.
const SRC = join(process.cwd(), "src")

// The name `architecture.test.ts` plants its throwaway modules under.
const PROBE_DIR = "__boundary_probe__"

// Both suffixes `vitest.config` collects.
const TEST_FILE = /\.(test|spec)\.tsx?$/

/**
 * The base modules a superset build replaces, by their path under `src/`.
 *
 * Each ships an inert default here and is swapped at build time by an alias
 * keyed on its `@/…` specifier. The list is checked for completeness below, so
 * a new seam fails this file until it is added rather than going unguarded.
 */
const SEAM_MODULES = [
  "app/nav/overlaySections.ts",
  "app/nav/overlayNavItems.ts",
  "app/nav/overlayLabelOverrides.ts",
  "app/nav/overlayWalletSlot.tsx",
  "app/overlayEntitlementResolver.tsx",
  "shared/telemetry/overlayTelemetry.ts",
] as const

const seamPaths = new Set(SEAM_MODULES.map((module) => join(SRC, module)))

/**
 * Every file under `src/`, retried once if the walk loses a race.
 *
 * `architecture.test.ts` plants throwaway modules under `__boundary_probe__` to
 * ask Biome what the real config says about a real path, and `rmSync`s each one
 * an assertion later. Vitest runs the two files in parallel, so a directory can
 * vanish between the parent listing and the recursion into it, which throws
 * rather than yielding a short list. Skipping the probe directory by name keeps
 * its files from being read; the retry covers the walk itself.
 */
function walkSrc(): string[] {
  for (let attempt = 0; ; attempt++) {
    try {
      return readdirSync(SRC, { recursive: true })
        .map(String)
        .filter((name) => !name.split(sep).includes(PROBE_DIR))
    } catch (error) {
      if (attempt > 0) throw error
    }
  }
}

const allFiles = walkSrc()

/**
 * Sources only: a test may reach a seam either way, and one deliberately does.
 *
 * `.spec` as well as `.test`, because `vitest.config` collects both and a spec
 * exempted here but collected there would be held to a rule its siblings are not.
 */
const sourceFiles = allFiles
  .filter((name) => /\.tsx?$/.test(name) && !TEST_FILE.test(name))
  .map((name) => join(SRC, name))

// `from "x"`, a bare `import "x"`, and a dynamic `import("x")`. Biome formats to
// double quotes, so a single-quoted specifier cannot survive `lint`.
const SPECIFIER = /(?:from|import)\s*\(?\s*"([^"]+)"/g

/** What a relative specifier written in `file` points at, with the extension TypeScript infers. */
function resolveRelative(file: string, specifier: string): string | undefined {
  const base = resolve(dirname(file), specifier)
  return [base, `${base}.ts`, `${base}.tsx`, join(base, "index.ts")].find(
    (candidate) => existsSync(candidate) && seamPaths.has(candidate),
  )
}

function specifiersIn(file: string): string[] {
  return [...readFileSync(file, "utf8").matchAll(SPECIFIER)].map(
    (match) => match[1] as string,
  )
}

const posix = (file: string) => relative(SRC, file).split(sep).join("/")
/** The same spelling `SEAM_MODULES` uses, for a path already relative to `src/`. */
const posixName = (name: string) => name.split(sep).join("/")

/**
 * A seam module has to be reached by its `@/…` specifier and never relatively.
 *
 * That specifier is the key the superset build's alias matches. A relative
 * import of the same file resolves to a path the alias never sees, so the
 * overlay's contribution vanishes with no error, in the one kind of build whose
 * only symptom is the empty default it was going to render anyway. Nothing about
 * this repository's own build can notice, which is why it is asserted here
 * rather than left to a review note; the general rule in the frontend-standards
 * skill (a same-directory sibling is clearer as `./Sibling`) has its exception
 * precisely here.
 */
describe("overlay seam modules", () => {
  it("are all present", () => {
    // A guard on the guard: a renamed or deleted seam would leave every
    // assertion below vacuously true.
    expect(sourceFiles.length).toBeGreaterThan(50)
    for (const module of SEAM_MODULES) {
      expect(existsSync(join(SRC, module)), `${module} is missing`).toBe(true)
    }
  })

  it("are the whole of what the list claims to cover", () => {
    // The assertion above catches a deleted seam. This one catches a forgotten
    // one, which is the drift that actually happened: #584 and #644 each added a
    // seam, both were reachable only relatively, and nothing said so until this
    // file existed.
    //
    // The discovery rule is the naming convention, which both this repository
    // and otari-ai already follow. Be clear about what that does and does not
    // buy: it makes the convention enforceable, not the seam set provable. A
    // seam named for what it does rather than `overlay<Capital>` is invisible
    // here, so naming one that way is the way to end up unguarded again. That is
    // the trade for having any automated check at all, since nothing else on
    // this side distinguishes a seam from an ordinary module.
    const discovered = allFiles
      .filter((name) => !TEST_FILE.test(name))
      .map(posixName)
      .filter((name) => /(^|\/)overlay[A-Z][^/]*\.tsx?$/.test(name))
      .sort()

    expect(
      discovered,
      "every module named `overlay<Capital>` under src/ is treated as an overlay " +
        "seam and must be listed in SEAM_MODULES. Add it there if it is one; " +
        "rename it if it is not (the name is what this file discovers by).",
    ).toEqual([...SEAM_MODULES].sort())
  })

  it.each(SEAM_MODULES)("%s is reached by its @/… specifier", (module) => {
    const alias = `@/${module.replace(/\.tsx?$/, "")}`
    const importers = sourceFiles.filter(
      (file) => !seamPaths.has(file) && specifiersIn(file).includes(alias),
    )

    // A seam nothing imports is not a seam: replacing it would change nothing.
    expect(importers.map(posix), `nothing imports ${alias}`).not.toEqual([])
  })

  it("are never reached relatively", () => {
    const offenders = sourceFiles.flatMap((file) =>
      specifiersIn(file)
        .filter((specifier) => specifier.startsWith("."))
        .filter((specifier) => resolveRelative(file, specifier))
        .map((specifier) => `${posix(file)} imports "${specifier}"`),
    )

    expect(offenders).toEqual([])
  })
})
