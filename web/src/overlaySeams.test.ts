import { existsSync, readdirSync, readFileSync } from "node:fs"
import { dirname, join, relative, resolve, sep } from "node:path"

import { describe, expect, it } from "vitest"

// Resolved from the Vitest root (web/) rather than import.meta.url, which the
// jsdom environment reports as an http URL. Same reason as src/routes.test.ts.
const SRC = join(process.cwd(), "src")

/**
 * The base modules a superset build replaces, by their path under `src/`.
 *
 * Each ships an inert default here (an empty contribution or a component that
 * renders nothing) and is swapped at build time by an alias keyed on its `@/…`
 * specifier, the technique GitLab documents as `ee_else_ce`. The alias table
 * itself belongs to the superset build's config in `otari-ai`, not here: its
 * replacements resolve against an overlay tree that this repository has none of
 * and is not meant to grow one of (see AGENTS.md). What this repository owes the
 * mechanism is the other half, and it is the half this file checks.
 *
 * Add a module here when you add a seam.
 */
const SEAM_MODULES = [
  "app/nav/overlaySections.ts",
  "app/nav/overlayLabelOverrides.ts",
  "app/nav/overlayWalletSlot.tsx",
] as const

const seamPaths = new Set(SEAM_MODULES.map((module) => join(SRC, module)))

/**
 * Every source file under `src/`, tests excluded: a test may reach a seam either
 * way, and one deliberately does (see `overlayWalletSlot.test.tsx`).
 *
 * `architecture.test.ts` plants throwaway modules under `__boundary_probe__` to
 * ask Biome what the real config says about a real path, and deletes each one an
 * assertion later. Vitest runs the two files in parallel, so a probe that exists
 * when this list is built may be gone by the time it is read. Skipping the
 * directory by name is what keeps that from being a coin flip; it holds nothing
 * this file has an opinion about either way.
 */
const sourceFiles = readdirSync(SRC, { recursive: true })
  .map(String)
  .filter((name) => /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name))
  .filter((name) => !name.split(sep).includes("__boundary_probe__"))
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
 * exactly here.
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
