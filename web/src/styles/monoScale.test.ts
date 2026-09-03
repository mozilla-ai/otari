import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"

import { describe, expect, it } from "vitest"

// Resolved from the Vitest root (web/) rather than import.meta.url, which the
// jsdom environment reports as an http URL. Same reason as foundation.test.ts.
const WEB = process.cwd()
const CSS = readFileSync(join(WEB, "src", "styles", "globals.css"), "utf8")
const SRC = join(WEB, "src")

/** The text of one `@utility` block, from its opening brace to the `}` in the
 *  first column that closes it. Same shape as foundation.test.ts's `block`. */
function utility(name: string): string {
  const start = CSS.indexOf(`@utility ${name} {`)
  expect(start, `no \`@utility ${name}\` in globals.css`).toBeGreaterThan(-1)
  const end = CSS.indexOf("\n}\n", start)
  expect(end, `\`${name}\` is never closed`).toBeGreaterThan(start)
  return CSS.slice(start, end)
}

const MONO_ROLES = [
  "text-mono-caption",
  "text-mono-micro",
  "text-mono-overline",
  "text-mono-figure",
] as const

/**
 * The mono half of the type scale.
 *
 * The sans roles hardcode `font-family: var(--font-sans)`, so a monospace run at
 * a role's own size had no role to wear, and 56 call sites re-spelled the size
 * as a literal instead. That is a gap in the token layer wearing a call-site
 * costume, and fixing it at the call sites would have been 56 hand substitutions
 * that each changed what a page renders.
 *
 * These roles are therefore promised to be size-preserving, which is the thing
 * worth pinning: `text-[13px]` compiles to `font-size: 13px` and nothing else,
 * so a role adding a line height, a tracking or an ink to the same call site
 * would move text that was only supposed to be renamed.
 */
describe("the mono type roles", () => {
  it.each(MONO_ROLES)("declares %s in globals.css", (role) => {
    // A role named in a comment but never declared is a className that produces
    // no CSS, on text that looks merely unstyled rather than broken.
    expect(CSS).toContain(`@utility ${role} {`)
  })

  it.each(MONO_ROLES)("sets %s on the mono family", (role) => {
    // The whole reason these exist. A mono role that resolved to the sans family
    // would be the gap it was written to close, silently.
    expect(utility(role)).toContain("font-family: var(--font-mono)")
  })

  it("takes the caption step from the same token the sans caption does", () => {
    // Not a repeated `13px`: the two halves of the scale have to move together
    // or they drift a point apart and nobody sees which one is wrong.
    expect(utility("text-mono-caption")).toContain(
      "font-size: var(--text-caption-step)",
    )
  })

  it.each(MONO_ROLES)("gives %s no ink of its own", (role) => {
    // The size-preserving promise. The 56 call sites carry their own color
    // classes (`text-muted`, `text-subtle`, `text-foreground`, or none at all),
    // and a role setting a color would override some of them and be overridden
    // by others, so half the conversions would change what is on the screen.
    expect(utility(role)).not.toMatch(/^\s*color:/m)
  })

  it("gives the caption and micro roles no metrics beyond their size", () => {
    // `text-mono-figure` is the deliberate exception and is excluded here: both
    // of its call sites already spell the same `leading-[36px]`, so folding it
    // in moves nothing. These two have no such agreement to fold.
    for (const role of ["text-mono-caption", "text-mono-micro"]) {
      const text = utility(role)
      expect(text, `${role} states a line height`).not.toMatch(
        /^\s*line-height:/m,
      )
      expect(text, `${role} states a tracking`).not.toMatch(
        /^\s*letter-spacing:/m,
      )
    }
  })
})

describe("no monospace run spells its own size", () => {
  const sources = readdirSync(SRC, { recursive: true })
    .map((name) => String(name).replaceAll("\\", "/"))
    .filter((name) => /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name))

  it("covers the source tree", () => {
    // Same guard as foundation.test.ts's sweeps: an empty list passes vacuously,
    // and a sweep that silently stopped finding files is the failure that looks
    // most like success.
    expect(sources.length).toBeGreaterThan(30)
  })

  it.each(sources)("pairs no arbitrary size with font-mono in %s", (name) => {
    const source = readFileSync(join(SRC, name), "utf8")
    for (const line of source.split("\n")) {
      expect(
        /font-mono/.test(line) && /text-\[\d/.test(line),
        `${name} spells a monospace size by hand: use text-mono-caption, text-mono-micro, text-mono-overline or text-mono-figure`,
      ).toBe(false)
    }
  })
})
