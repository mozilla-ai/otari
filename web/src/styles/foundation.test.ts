import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"

import { describe, expect, it } from "vitest"

// Resolved from the Vitest root (web/) rather than import.meta.url, which the
// jsdom environment reports as an http URL. Same reason as src/routes.test.ts.
const WEB = process.cwd()
const CSS = readFileSync(join(WEB, "src", "styles", "globals.css"), "utf8")

// The line that separates the rehomed design foundation from the pre-rehome
// `--otari-*` system it is replacing. Everything above it is the foundation;
// everything below is on its way out.
const BRIDGE_MARKER = "MIGRATION BRIDGE"

/**
 * The text of one top-level block, from `selector {` to the `}` that closes it
 * in the first column. Every block in globals.css is written that way, so a
 * brace counter would only add ways to be subtly wrong about a nested at-rule.
 */
function block(selector: string): string {
  const start = CSS.indexOf(`${selector} {`)
  expect(start, `no \`${selector} {\` block in globals.css`).toBeGreaterThan(-1)
  const end = CSS.indexOf("\n}\n", start)
  expect(end, `\`${selector}\` block is never closed`).toBeGreaterThan(start)
  return CSS.slice(start, end)
}

/** Every custom property *declared* in a block, as name → value. */
function declarations(text: string): Map<string, string> {
  const found = new Map<string, string>()
  for (const [, name, value] of text.matchAll(
    /^\s{2}(--[a-z0-9-]+):\s*([^;]+);/gm,
  )) {
    found.set(name, value.trim().replace(/\s+/g, " "))
  }
  return found
}

const LIGHT = declarations(block(':root,\n.light,\n[data-theme="light"]'))
const DARK = declarations(block('.dark,\n[data-theme="dark"]'))

// The HeroUI variables the mapping owns. A HeroUI component reads these and
// nothing else, so this list is the whole reason a `<Card>` or a `<Chip>` wears
// the Otari palette without a single className at the call site.
//
// `--content1` … `--content4` are deliberately absent: upstream's stylesheet
// still maps them, but @heroui/styles v3 neither declares nor reads them, so
// requiring them here would enforce dead configuration and imply a `bg-content1`
// utility that does not exist.
const HEROUI_VARIABLES = [
  "--background",
  "--foreground",
  "--muted",
  "--border",
  "--default",
  "--default-foreground",
  "--accent",
  "--accent-foreground",
  "--focus",
  "--link",
  "--success",
  "--success-foreground",
  "--warning",
  "--warning-foreground",
  "--danger",
  "--danger-foreground",
  "--surface",
  "--surface-foreground",
  "--surface-secondary",
  "--surface-secondary-foreground",
  "--surface-tertiary",
  "--surface-tertiary-foreground",
  "--surface-shadow",
  "--overlay",
  "--overlay-foreground",
  "--overlay-shadow",
]

describe("design foundation tokens", () => {
  it("declares the same token set in both themes", () => {
    // Each theme block owns the complete set of variables it needs rather than
    // inheriting from its sibling, which is what makes toggling the theme a
    // change of values and never a change of which values exist. A token added
    // to one block and forgotten in the other falls back to the light value in
    // dark mode, which reads as a contrast bug three pages away from the edit.
    expect([...DARK.keys()].sort()).toEqual([...LIGHT.keys()].sort())
  })

  it("maps every HeroUI variable to a token rather than a literal", () => {
    for (const variable of HEROUI_VARIABLES) {
      for (const [theme, tokens] of [
        ["light", LIGHT],
        ["dark", DARK],
      ] as const) {
        const value = tokens.get(variable)
        expect(
          value,
          `${variable} is unmapped in the ${theme} theme`,
        ).toBeDefined()
        // A hex here would be a second source of truth for a color the token
        // above already names, and it is the one place where that would go
        // unnoticed: the utility keeps working, it just stops tracking the token.
        expect(
          value,
          `${variable} bypasses the tokens in the ${theme} theme`,
        ).toMatch(/^var\(--(color|shadow)-[a-z0-9-]+\)$/)
      }
    }
  })

  it("registers its non-HeroUI utilities in @theme", () => {
    // Tailwind v4 generates a utility only for what @theme declares, so a token
    // HeroUI has no counterpart for is unreachable from a className until it is
    // listed here. These are the ones a component is expected to reach for.
    const theme = block("@theme")
    for (const token of [
      "--font-sans",
      "--font-mono",
      "--font-display",
      "--color-background-alt",
      "--color-surface-alt",
      "--color-link",
      "--color-attention",
      "--color-code-surface",
      "--shadow-elevation-sm",
      "--shadow-modal",
    ]) {
      expect(theme).toContain(`${token}:`)
    }
  })

  // Every utility the house style tells a component to reach for, and where it
  // has to come from. This is the list that failed silently once: the status
  // tokens were declared in both theme blocks and documented as classes, but
  // only the three HeroUI happens to generate existed, so `bg-danger-subtle`
  // and `text-info` were classNames that produced no CSS, on a page that looked
  // merely unstyled rather than broken. Documenting a new utility means adding
  // it here in the same commit, which is what keeps the docs and the stylesheet
  // from disagreeing again.
  //
  // "heroui" means HeroUI's own @theme generates it from the variable our
  // mapping block feeds (`--success`, `--muted`, `--surface`, …), so it is a
  // class whether or not we register the `--color-*` token behind it.
  const DOCUMENTED_UTILITIES: Array<[string, "heroui" | "ours"]> = [
    ["bg-background", "heroui"],
    ["text-foreground", "heroui"],
    ["text-muted", "heroui"],
    ["border-border", "heroui"],
    ["bg-surface", "heroui"],
    ["text-success", "heroui"],
    ["text-warning", "heroui"],
    ["text-danger", "heroui"],
    ["bg-success-subtle", "ours"],
    ["bg-warning-subtle", "ours"],
    ["bg-danger-subtle", "ours"],
    ["text-info", "ours"],
    ["bg-info-subtle", "ours"],
    ["bg-primary-subtle", "ours"],
    ["bg-attention", "ours"],
    ["bg-attention-subtle", "ours"],
    ["bg-background-alt", "ours"],
    ["bg-surface-alt", "ours"],
    ["text-link", "ours"],
    ["bg-code-surface", "ours"],
  ]

  it.each(DOCUMENTED_UTILITIES.filter(([, owner]) => owner === "ours"))(
    "%s is registered in @theme",
    (utility) => {
      // Strip the property prefix: `bg-danger-subtle` is generated by
      // `--color-danger-subtle`, `text-link` by `--color-link`.
      const token = `--color-${utility.replace(/^(bg|text|border)-/, "")}`
      expect(block("@theme")).toContain(`${token}:`)
    },
  )

  it("fences the pre-rehome tokens below the migration bridge", () => {
    // The bridge is allowed to exist; it is not allowed to spread. A new
    // `--otari-*` declared up in the foundation would be a third palette
    // growing inside the file that is meant to be retiring the second one.
    const bridge = CSS.indexOf(BRIDGE_MARKER)
    expect(
      bridge,
      "globals.css has no migration bridge marker",
    ).toBeGreaterThan(-1)
    const strays = [...CSS.slice(0, bridge).matchAll(/(--otari-[a-z0-9-]+):/g)]
    expect(strays.map((match) => match[1])).toEqual([])
  })
})

describe("shared/components/ui", () => {
  const UI_DIR = join(WEB, "src", "shared", "components", "ui")
  const sources = readdirSync(UI_DIR, { recursive: true })
    .map(String)
    .filter((name) => name.endsWith(".tsx") && !name.endsWith(".test.tsx"))

  it("covers the primitives directory", () => {
    // A guard on the guard: an emptied or moved directory would leave the two
    // assertions below passing over nothing at all.
    expect(sources.length).toBeGreaterThan(0)
  })

  // This is the directory new work is built from, so it is the one place where
  // "semantic tokens only, no hardcoded palette colors" can be enforced rather
  // than asked for. The bridge components in the parent directory predate the
  // rule and are exempt by being outside it; converting one is what moves it in.
  it.each(sources)("styles %s from semantic tokens only", (name) => {
    const source = readFileSync(join(UI_DIR, name), "utf8")
    expect(
      source,
      "a raw hex color belongs in globals.css as a token",
    ).not.toMatch(/#[0-9a-fA-F]{3,8}\b/)
    // bg-red-500, text-gray-900, border-emerald-200: Tailwind's numbered
    // palette. A status color is a token (`text-danger`, `bg-success-subtle`),
    // not a shade picked at the call site.
    expect(
      source,
      "a numbered Tailwind palette class is not a token",
    ).not.toMatch(
      /\b(?:bg|text|border|ring|fill|stroke|from|via|to|outline|decoration|shadow|accent|caret|divide|placeholder)-(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}\b/,
    )
  })
})
