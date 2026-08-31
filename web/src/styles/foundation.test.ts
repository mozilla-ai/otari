import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"

import { describe, expect, it } from "vitest"

// Resolved from the Vitest root (web/) rather than import.meta.url, which the
// jsdom environment reports as an http URL. Same reason as src/routes.test.ts.
const WEB = process.cwd()
const CSS = readFileSync(join(WEB, "src", "styles", "globals.css"), "utf8")

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

/** Source with `//` and block comments removed, for rules about written code. */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "")
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
  "--backdrop",
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

const TYPE_NAMESPACES = /^--(text|font-weight|tracking)-/

/**
 * WCAG 2.x relative luminance, sRGB. Kept here rather than pulled in as a
 * dependency: it is eight lines and the alternative is a package in the
 * dependency graph for one assertion.
 */
function luminance(hex: string): number {
  const m = /^#([0-9a-f]{6})$/i.exec(hex.trim())
  if (!m) throw new Error(`not a six-digit hex color: ${hex}`)
  const channel = (v: number): number => {
    const c = v / 255
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }
  const n = Number.parseInt(m[1], 16)
  return (
    0.2126 * channel((n >> 16) & 0xff) +
    0.7152 * channel((n >> 8) & 0xff) +
    0.0722 * channel(n & 0xff)
  )
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}

describe("text on every ground it can land on", () => {
  // The repair in this change was measured against all six grounds each theme
  // declares, and the measurement is the assertion. Without it the WCAG AA fix
  // regresses on the next palette edit with no symptom until somebody renders
  // it, which is the same invisible-until-rendered class of failure the cascade
  // tests above exist for.
  //
  // Both of these failed AA on every ground before this change (light worst case
  // 3.01, dark 2.91), and the reason it went unnoticed for so long is that the
  // dashboard had no regular-weight text: extra stem weight reads as more
  // contrast than the ratio gives.
  const GROUNDS = [
    "--color-background",
    "--color-background-muted",
    "--color-background-subtle",
    "--color-surface",
    "--color-surface-muted",
    "--color-surface-subtle",
  ] as const
  const INK = [
    "--color-text",
    "--color-text-muted",
    "--color-text-subtle",
  ] as const
  // 4.5:1 is AA for normal-size text, which is what all three of these pair with.
  const AA_NORMAL = 4.5

  it.each([
    ["light", LIGHT],
    ["dark", DARK],
  ] as const)("clears AA for normal text in the %s theme", (_theme, tokens) => {
    for (const ink of INK) {
      for (const ground of GROUNDS) {
        const fg = tokens.get(ink)
        const bg = tokens.get(ground)
        expect(fg, `${ink} is not declared`).toBeDefined()
        expect(bg, `${ground} is not declared`).toBeDefined()
        const ratio = contrast(fg as string, bg as string)
        // Compare the unrounded ratio. Rounding first would let 4.496 pass as
        // 4.50, which is a test that goes green on a value that fails AA.
        // Rounding belongs in the message and nowhere else.
        expect(
          ratio,
          `${ink} on ${ground} is ${ratio.toFixed(2)}:1, under the ${AA_NORMAL}:1 AA floor for normal text`,
        ).toBeGreaterThanOrEqual(AA_NORMAL)
      }
    }
  })
})

describe("the type scale's two halves", () => {
  // The scale is split on purpose and the split is easy to undo by accident, so
  // it is pinned from both sides. `@heroui/styles` is a prebuilt Tailwind
  // stylesheet carrying its own `@layer theme` with the default `--text-*`,
  // `--font-weight-*` and `--tracking-*` in it, and it is imported after
  // Tailwind, so an override of one of those written in `@theme` loses on source
  // order: it compiles, emits, and changes nothing. The values therefore live in
  // an unlayered `:root` block, which beats every layer, and only the keys
  // HeroUI never declares are registered in `@theme`.
  // `block()` is a first-match `indexOf`, and globals.css has two top-level
  // `:root {` blocks: this one and the base reset further down. It resolves
  // correctly only because this one comes first, so assert that rather than
  // depend on it: a reordering would otherwise point every failure below at the
  // wrong block.
  const TYPE_VALUES = declarations(block(":root"))

  it("matches the type block rather than the base reset", () => {
    expect(
      TYPE_VALUES.get("--text-xs"),
      "block(':root') matched a different :root block (the base reset?); anchor it on something distinguishing",
    ).toBeDefined()
  })

  it("declares every contested type key in the unlayered :root block", () => {
    for (const key of [
      "--text-xs",
      "--text-xs--line-height",
      "--text-sm",
      "--text-sm--line-height",
      "--text-base",
      "--text-base--line-height",
      "--text-lg",
      "--text-lg--line-height",
      "--text-xl",
      "--text-xl--line-height",
      "--text-2xl",
      "--text-2xl--line-height",
      "--font-weight-normal",
      "--font-weight-medium",
      "--font-weight-semibold",
      "--font-weight-bold",
      "--tracking-tight",
      "--text-caption-step",
      "--text-caption-step--line-height",
      "--text-caption-step--letter-spacing",
    ]) {
      expect(
        TYPE_VALUES.get(key),
        `${key} is not declared in the unlayered :root block, so @heroui/styles' own value wins`,
      ).toBeDefined()
    }
  })

  it("keeps type metrics out of the theme blocks", () => {
    // Not a style preference. Type has no theme axis, and declaring these twice
    // would allow one block to be edited and the other missed, giving the two
    // themes different type metrics: invisible in whichever theme you are
    // looking at, and undetectable by the both-themes assertion below, because
    // the key sets would still match.
    for (const [theme, tokens] of [
      ["light", LIGHT],
      ["dark", DARK],
    ] as const) {
      const offenders = [...tokens.keys()].filter((name) =>
        TYPE_NAMESPACES.test(name),
      )
      expect(
        offenders,
        `the ${theme} theme block declares type metrics (${offenders.join(", ")}); they belong in the unlayered :root block, declared once`,
      ).toEqual([])
    }
  })

  it("registers in @theme only the keys HeroUI does not declare", () => {
    // Tracking is emitted into `.text-xs` and friends only if the key is
    // registered at build time, so these have to be in `@theme`; a value at
    // :root alone is never read, because the utility would not reference it.
    const theme = declarations(block("@theme"))
    for (const key of [
      "--text-xs--letter-spacing",
      "--text-sm--letter-spacing",
      "--text-base--letter-spacing",
      "--text-lg--letter-spacing",
      "--text-xl--letter-spacing",
      "--text-2xl--letter-spacing",
    ]) {
      expect(
        theme.get(key),
        `${key} is not registered in @theme, so its utility never references it`,
      ).toBeDefined()
    }
    // The other direction: a size or weight registered here is the silent-no-op
    // mistake this split exists to prevent.
    // No exceptions any more: the only type keys `@theme` may carry are the
    // letter-spacing registrations. A *size* here would also generate a
    // same-named utility that outranks the role of that name in the same layer,
    // which is how the caption step silently killed `@utility text-caption`'s
    // own metrics before it was renamed to `--text-caption-step`.
    const contested = [...theme.keys()].filter(
      (name) =>
        TYPE_NAMESPACES.test(name) && !name.endsWith("--letter-spacing"),
    )
    expect(
      contested,
      `@theme registers type values @heroui/styles also declares (${contested.join(", ")}); it loses on source order, so these belong in the unlayered :root block`,
    ).toEqual([])
  })
})

describe("the flat surface: no corners, no elevation", () => {
  // Both halves of the divided-surface direction are one-line token changes
  // that reach the whole app, and both fail silently if they land in the wrong
  // place: a squared corner that is still round, or a shadow zeroed in one
  // theme only, compiles clean and reports nothing.

  // The text between the end of the previous top-level block and the radius
  // declaration. Enough to tell whether the declaration is unlayered: if a
  // `@theme` or `@layer` opener sits in there, it is not.
  const radiusContext = (() => {
    const decl = CSS.indexOf("--radius: 0px;")
    // Empty rather than an assertion when the declaration is gone, so the two
    // tests below fail with their own messages instead of taking the whole
    // file down at collection time.
    if (decl === -1) return ""
    // Comments stripped: the block below this one explains the cascade rule in
    // prose, and the words `@theme` and `@layer` in a comment are not openers.
    return CSS.slice(CSS.lastIndexOf("\n}\n", decl), decl).replace(
      /\/\*[\s\S]*?\*\//g,
      "",
    )
  })()

  it("zeroes the one radius key every corner derives from", () => {
    // `--radius-xl|2xl|3xl` and `--field-radius` are all `calc(var(--radius) * n)`
    // in the emitted stylesheet, so this key is the whole radius surface.
    const declarations = [...CSS.matchAll(/^\s*--radius:\s*([^;]+);/gm)].map(
      ([, value]) => value.trim(),
    )
    expect(
      declarations,
      "`--radius` should be declared exactly once, at zero",
    ).toEqual(["0px"])
  })

  it("declares the radius unlayered, where it can outrank @heroui/styles", () => {
    // `@heroui/styles` declares `--radius` inside its own `@layer base`. An
    // override written in `@theme` is hoisted to the top theme layer, compiles,
    // emits, and loses on source order. Only an unlayered declaration wins.
    expect(
      radiusContext,
      "the `--radius` declaration is not in a `:root` block",
    ).toContain(":root {")
    const opener = radiusContext.match(/^@(theme|layer)\b.*/m)
    expect(
      opener?.[0],
      "the `--radius` declaration sits inside an at-rule block, so @heroui/styles' own value wins on source order",
    ).toBeUndefined()
  })

  it("keeps the radius out of the theme blocks", () => {
    // Same rule as the type metrics above: a corner has no theme axis, so
    // declaring it twice only creates a way for the two themes to drift.
    for (const [theme, tokens] of [
      ["light", LIGHT],
      ["dark", DARK],
    ] as const) {
      const offenders = [...tokens.keys()].filter((name) =>
        name.startsWith("--radius"),
      )
      expect(
        offenders,
        `the ${theme} theme block declares radius keys (${offenders.join(", ")}); radius is declared once, unlayered`,
      ).toEqual([])
    }
  })

  it("zeroes every elevation token in both themes", () => {
    // A shadow *does* have a theme axis, so unlike the radius these are
    // declared per theme, and that is exactly what makes a half-done change
    // possible: zeroed on light, still lifting on dark, invisible in whichever
    // theme you happen to be looking at. `--shadow-elevation-sm|md|lg` and
    // `--shadow-modal` (the registered utilities), `--surface-shadow` and
    // `--overlay-shadow` (which HeroUI's card, popover, modal and toast rules
    // read) all alias these four.
    for (const [theme, tokens] of [
      ["light", LIGHT],
      ["dark", DARK],
    ] as const) {
      for (const key of [
        "--shadow-sm",
        "--shadow-md",
        "--shadow-lg",
        "--shadow-modal",
      ]) {
        expect(
          tokens.get(key),
          `${theme}'s ${key} still casts; the surface is one plane in both themes`,
        ).toBe("none")
      }
    }
  })
})

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
    ["text-attention", "ours"],
    ["border-attention-border", "ours"],
    ["bg-background-alt", "ours"],
    ["bg-surface-alt", "ours"],
    ["bg-surface-subtle", "ours"],
    ["text-link", "ours"],
    ["text-link-hover", "ours"],
    ["text-primary-subtle-foreground", "ours"],
    ["bg-code-surface", "ours"],
    ["border-control-border", "ours"],
    ["bg-control-thumb", "ours"],
    ["text-accent", "heroui"],
    ["bg-accent", "heroui"],
    ["text-accent-foreground", "heroui"],
    ["bg-muted", "heroui"],
    ["bg-field", "heroui"],
    ["border-field-border", "heroui"],
    ["bg-backdrop", "heroui"],
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

  it("keeps the retired pre-rehome palette retired", () => {
    // The `--otari-*` palette this dashboard was built on is gone, and the one
    // way it comes back is somebody reintroducing a variable rather than
    // finding the role they need. Class names keeping the `otari-` namespace
    // (`.otari-table`, `.otari-markdown`) are not that and are matched around.
    const strays = [...CSS.matchAll(/(--otari-[a-z0-9-]+)\s*:/g)]
    expect(strays.map((match) => match[1])).toEqual([])
  })

  // The handful of places a token value has to be repeated as a literal, because
  // the file reading it cannot read a custom property: the `theme-color` meta tag
  // and the boot splash in index.html (which paints before this stylesheet is a
  // certainty), and the manifest an installed app opens on. Each of those files
  // carried a comment saying nothing enforced the copy. Something does now.
  describe("literal copies of a token outside CSS", () => {
    const HTML = readFileSync(join(WEB, "index.html"), "utf8")
    const MANIFEST = readFileSync(
      join(WEB, "public", "pwa", "manifest.webmanifest"),
      "utf8",
    )
    /** A light-theme token's value, failing rather than comparing `undefined`. */
    const light = (token: string) => {
      const value = LIGHT.get(token)
      expect(value, `${token} is not declared in the light theme`).toBeDefined()
      return value as string
    }

    it.each([
      ["--color-background", "background"],
      ["--color-border", "border"],
      ["--color-primary", "primary"],
    ])("index.html falls back to %s", (token) => {
      // `var(--color-x, #literal)`: the fallback is what paints before
      // globals.css has loaded, so a drifted one is a flash of the wrong color.
      expect(HTML).toContain(`var(${token}, ${light(token)})`)
    })

    it("index.html tints the browser chrome with --color-primary", () => {
      expect(HTML).toContain(
        `<meta name="theme-color" content="${light("--color-primary")}" />`,
      )
    })

    it("the web app manifest matches the canvas and the brand", () => {
      const manifest = JSON.parse(MANIFEST) as Record<string, string>
      expect(manifest.background_color).toBe(light("--color-background"))
      expect(manifest.theme_color).toBe(light("--color-primary"))
    })
  })
})

describe("semantic tokens only", () => {
  // Every source file that styles anything, which is the whole of `src` now
  // that there is no bridge tree left to be exempt. This used to be scoped to
  // `shared/components/ui/` because the hand-rolled primitives beside it
  // predated the foundation; they are on it now, so the exemption has no
  // subject and the rule is the repo's.
  const SRC = join(WEB, "src")
  // The one deliberate exception, and it says why at the top of the file: the
  // share card is rasterized through an <img>-loaded SVG document, where a
  // custom property does not resolve, so its palette has to be literal.
  const EXCEPTIONS = new Set(["features/usage/ShareCard.tsx"])
  const sources = readdirSync(SRC, { recursive: true })
    .map((name) => String(name).replaceAll("\\", "/"))
    .filter(
      (name) =>
        /\.tsx?$/.test(name) &&
        !/\.test\.tsx?$/.test(name) &&
        !EXCEPTIONS.has(name),
    )

  it("covers the source tree", () => {
    // A guard on the guard: a moved directory or a broken filter would leave
    // the assertions below passing over nothing at all.
    expect(sources.length).toBeGreaterThan(50)
  })

  it.each(sources)("styles %s from semantic tokens only", (name) => {
    const source = readFileSync(join(SRC, name), "utf8")
    // Six or eight digits anywhere, three or four only when quoted or in a
    // Tailwind arbitrary value: `#478` in "issue #478" is a ticket, not a color,
    // and this file's whole job is to be trusted rather than muted.
    expect(
      source,
      "a raw hex color belongs in globals.css as a token",
    ).not.toMatch(
      /["'`[]#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b|#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b/,
    )
    // bg-red-500, text-gray-900, border-emerald-200: Tailwind's numbered
    // palette. A status color is a token (`text-danger`, `bg-success-subtle`),
    // not a shade picked at the call site, and a numbered class is invisible to
    // the dark theme. The optional side/axis segment matters: `border-l-red-500`
    // is the same mistake and slipped past a version of this pattern that only
    // knew `border-`.
    expect(
      source,
      "a numbered Tailwind palette class is not a token",
    ).not.toMatch(
      /\b(?:bg|text|border|ring|fill|stroke|from|via|to|outline|decoration|shadow|accent|caret|divide|placeholder)(?:-[tblrsexy])?-(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}\b/,
    )
    // A focus ring is an accessibility floor, and until this rule it was spelled
    // at nine call sites that did not agree: four drew it from `accent`, two
    // from `focus`, and the rest left the width and offset to the browser. None
    // of that fails anything, which is why it drifted. The ring is defined once
    // now, in `@layer base` for plain elements and as `@utility otari-focus-ring`
    // the two that have to outrank a HeroUI component rule, so a call site has
    // nothing left to spell. `ring-0` and `ring-offset-0` are deliberately not
    // matched: those suppress HeroUI's own inner ring rather than drawing ours.
    //
    // Matched against the code with comments removed. `rowStyles.ts` explains
    // why its rows cannot use the base rule, and that explanation has to be
    // able to name the thing it is explaining.
    expect(
      stripComments(source),
      "a focus ring is defined once in globals.css; use `otari-focus-ring`, do not spell one here",
    ).not.toMatch(
      /\b(?:ring|outline)-(?:accent|primary|focus)\b|\boutline-offset-\d|\b(?:ring|outline)-[1-9]\b/,
    )
    // `bg-white` / `text-black` are the same problem without a number: a
    // literal that never follows the theme.
    expect(
      source,
      "bg-white / text-black do not follow the theme; use a surface or text token",
    ).not.toMatch(
      /\b(?:bg|text|border|ring|fill|stroke|from|via|to|outline|decoration|shadow|accent|caret|divide|placeholder)(?:-[tblrsexy])?-(?:white|black)\b/,
    )
  })
})

// The chrome's own type roles, added because the nav files had grown six
// arbitrary sizes between them (13.5px twice, 13px, 11.5px, 11px, 9px), which is
// a scale with no single place to read it and nothing to keep it from growing a
// seventh. Both halves are asserted: the roles exist, and the chrome uses them.
describe("the shell chrome's type roles", () => {
  const CHROME_ROLES = [
    "text-chrome-row",
    "text-chrome-meta",
    "text-chrome-initials",
  ]

  it.each(CHROME_ROLES)("declares %s in globals.css", (role) => {
    // The same failure the documented-utilities list guards against: a role
    // named in the scale's comment but never declared is a className that
    // produces no CSS, on text that looks merely unstyled rather than broken.
    expect(CSS).toContain(`@utility ${role} {`)
  })

  it("leaves no arbitrary font size in the nav chrome", () => {
    const NAV = join(WEB, "src", "app", "nav")
    const offenders = readdirSync(NAV)
      .filter((name) => /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name))
      .filter((name) =>
        // `text-[` followed by a digit: an arbitrary size, as opposed to an
        // arbitrary color, which the token rules above already cover.
        /text-\[\d/.test(readFileSync(join(NAV, name), "utf8")),
      )

    expect(offenders, "use a text-chrome-* role, or add one").toEqual([])
  })
})
