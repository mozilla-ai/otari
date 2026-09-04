import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"

import { describe, expect, it } from "vitest"

import { buildManifest } from "../../pwaManifest"

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

  // The other half of the declare-vs-register trap, and the half that fails
  // silently in the opposite direction. A self-referential registration
  // (`--color-x: var(--color-x)`) exists only to make the utility, and it
  // resolves to nothing unless a theme block declares a real value. When it
  // does not, `text-x` is a class that compiles, ships, and paints the
  // inherited color instead: no error, no warning, no failing check. That is
  // how `--color-accent-glyph` lost both declarations to a careless edit and
  // turned a white checkmark black.
  it("declares a real value for every self-referential @theme key", () => {
    const theme = declarations(block("@theme"))
    const undeclared: string[] = []
    for (const [key, value] of theme) {
      if (value !== `var(${key})`) continue
      if (!LIGHT.has(key) || !DARK.has(key)) undeclared.push(key)
    }
    expect(
      undeclared,
      "registered in @theme but not declared in both theme blocks, so the utility resolves to nothing",
    ).toEqual([])
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
  // certainty), and the manifest an installed app opens on (generated by
  // web/pwaManifest.ts). Each of those carried a comment saying nothing
  // enforced the copy. Something does now.
  describe("literal copies of a token outside CSS", () => {
    const HTML = readFileSync(join(WEB, "index.html"), "utf8")
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
      const manifest = buildManifest("/")
      expect(manifest.background_color).toBe(light("--color-background"))
      expect(manifest.theme_color).toBe(light("--color-primary"))
    })
  })
})

// A comment opener inside a comment. CSS comments do not nest, so `/*` after
// one has already opened is not an error and nothing fails: the text just
// runs on to the next `*/` and the file quietly grows a duplicated header.
// Three of those reached this file from scripted edits and were found by
// somebody reading the lines next to them, which is not a way of finding
// things.
it("has no comment opened inside another comment in globals.css", () => {
  const offenders: string[] = []
  let inComment = false
  for (let i = 0; i < CSS.length - 1; i++) {
    if (!inComment && CSS[i] === "/" && CSS[i + 1] === "*") {
      inComment = true
      i++
    } else if (inComment && CSS[i] === "*" && CSS[i + 1] === "/") {
      inComment = false
      i++
    } else if (inComment && CSS[i] === "/" && CSS[i + 1] === "*") {
      offenders.push(
        `line ${CSS.slice(0, i).split("\n").length}: ${CSS.slice(i, i + 60)}`,
      )
      i++
    }
  }
  expect(offenders, "a `/*` inside a comment is a duplicated header").toEqual(
    [],
  )
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

// A bare heading is not unstyled: the `@layer base` rule in globals.css hands
// `h1`-`h6` the display face, so a heading that skips the type scale renders in
// Mozilla Headline at whatever size its hand-rolled classes say. otari#810
// migrated 35 sites and still left six behind (otari-ai#1933), which is what a
// rule without a test does.
describe("headings wear a type role", () => {
  const SRC = join(WEB, "src")
  const sources = readdirSync(SRC, { recursive: true })
    .map((name) => String(name).replaceAll("\\", "/"))
    .filter((name) => name.endsWith(".tsx") && !name.endsWith(".test.tsx"))

  it("covers the source tree", () => {
    // Same guard as the token sweep above: an empty list passes vacuously.
    expect(sources.length).toBeGreaterThan(30)
  })

  it.each(sources)("puts every heading in %s on the scale", (name) => {
    // Block comments go first: Login.tsx narrates its layout in JSX comments
    // that mention `<h1>` in prose.
    const source = readFileSync(join(SRC, name), "utf8").replace(
      /\/\*[\s\S]*?\*\//g,
      "",
    )
    for (const match of source.matchAll(/<h[1-6]\b([^>]*)>/g)) {
      const attributes = match[1]
      const literal = attributes.match(/className="([^"]*)"/)
      if (literal) {
        expect(
          literal[1],
          `${match[0]} in ${name} hand-rolls its type; use text-display, text-heading, text-title, or text-overline`,
        )
          // The stems name role FAMILIES, and `\b` matches before a hyphen, so
          // a named sub-step such as `text-display-sub` satisfies this by
          // design rather than by accident. A role named outside these stems
          // would be correct and would still fail here, so it has to arrive
          // with a change to this pattern.
          //
          // `text-overline` is a heading role here even though it is not one
          // of the page's three sizes: the role exists to label the group
          // beneath it, so heading that group is the thing it is for. The
          // alternative was demoting the element to a `span`, which would take
          // the label out of the document outline to satisfy a check. This
          // list and the ink-pairing list below already differ; they answer
          // different questions.
          .toMatch(/\btext-(?:display|heading|title|overline)\b/)
      } else {
        // No string literal, so the one acceptable shape left is an
        // expression. A heading with no className at all is the bare case
        // the base layer turns serif.
        const expression = attributes.match(/className=\{([^}]*)\}/)
        expect(
          expression,
          `${match[0]} in ${name} is a bare heading, which the base layer renders in the display face; give it a type role`,
        ).not.toBeNull()
        let value = (expression as RegExpMatchArray)[1].trim()
        // A lone identifier is read through its declaration in the same
        // file, which is how Login.tsx spells its HEADING constant. Any
        // other expression (a ternary, a template) has to name a role in
        // its own text.
        if (/^\w+$/.test(value)) {
          const declaration = source.match(
            new RegExp(`\\b${value} = "([^"]*)"`),
          )
          expect(
            declaration,
            `${match[0]} in ${name} reads ${value}, which is not a same-file string constant this test can check; use a literal or a const`,
          ).not.toBeNull()
          value = (declaration as RegExpMatchArray)[1]
        }
        expect(
          value,
          `${match[0]} in ${name} hand-rolls its type; use text-display, text-heading, text-title, or text-overline`,
        ).toMatch(/\btext-(?:display|heading|title|overline)\b/)
      }
    }
  })
})

// The content scale's roles, and the spelling that kept them empty. Before this
// sweep `text-caption` had one consumer, `text-body`, `text-emphasis` and
// `text-overline` between them a handful, and 174 sites wrote the caption role
// out as `text-xs text-muted`: a size a point under the role's own and an ink
// the role already sets. A scale nobody uses is documentation, so the roles are
// enforced here the way the chrome's are below.
//
// A role sets family, size, line-height, tracking, weight and ink together, and
// each of those is a utility somebody can write beside it. That pairing is not a
// style preference: Tailwind emits `@utility` definitions ahead of its own
// theme-derived utilities, so a `text-xs` beside `text-caption` wins the size
// and leaves the role compiling, emitting, and doing nothing. `font-mono` is the
// one pairing deliberately left alone, because a role sets the body face and an
// identifier does not belong in it.
describe("content text wears a type role", () => {
  const ROLES = [
    "text-display",
    "text-heading",
    "text-title",
    "text-body",
    "text-emphasis",
    "text-caption",
    "text-overline",
  ] as const
  const ANY_ROLE = `text-(?:${ROLES.map((role) => role.slice("text-".length)).join("|")})`
  // The ink each role sets for itself, and therefore the one a call site wearing
  // it never repeats.
  const ROLE_INK: Array<[string, string]> = [
    ["text-(?:caption|overline)", "text-muted"],
    ["text-(?:display|heading|title|body|emphasis)", "text-foreground"],
  ]

  it("gives every role an ink it must not repeat", () => {
    // ROLE_INK spells the role names a second time, so one added above and
    // missed below would go unchecked rather than fail. This is the guard on
    // that guard.
    const covered = ROLE_INK.flatMap(([roles]) =>
      roles
        .replace(/^text-\(\?:|\)$/g, "")
        .split("|")
        .map((name) => `text-${name}`),
    )
    expect(covered.sort()).toEqual([...ROLES].sort())
  })

  it.each(ROLES)("declares %s in globals.css", (role) => {
    // Same failure the documented-utilities list guards against: a role named in
    // the scale's comment but never declared is a className that produces no CSS.
    expect(CSS).toContain(`@utility ${role} {`)
  })

  /**
   * Two utilities inside one class list. The span between them may not cross a
   * quote, so `className={ok ? "text-caption" : "text-xs text-muted"}` reads as
   * the two lists it is rather than as one. Each name is anchored against a
   * variant prefix and against a longer name, which leaves `sm:text-xs` and
   * `hover:text-muted` the deliberate overrides they look like.
   */
  const together = (first: string, second: string): RegExp => {
    const alone = (name: string) => `(?<![:\\w-])(?:${name})(?![\\w-])`
    return new RegExp(
      `${alone(first)}[^"'\`\\n]*${alone(second)}|${alone(second)}[^"'\`\\n]*${alone(first)}`,
    )
  }

  const SRC = join(WEB, "src")
  const sources = readdirSync(SRC, { recursive: true })
    .map((name) => String(name).replaceAll("\\", "/"))
    .filter((name) => /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name))
  // Block comments go first, for the reason the heading sweep drops them:
  // `ActivityPage` explains in a JSX comment, in backticks, why a `<th>` on
  // `text-overline` carries no `text-muted`.
  const read = (name: string) =>
    readFileSync(join(SRC, name), "utf8").replace(/\/\*[\s\S]*?\*\//g, "")

  it("covers the source tree", () => {
    // Same guard as the sweeps above: an empty list passes vacuously.
    expect(sources.length).toBeGreaterThan(30)
  })

  /**
   * The sites where 12px is the ruled size, and why the rule needed the
   * exception rather than the pages needing a change.
   *
   * The discriminator is the referent: `text-caption` is for text whose
   * referent is a CONTROL, text that helps you operate the thing it sits
   * under. Metadata, page-referent prose, and a note about an action's effect
   * rather than its operation are 12px on purpose, and an audit read every
   * `text-xs` in the tree and ruled each site individually. This rule cannot
   * express "referent is a control", so it cannot tell the two apart, and
   * making it green by converting these would reverse a per-site ruling and
   * move 33 pieces of text up a point. That is why the list is here and not
   * in the pages.
   *
   * Every entry was checked to be present verbatim at the audit's final
   * commit, so each is a site the audit saw and left, not one it never
   * reached.
   *
   * Read the site before trusting either source. The record is complete for
   * what was ruled, and a code comment carries decisions made below the
   * reporting threshold: `--text-xl`'s retune from 20 to 22 is argued at the
   * stat tile and never reached the record, because the change was reported
   * upstream as its other half.
   */
  const CAPTION_SIZE_IS_RULED: Array<[string, string]> = [
    // Metadata: identifiers, table-cell content, figures and status labels,
    // which are read rather than operated.
    [
      "features/organization/OrganizationGeneralPage.tsx",
      "the organization slug in a copyable",
    ],
    [
      "features/admin/DeploymentAccountsPage.tsx",
      "an account's email under its name, in a cell",
    ],
    [
      "features/workspaces/WorkspacesPage.tsx",
      "a workspace description truncated in a cell, and its budget helper",
    ],
    [
      "features/models/ModelsPage.tsx",
      "the selector, the family, and the not-discovered label",
    ],
    [
      "features/models/ModelScopeControl.tsx",
      "the blocked-from-every-model state banner",
    ],
    [
      "features/overview/OverviewPage.tsx",
      "an error-count figure and a chart's figcaption",
    ],
    ["shared/components/surface.tsx", "the KPI cell's severity and delta line"],
    [
      "features/organization/OrganizationMembersPage.tsx",
      "a table head row, an empty state, and fieldset prose",
    ],
    [
      "features/budgets/BudgetsPage.tsx",
      "the delete confirmation's consequence text",
    ],
    [
      "features/activity/ActivityTimeline.tsx",
      "the brush's drag hint beside the chart",
    ],
    // Page-referent: prose that describes the page rather than a control on
    // it. The auth pages' centered text was ruled this explicitly.
    ["features/auth/Login.tsx", "the divider row and the page's footer prose"],
    [
      "features/invitations/AcceptInvitationPage.tsx",
      "the centered next-step prose, page-referent",
    ],
    // The named edge: a note about what an action will do, in the row with
    // the button, rather than about how to operate a control.
    [
      "features/routing/RoutingPage.tsx",
      "consequence notes in the button row, and the candidate cap",
    ],
    ["features/usage/ShareDialog.tsx", "a notice in the dialog's button row"],
    // Excluded on its own grounds rather than the referent one: this is a
    // `<legend>` carrying `font-medium`, and `text-caption` sets
    // `font-weight: normal`, so converting it would trade this failure for
    // the one that bans a weight beside a role. A fix that moves a failure
    // sideways is not a fix.
    [
      "features/tools/WorkspaceCodeExecutionPolicyCard.tsx",
      "a legend, not a caption, and it carries a weight",
    ],
  ]
  const RULED = new Map(CAPTION_SIZE_IS_RULED)

  it("names no site that has stopped hand-rolling the caption role", () => {
    // The guard that makes the list above defensible. A path that no longer
    // offends is an entry protecting nothing, and without this it keeps
    // passing silently while the file it names has moved on.
    const stale = [...RULED.keys()].filter(
      (name) =>
        !sources.includes(name) ||
        !together("text-xs", "text-muted").test(read(name)),
    )
    expect(
      stale,
      "these are exempted from the caption rule but no longer need to be; delete the entries",
    ).toEqual([])
  })

  it.each(sources)("spells the caption role in %s as text-caption", (name) => {
    if (RULED.has(name)) return
    expect(
      read(name),
      `${name} hand-rolls the caption role as \`text-xs text-muted\`: 12px where the role is 13, plus the ink it already sets. Use text-caption`,
    ).not.toMatch(together("text-xs", "text-muted"))
  })

  it.each(sources)("overrides no type role's own metrics in %s", (name) => {
    const source = read(name)
    for (const [what, utility] of [
      ["a font size", "text-(?:xs|sm|base|lg|xl|2xl)"],
      [
        "a font weight",
        "font-(?:thin|extralight|light|normal|medium|semibold|bold|extrabold|black)",
      ],
      ["a line height", String.raw`leading-[\w./%\[\]-]+`],
      ["a tracking", String.raw`tracking-[\w./%\[\]-]+`],
    ]) {
      expect(
        source,
        `${name} puts ${what} beside a type role; the role carries its own, and the utility beside it wins with nothing to show for it`,
      ).not.toMatch(together(ANY_ROLE, utility))
    }
  })

  it.each(sources)("repeats no type role's own ink in %s", (name) => {
    const source = read(name)
    for (const [role, ink] of ROLE_INK) {
      expect(
        source,
        `${name} pairs a type role with \`${ink}\`, which that role already sets`,
      ).not.toMatch(together(role, ink))
    }
  })
})

// The chrome's own type roles, added because the nav files had grown six
// arbitrary sizes between them (13.5px twice, 13px, 11.5px, 11px, 9px), which is
// a scale with no single place to read it and nothing to keep it from growing a
// seventh. Both halves are asserted: the roles exist, and the chrome uses them.
describe("the shell chrome's type roles", () => {
  const SHELL_ROLES = [
    "text-shell-label",
    "text-shell-secondary",
    "text-shell-monogram",
  ]

  it("declares text-subtle as a utility, not a color token", () => {
    // `--color-text-subtle` would generate `text-text-subtle`, and
    // `--color-subtle` would also mint `bg-subtle` and `border-subtle`. The
    // utility form is the only one that yields this name and nothing else.
    expect(CSS).toContain("@utility text-subtle {")
    expect(
      declarations(block("@theme")).get("--color-subtle"),
      "`--color-subtle` in @theme would collide with `@utility text-subtle` in the same layer",
    ).toBeUndefined()
  })

  it.each(SHELL_ROLES)("declares %s in globals.css", (role) => {
    // The same failure the documented-utilities list guards against: a role
    // named in the scale's comment but never declared is a className that
    // produces no CSS, on text that looks merely unstyled rather than broken.
    expect(CSS).toContain(`@utility ${role} {`)
  })
})

// The scale's smallest step is 12px (`--text-xs`), so a size written at a call
// site is either a duplicate of a step or under the floor. This was scoped to
// `src/app/nav` while the chrome roles were the only ones being enforced.
describe("no font size is written at a call site", () => {
  it("leaves no arbitrary font size anywhere in the tree", () => {
    const SRC = join(WEB, "src")
    const offenders = readdirSync(SRC, { recursive: true })
      .map((name) => String(name).replaceAll("\\", "/"))
      .filter((name) => /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name))
      .filter((name) =>
        // `text-[` followed by a digit: an arbitrary size, as opposed to an
        // arbitrary color, which the token rules above already cover.
        /text-\[\d/.test(readFileSync(join(SRC, name), "utf8")),
      )

    expect(
      offenders,
      "use a type role (text-caption, text-overline, …) or a text-shell-* one, or add one",
    ).toEqual([])
  })
})

// Touch targets. The rule ("at least 44px on the phone viewport") is in the
// frontend-standards responsiveness guide and was written at ~180 `size="sm"`
// call sites that do not meet it, so it is enforced as one floor in the
// stylesheet rather than as a className each of them has to remember.
describe("the phone viewport's touch-target floor", () => {
  it("raises every HeroUI button to 44px below the shell's mobile boundary", () => {
    // `[data-slot="button"]` is HeroUI's Button and nothing else, and 767px is
    // AppShell's own MOBILE_QUERY, so the rule turns on exactly where the
    // sidebar becomes a drawer.
    expect(CSS).toMatch(
      /@media \(max-width: 767px\) \{\s*\[data-slot="button"\] \{\s*min-height: 2\.75rem;\s*min-width: 2\.75rem;/,
    )
  })
})

// Form controls. The shared `Checkbox` (`shared/components/ui.tsx`) is the one
// on the tokens, and a bare `<input type="checkbox">` is the browser's own:
// system blue in both themes, and a 13px box on a page whose smallest touch
// target is meant to be 44.
describe("checkboxes come from the design foundation", () => {
  const SRC = join(WEB, "src")
  const sources = readdirSync(SRC, { recursive: true })
    .map((name) => String(name).replaceAll("\\", "/"))
    .filter((name) => name.endsWith(".tsx") && !name.endsWith(".test.tsx"))

  it("covers the source tree", () => {
    expect(sources.length).toBeGreaterThan(30)
  })

  it.each(sources)("uses no raw checkbox in %s", (name) => {
    expect(
      readFileSync(join(SRC, name), "utf8"),
      `${name} hand-rolls a checkbox; use Checkbox from shared/components/ui`,
    ).not.toContain('type="checkbox"')
  })
})

// Column headers. A `<th>` with no `scope` leaves a screen reader guessing
// which cells it heads, and the dashboard's hand-rolled tables (the ones that
// are not `DataTable`, which gets this from react-aria) are where they live.
describe("every column header says what it heads", () => {
  const SRC = join(WEB, "src")
  const sources = readdirSync(SRC, { recursive: true })
    .map((name) => String(name).replaceAll("\\", "/"))
    .filter((name) => name.endsWith(".tsx") && !name.endsWith(".test.tsx"))

  it("covers the source tree", () => {
    expect(sources.length).toBeGreaterThan(30)
  })

  it.each(sources)("gives every <th> in %s a scope", (name) => {
    // Block comments first, for the same reason the heading sweep above drops
    // them: `ActivityPage` explains its header row in a JSX comment that spells
    // `<th>` in prose.
    const source = readFileSync(join(SRC, name), "utf8").replace(
      /\/\*[\s\S]*?\*\//g,
      "",
    )
    for (const match of source.matchAll(/<th\b([^>]*)>/g)) {
      expect(
        match[1],
        `${match[0]} in ${name} has no scope; a column header is scope="col" and a row header scope="row"`,
      ).toMatch(/\bscope=/)
    }
  })
})

// A number field's stepper is the user agent's, so it is spaced from the value
// in CSS or not at all, and the number it is spaced by has to be the one the
// trigger family already uses for a field's trailing glyph. Two places
// declaring the gap between a field's text and the thing at its inline end is
// how they come to disagree, which is what the type and radius sweeps above
// guard in their own families.
describe("a field's trailing glyph is spaced once", () => {
  const stepper = block(
    ".input::-webkit-inner-spin-button,\n.input::-webkit-outer-spin-button",
  )
  // The triggers' slot block, not the padding block above it: both carry the
  // same selector list, and `block()` is a first-match `indexOf`, so the gap is
  // read off the one that declares the layout.
  const slotGap = CSS.match(
    /\.dropdown__trigger \{[^}]*display:\s*flex;[^}]*gap:\s*([^;]+);/,
  )?.[1].trim()
  const margin = stepper.match(/margin-inline-start:\s*([^;]+);/)?.[1].trim()

  it("spaces the stepper from the value at all", () => {
    expect(margin, "nothing spaces the stepper from the value").toBe("8px")
  })

  it("spaces it by the gap the trigger family already uses", () => {
    expect(slotGap, "no gap found in the trigger slot block").toBeDefined()
    expect(margin, "the stepper's gap has drifted from the trigger's").toBe(
      slotGap,
    )
  })
})

// Three button variants, and the only thing that can hold them to three.
//
// `Button` is HeroUI's, re-exported directly from shared/components/ui, so its
// `variant` prop is typed by HeroUI's union and still accepts the four names
// this product retired. Removing our CSS for them does not make
// `variant="secondary"` a type error; it makes it a silently unstyled button,
// which is why the reduction needs a check rather than a convention.
//
// Keyed on the HOST rather than on the word, which is load-bearing: `Chip` has
// its own variant union that happens to share `secondary`, and two legitimate
// `<Chip variant="secondary">` render on the Budgets page. A scan for the word
// alone would fail on correct code the first time anyone ran it, and a gate
// that fails on correct code is one people learn to suppress.
describe("buttons come in three variants", () => {
  const SRC = join(WEB, "src")
  const RETIRED = ["outline", "secondary", "tertiary", "danger-soft"]
  const sources = readdirSync(SRC, { recursive: true })
    .map((name) => String(name).replaceAll("\\", "/"))
    .filter((name) => name.endsWith(".tsx") && !name.endsWith(".test.tsx"))

  /** Every retired variant a `Button` carries in one file, with its line. */
  function offenders(source: string): string[] {
    const found: string[] = []
    const text = source.replace(/\/\*[\s\S]*?\*\//g, "")
    for (const match of text.matchAll(/variant\s*[=:]\s*"([a-z-]+)"/g)) {
      const name = match[1]
      if (!RETIRED.includes(name)) continue
      const before = text.slice(0, match.index)
      // The nearest opening tag before the prop is the component it is on.
      const host = [...before.matchAll(/<([A-Z][A-Za-z0-9]*)/g)].pop()?.[1]
      // `buttonVariants({ variant: … })` builds a button's className without a
      // JSX host, so it is matched on the call rather than on a tag.
      const call = /buttonVariants\s*\(\s*\{[^}]*$/.test(before)
      if (host === "Button" || call) {
        const line = before.split("\n").length
        found.push(`line ${line}: ${match[0]}`)
      }
    }
    return found
  }

  it("covers the source tree", () => {
    expect(sources.length).toBeGreaterThan(30)
  })

  it.each(sources)("uses no retired button variant in %s", (name) => {
    const source = readFileSync(join(SRC, name), "utf8")
    expect(
      offenders(source),
      `${name} gives a Button a retired variant; use primary, ghost or danger`,
    ).toEqual([])
  })

  it("still reads a retired variant on a Button as an offence", () => {
    // The scan's own mutation check: this is what a reintroduced variant looks
    // like, and it has to be caught by the same code the sweep above runs.
    expect(
      offenders('<Button size="sm" variant="secondary">Go</Button>'),
    ).toEqual(['line 1: variant="secondary"'])
    expect(
      offenders('buttonVariants({ size: "sm", variant: "danger-soft" })'),
    ).toEqual(['line 1: variant: "danger-soft"'])
  })

  it("leaves another component's identically named variant alone", () => {
    // Chip's own vocabulary, which the product uses and must keep.
    expect(
      offenders('<Chip size="sm" variant="secondary">soft</Chip>'),
    ).toEqual([])
  })
})
