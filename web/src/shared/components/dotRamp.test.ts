import { readdirSync, readFileSync, statSync } from "node:fs"
import { join } from "node:path"
import { describe, expect, it } from "vitest"

/**
 * A status dot takes its value off the TEXT ramp, never the surface ramp.
 *
 * The two ramps look interchangeable in a class name and are not: the surface
 * values are what sits *behind* content, so they are tuned to be nearly the
 * page. `bg-surface-subtle` on a 6px square measured 1.14:1 against the page in
 * light and 1.22:1 in dark, which is not a quiet dot, it is no dot. The right
 * word on the wrong ramp, and the same class of slip as a row separator taking
 * the section tier.
 *
 * Fourteen sites had it, in eleven files, because "subtle" was the property
 * being reached for and the ramp was incidental. This is what makes the next
 * one fail here rather than on somebody's screen.
 */
const SRC = join(import.meta.dirname, "..", "..")

function* walk(dir: string): Generator<string> {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) {
      yield* walk(full)
    } else if (/\.tsx?$/.test(entry) && !/\.test\.tsx?$/.test(entry)) {
      yield full
    }
  }
}

/**
 * `isDanger` on a row action marks the ARMED state and nothing else.
 *
 * Six sites had it at rest on the first pass, which is what a colour with one
 * job looks like when it is used as a category instead: Remove was red before
 * anybody touched it, so by the time the ink meant "this click commits" the
 * reader had stopped seeing it. An action whose confirmation is a dialog stays
 * muted throughout, because the dialog is where the danger lives.
 *
 * A source read rather than a render, for the same reason the dot rule is: what
 * is being held is which prop a call site may pass unconditionally.
 */
describe("row actions", () => {
  it("never paint danger at rest", () => {
    const offenders: string[] = []
    for (const file of walk(SRC)) {
      if (file.endsWith("surface.tsx")) continue
      const source = readFileSync(file, "utf8")
      for (const [index, line] of source.split("\n").entries()) {
        // A bare `isDanger` or one bound to something other than an armed
        // flag. `isDanger={armed...}` is the shape that is allowed.
        const bare = /\bisDanger\b(?!\s*=)/.test(line)
        const bound = /\bisDanger=\{([^}]*)\}/.exec(line)
        const armed = bound ? /armed|isArmed|pending/i.test(bound[1]) : false
        if ((bare || bound) && !armed) {
          // Inside an armed branch the prop is bare and correct, so the line
          // alone cannot decide: look for the component's own armed guard
          // within the few lines above it.
          const context = source
            .split("\n")
            .slice(Math.max(0, index - 12), index)
            .join(" ")
          if (!/armed|confirmLabel/i.test(context)) {
            offenders.push(`${file.slice(SRC.length + 1)}:${index + 1}`)
          }
        }
      }
    }
    expect(offenders).toEqual([])
  })
})

describe("status dots", () => {
  it("never take a surface-ramp value", () => {
    // A `<Dot>`'s only prop is its class, and a dot's colour is sometimes
    // assembled a few lines above it as a `dot:` field, so both spellings are
    // searched: the element, and the property that feeds it.
    const offenders: string[] = []
    for (const file of walk(SRC)) {
      const source = readFileSync(file, "utf8")
      for (const [index, line] of source.split("\n").entries()) {
        const isDot = /<Dot\b/.test(line) || /\bdot:\s*"/.test(line)
        if (isDot && /bg-(surface|background)/.test(line)) {
          offenders.push(`${file.slice(SRC.length + 1)}:${index + 1}`)
        }
      }
    }
    expect(offenders).toEqual([])
  })
})
