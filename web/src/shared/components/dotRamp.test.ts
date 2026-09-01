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
