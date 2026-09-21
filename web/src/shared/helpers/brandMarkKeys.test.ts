import { describe, expect, it } from "vitest"
import { MARK_KEYS_FOR_TEST } from "@/shared/helpers/brandMarkKeys"
import {
  MARK_TABLE_KEYS_FOR_TEST,
  makerMark,
  providerMark,
} from "@/shared/helpers/brandMarks"

// The cost of splitting the key lists from the geometry they mirror: the two can
// drift, and a drift is invisible on the page. A key with no glyph renders a
// reserved box that never fills, which reads as a mark that failed to load; a
// glyph with no key is simply never asked for.
//
// Deliberately asserted in both directions, so neither failure mode can ship.
describe("the mark key lists against the geometry tables", () => {
  it("has a glyph for every provider id it lists", () => {
    const missing = [...MARK_KEYS_FOR_TEST.providers].filter(
      (id) => providerMark(id) === undefined,
    )

    expect(missing).toEqual([])
  })

  it("has a glyph for every maker slug it lists", () => {
    const missing = [...MARK_KEYS_FOR_TEST.makers].filter(
      (slug) => makerMark(slug) === undefined,
    )

    expect(missing).toEqual([])
  })

  it("lists exactly the keys the geometry tables hold", () => {
    // The other direction, compared against the tables themselves rather than
    // against a count somebody typed: a glyph added without its key fails here,
    // where the forward tests above cannot see it.
    expect([...MARK_KEYS_FOR_TEST.providers].sort()).toEqual(
      [...MARK_TABLE_KEYS_FOR_TEST.providers].sort(),
    )
    expect([...MARK_KEYS_FOR_TEST.makers].sort()).toEqual(
      [...MARK_TABLE_KEYS_FOR_TEST.makers].sort(),
    )
  })
})
