import { describe, expect, it } from "vitest"

import type { OrganizationPricingOverride } from "@/client"

import {
  findOverlapping,
  isValidModelKey,
  overrideStatus,
  parseRate,
  periodBlockedReason,
  periodsOverlap,
} from "./pricingOverride"

const HOUR = 3_600_000
const NOW = Date.parse("2026-08-20T12:00:00.000Z")

function override(
  fields: Partial<OrganizationPricingOverride> = {},
): OrganizationPricingOverride {
  return {
    id: "override-1",
    organization_id: "org-1",
    model_key: "openai:gpt-4o",
    input_price_per_million: 2.5,
    output_price_per_million: 5,
    cache_read_price_per_million: null,
    cache_write_price_per_million: null,
    cache_write_1h_price_per_million: null,
    pricing_tiers: [],
    effective_from: new Date(NOW - HOUR).toISOString(),
    effective_to: null,
    created_at: new Date(NOW - HOUR).toISOString(),
    updated_at: new Date(NOW - HOUR).toISOString(),
    ...fields,
  }
}

describe("parseRate", () => {
  it("reads a non-negative number", () => {
    expect(parseRate("2.5")).toBe(2.5)
    expect(parseRate(" 0 ")).toBe(0)
  })

  // A blank optional rate means "price these tokens as fresh input", which is
  // not the same as pricing them at zero, so the two must not collapse.
  it("distinguishes a blank field from zero", () => {
    expect(parseRate("")).toBeUndefined()
    expect(parseRate("   ")).toBeUndefined()
    expect(parseRate("0")).toBe(0)
  })

  it("rejects a negative or unparseable rate", () => {
    expect(parseRate("-1")).toBeNaN()
    expect(parseRate("abc")).toBeNaN()
    expect(parseRate("Infinity")).toBeNaN()
  })
})

describe("isValidModelKey", () => {
  it("requires a provider prefix, in either separator", () => {
    expect(isValidModelKey("openai:gpt-4o")).toBe(true)
    expect(isValidModelKey("openai/gpt-4o")).toBe(true)
    expect(isValidModelKey("home_lab:llama-3")).toBe(true)
  })

  it("rejects a bare model name, which nothing would ever read back", () => {
    expect(isValidModelKey("gpt-4o")).toBe(false)
    expect(isValidModelKey("")).toBe(false)
    expect(isValidModelKey("openai:")).toBe(false)
    expect(isValidModelKey(":gpt-4o")).toBe(false)
  })
})

describe("periodBlockedReason", () => {
  it("accepts an open-ended period", () => {
    expect(periodBlockedReason("2026-08-20T12:00", "")).toBeUndefined()
  })

  it("accepts an end after the start", () => {
    expect(
      periodBlockedReason("2026-08-20T12:00", "2026-08-21T12:00"),
    ).toBeUndefined()
  })

  // Copilot caught this on #674: a present-but-unparseable value used to fall
  // through the blank check, which let Save enable, sent `null` to the server
  // (read as "starts now" / "no end"), and skipped the overlap check on the way,
  // because every comparison against NaN is false.
  it("refuses a start that is present but not a date", () => {
    expect(periodBlockedReason("not-a-date", "")).toMatch(
      /start is not a date/i,
    )
  })

  it("refuses an end that is present but not a date", () => {
    expect(periodBlockedReason("2026-08-20T12:00", "not-a-date")).toMatch(
      /end is not a date/i,
    )
  })

  it("still accepts both blank, which is the common case", () => {
    expect(periodBlockedReason("", "")).toBeUndefined()
  })

  it("refuses an inverted period", () => {
    expect(periodBlockedReason("2026-08-21T12:00", "2026-08-20T12:00")).toMatch(
      /must be after/i,
    )
  })

  // Half-open, so an end equal to the start covers no instant at all.
  it("refuses a zero-width period", () => {
    expect(periodBlockedReason("2026-08-20T12:00", "2026-08-20T12:00")).toMatch(
      /no time at all/i,
    )
  })
})

describe("periodsOverlap", () => {
  const stored = { from: NOW, to: NOW + 10 * HOUR }

  it.each([
    ["straddling the start", { from: NOW - HOUR, to: NOW + HOUR }, true],
    ["straddling the end", { from: NOW + 9 * HOUR, to: NOW + 11 * HOUR }, true],
    ["inside", { from: NOW + HOUR, to: NOW + 2 * HOUR }, true],
    ["enclosing", { from: NOW - HOUR, to: NOW + 11 * HOUR }, true],
    ["identical", { from: NOW, to: NOW + 10 * HOUR }, true],
    ["entirely before", { from: NOW - 5 * HOUR, to: NOW - HOUR }, false],
    ["entirely after", { from: NOW + 11 * HOUR, to: NOW + 12 * HOUR }, false],
  ])("%s", (_label, candidate, expected) => {
    expect(periodsOverlap(stored, candidate)).toBe(expected)
  })

  // The two boundary cases, which are the ones worth being sure about: touching
  // is not overlapping, so a rate can be retired straight into its successor.
  it("does not treat touching periods as overlapping", () => {
    expect(periodsOverlap(stored, { from: NOW - HOUR, to: NOW })).toBe(false)
    expect(
      periodsOverlap(stored, { from: NOW + 10 * HOUR, to: undefined }),
    ).toBe(false)
  })

  it("treats an open end as infinity on either side", () => {
    expect(periodsOverlap({ from: NOW, to: undefined }, stored)).toBe(true)
    expect(
      periodsOverlap(
        { from: NOW, to: undefined },
        { from: NOW - HOUR, to: NOW },
      ),
    ).toBe(false)
  })
})

describe("findOverlapping", () => {
  it("finds a clash on the same model", () => {
    const existing = [override({ effective_to: null })]

    const clash = findOverlapping(existing, {
      modelKey: "openai:gpt-4o",
      from: NOW,
      to: undefined,
    })

    expect(clash?.id).toBe("override-1")
  })

  it("ignores a different model, which may share a period freely", () => {
    const existing = [override()]

    expect(
      findOverlapping(existing, {
        modelKey: "anthropic:claude-sonnet-5",
        from: NOW,
        to: undefined,
      }),
    ).toBeUndefined()
  })

  // Without the exclusion, editing a row while keeping its period would report
  // the row clashing with itself and disable its own save.
  it("excludes the row being edited", () => {
    const existing = [override()]

    expect(
      findOverlapping(existing, {
        modelKey: "openai:gpt-4o",
        from: NOW,
        to: undefined,
        excludeId: "override-1",
      }),
    ).toBeUndefined()
  })
})

describe("overrideStatus", () => {
  it("is active inside an open-ended period", () => {
    expect(overrideStatus(override(), NOW)).toBe("active")
  })

  it("is scheduled before it starts", () => {
    const future = override({
      effective_from: new Date(NOW + HOUR).toISOString(),
    })
    expect(overrideStatus(future, NOW)).toBe("scheduled")
  })

  it("is expired once the end has passed", () => {
    const past = override({
      effective_from: new Date(NOW - 2 * HOUR).toISOString(),
      effective_to: new Date(NOW - HOUR).toISOString(),
    })
    expect(overrideStatus(past, NOW)).toBe("expired")
  })

  // The end is exclusive, so an override is already expired at the instant its
  // period ends rather than one tick later.
  it("is expired at the exact instant its period ends", () => {
    const ending = override({ effective_to: new Date(NOW).toISOString() })
    expect(overrideStatus(ending, NOW)).toBe("expired")
  })
})
