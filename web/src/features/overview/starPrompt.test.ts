import { afterEach, describe, expect, it, vi } from "vitest"

import type { UsageSummary } from "@/client"
import {
  hasEarnedStarPrompt,
  mayOfferStarPrompt,
  retireStarPrompt,
  STAR_PROMPT_STORE_KEY,
} from "@/features/overview/starPrompt"
import { seriesPoint, usageTotals } from "@/tests/fixtures"

/** A window of successes: one entry per day bucket, the count on each. */
function successes(perDay: number[]): UsageSummary {
  return {
    start_date: "2026-10-04T00:00:00Z",
    end_date: "2026-11-02T00:00:00Z",
    bucket: "day",
    totals: usageTotals({
      request_count: perDay.reduce((sum, count) => sum + count, 0),
    }),
    by_model: [],
    by_user: [],
    by_api_key: [],
    by_source: [],
    by_source_label: [],
    by_endpoint: [],
    by_provider: [],
    by_tool: [],
    by_tag: [],
    errors_by_status_code: [],
    series: perDay.map((requests, day) =>
      seriesPoint({
        bucket_start: `2026-10-${String(day + 4).padStart(2, "0")}T00:00:00Z`,
        requests,
      }),
    ),
  }
}

describe("when the invitation to star Otari has been earned", () => {
  it("asks nothing before the usage has been read", () => {
    expect(hasEarnedStarPrompt(undefined)).toBe(false)
  })

  it("asks once 25 successes span two days", () => {
    expect(hasEarnedStarPrompt(successes([20, 0, 5]))).toBe(true)
  })

  it("does not ask on one busy afternoon, however busy", () => {
    // A coding agent can make this many calls for one prompt. Coming back is
    // the signal, not the count.
    expect(hasEarnedStarPrompt(successes([400]))).toBe(false)
  })

  it("does not ask before 25, however many days", () => {
    expect(hasEarnedStarPrompt(successes([4, 4, 4, 4, 4, 4]))).toBe(false)
  })

  it("does not count an empty day bucket as a day of use", () => {
    expect(hasEarnedStarPrompt(successes([25, 0, 0]))).toBe(false)
  })
})

describe("what a browser remembers about the invitation", () => {
  afterEach(() => {
    window.localStorage.removeItem(STAR_PROMPT_STORE_KEY)
    vi.restoreAllMocks()
  })

  it("may ask a browser that has not answered", () => {
    expect(mayOfferStarPrompt()).toBe(true)
  })

  it("never asks again once either answer is recorded", () => {
    retireStarPrompt("dismissed")
    expect(mayOfferStarPrompt()).toBe(false)
    expect(window.localStorage.getItem(STAR_PROMPT_STORE_KEY)).toBe("dismissed")
  })

  it("does not ask a browser that could not remember the answer", () => {
    // An invitation that cannot be put away would return on every visit.
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError")
    })
    expect(mayOfferStarPrompt()).toBe(false)
  })

  it("keeps going when the store refuses the answer", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("full", "QuotaExceededError")
    })
    expect(() => retireStarPrompt("starred")).not.toThrow()
  })
})
