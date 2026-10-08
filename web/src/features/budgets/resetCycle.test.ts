import { describe, expect, it } from "vitest"

import {
  type CycleDraft,
  cycleDraftFrom,
  cycleFields,
  cycleFieldsFromDraft,
  cycleLabel,
  cycleValue,
  findCycleProblem,
  maskFromWeekdays,
  ordinal,
  type ReadCycleFields,
  shortCycleLabel,
  weekdaysFromMask,
} from "@/features/budgets/resetCycle"

function cycle(overrides: Partial<ReadCycleFields> = {}): ReadCycleFields {
  return {
    reset_cycle: null,
    reset_every_n: null,
    reset_anchor_at: null,
    reset_weekdays: null,
    reset_month_day: null,
    reset_month: null,
    ...overrides,
  }
}

function draft(overrides: Partial<CycleDraft> = {}): CycleDraft {
  return {
    cycle: "never",
    everyN: "",
    anchorAt: "2026-10-05",
    storedAnchorAt: null,
    weekdayBits: [],
    monthDay: "1",
    month: "1",
    ...overrides,
  }
}

const MONDAY = 0
const FRIDAY = 4

describe("weekday masks", () => {
  it("round trips a set of weekdays", () => {
    expect(weekdaysFromMask(maskFromWeekdays([MONDAY, FRIDAY]))).toEqual([
      MONDAY,
      FRIDAY,
    ])
  })

  it("reads no weekdays out of an empty mask", () => {
    expect(weekdaysFromMask(0)).toEqual([])
    expect(weekdaysFromMask(null)).toEqual([])
  })
})

describe("ordinal", () => {
  it("names the days an operator picks from", () => {
    expect([1, 2, 3, 4, 21, 22, 23, 28].map(ordinal)).toEqual([
      "1st",
      "2nd",
      "3rd",
      "4th",
      "21st",
      "22nd",
      "23rd",
      "28th",
    ])
  })

  it("gets the teens right, which the last-digit rule does not", () => {
    expect([11, 12, 13].map(ordinal)).toEqual(["11th", "12th", "13th"])
  })
})

describe("cycleLabel", () => {
  it("says never when a budget carries no cycle", () => {
    expect(cycleLabel(cycle())).toBe("Never")
  })

  it("names each calendar cycle with the instant it resets on", () => {
    expect(cycleLabel(cycle({ reset_cycle: "daily" }))).toBe(
      "Daily, at 00:00 UTC",
    )
    expect(
      cycleLabel(cycle({ reset_cycle: "monthly", reset_month_day: 15 })),
    ).toBe("Monthly, the 15th at 00:00 UTC")
    expect(
      cycleLabel(
        cycle({ reset_cycle: "yearly", reset_month: 3, reset_month_day: 1 }),
      ),
    ).toBe("Yearly, March 1st (UTC)")
  })

  it("lists a weekly cycle's days the way a person reads a list", () => {
    expect(
      cycleLabel(
        cycle({
          reset_cycle: "weekly",
          reset_weekdays: maskFromWeekdays([MONDAY, FRIDAY]),
        }),
      ),
    ).toBe("Every Monday and Friday (UTC)")
    expect(
      cycleLabel(
        cycle({
          reset_cycle: "weekly",
          reset_weekdays: maskFromWeekdays([MONDAY, 2, FRIDAY]),
        }),
      ),
    ).toBe("Every Monday, Wednesday and Friday (UTC)")
  })

  it("counts an interval cycle in its own unit", () => {
    expect(
      cycleLabel(cycle({ reset_cycle: "every_n_hours", reset_every_n: 6 })),
    ).toBe("Every 6 hours")
    expect(
      cycleLabel(cycle({ reset_cycle: "every_n_hours", reset_every_n: 1 })),
    ).toBe("Every 1 hour")
    expect(
      cycleLabel(cycle({ reset_cycle: "every_n_days", reset_every_n: 14 })),
    ).toBe("Every 14 days")
  })

  it("shows a cycle a newer gateway wrote rather than calling it never", () => {
    // "Never" would be a lie about a budget that does reset, and this label is
    // read by someone deciding whether a refusal was due.
    expect(cycleLabel(cycle({ reset_cycle: "fortnightly" }))).toBe(
      "fortnightly",
    )
  })
})

describe("shortCycleLabel", () => {
  it("has no unit for a budget that never resets", () => {
    expect(shortCycleLabel(cycle())).toBeUndefined()
  })

  it("names a single-day week as a week and several as a count", () => {
    expect(
      shortCycleLabel(
        cycle({
          reset_cycle: "weekly",
          reset_weekdays: maskFromWeekdays([MONDAY]),
        }),
      ),
    ).toBe("week")
    // Several days are periods of different lengths, so no single unit is true.
    expect(
      shortCycleLabel(
        cycle({
          reset_cycle: "weekly",
          reset_weekdays: maskFromWeekdays([MONDAY, FRIDAY]),
        }),
      ),
    ).toBe("2x a week")
  })

  it("reads as a bare unit beside a figure", () => {
    expect(shortCycleLabel(cycle({ reset_cycle: "monthly" }))).toBe("month")
    expect(
      shortCycleLabel(
        cycle({ reset_cycle: "every_n_hours", reset_every_n: 1 }),
      ),
    ).toBe("hour")
    expect(
      shortCycleLabel(
        cycle({ reset_cycle: "every_n_days", reset_every_n: 14 }),
      ),
    ).toBe("14 days")
  })
})

describe("cycleFields", () => {
  it("carries only the settings its own cycle takes", () => {
    expect(cycleFields("weekly", { weekdayBits: [MONDAY, FRIDAY] })).toEqual({
      reset_cycle: "weekly",
      reset_every_n: null,
      reset_anchor_at: null,
      reset_weekdays: maskFromWeekdays([MONDAY, FRIDAY]),
      reset_month_day: null,
      reset_month: null,
    })
  })

  it("clears the previous cycle's settings rather than stranding them", () => {
    // The bug this exists to stop: switching a weekly budget to monthly while
    // leaving the weekday mask behind is two valid-looking fields and a row the
    // server refuses.
    const monthly = cycleFields("monthly", {
      monthDay: 15,
      weekdayBits: [MONDAY, FRIDAY],
    })
    expect(monthly.reset_weekdays).toBeNull()
    expect(monthly.reset_month_day).toBe(15)
  })

  it("sends nothing but nulls for never", () => {
    expect(cycleFields("never", { everyN: 6, monthDay: 15 })).toEqual({
      reset_cycle: null,
      reset_every_n: null,
      reset_anchor_at: null,
      reset_weekdays: null,
      reset_month_day: null,
      reset_month: null,
    })
  })

  it("treats an empty weekday set as no mask rather than as zero", () => {
    expect(cycleFields("weekly", { weekdayBits: [] }).reset_weekdays).toBeNull()
  })
})

describe("cycleFieldsFromDraft", () => {
  it("anchors at midnight UTC, not at the browser's local time", () => {
    const fields = cycleFieldsFromDraft(
      draft({ cycle: "every_n_hours", everyN: "6", anchorAt: "2026-10-05" }),
    )
    expect(fields.reset_anchor_at).toBe("2026-10-05T00:00:00Z")
    expect(fields.reset_every_n).toBe(6)
  })

  it("sends a stored anchor back unchanged while its date is untouched", () => {
    // A migrated budget's anchor carries a time of day; truncating it to
    // midnight on a rename would move the phase and retime every ceiling.
    const stored = "2026-10-05T14:37:12+00:00"
    const opened = cycleDraftFrom(
      cycle({
        reset_cycle: "every_n_hours",
        reset_every_n: 6,
        reset_anchor_at: stored,
      }),
    )
    expect(cycleFieldsFromDraft(opened).reset_anchor_at).toBe(stored)
    expect(
      cycleFieldsFromDraft({ ...opened, anchorAt: "2026-10-06" })
        .reset_anchor_at,
    ).toBe("2026-10-06T00:00:00Z")
  })

  it("reads a stored anchor with no offset as UTC", () => {
    const opened = cycleDraftFrom(
      cycle({
        reset_cycle: "every_n_days",
        reset_every_n: 2,
        reset_anchor_at: "2026-10-05T23:30:00",
      }),
    )
    expect(opened.anchorAt).toBe("2026-10-05")
  })

  it("sends no interval for a half-typed number", () => {
    expect(
      cycleFieldsFromDraft(draft({ cycle: "every_n_days", everyN: "" }))
        .reset_every_n,
    ).toBeNull()
  })
})

describe("cycleDraftFrom and cycleValue", () => {
  it("opens a new budget on monthly, not on never", () => {
    // A new budget defaulting to "never" is a cap that never turns over, which
    // is not what someone creating one means.
    expect(cycleValue(undefined)).toBe("monthly")
    expect(cycleDraftFrom(undefined).cycle).toBe("monthly")
  })

  it("keeps a stored budget that resets on nothing as never", () => {
    expect(cycleValue(cycle())).toBe("never")
  })

  it("opens a stored budget on its own cycle and settings", () => {
    const opened = cycleDraftFrom(
      cycle({
        reset_cycle: "weekly",
        reset_weekdays: maskFromWeekdays([MONDAY, FRIDAY]),
      }),
    )
    expect(opened.cycle).toBe("weekly")
    expect(opened.weekdayBits).toEqual(["0", "4"])
  })

  it("takes the date out of a stored anchor, dropping its time", () => {
    const opened = cycleDraftFrom(
      cycle({
        reset_cycle: "every_n_days",
        reset_every_n: 14,
        reset_anchor_at: "2026-09-24T00:00:00+00:00",
      }),
    )
    expect(opened.anchorAt).toBe("2026-09-24")
    expect(opened.everyN).toBe("14")
  })
})

describe("findCycleProblem", () => {
  it("passes a cycle that needs no settings", () => {
    expect(findCycleProblem(draft())).toBeUndefined()
    expect(findCycleProblem(draft({ cycle: "daily" }))).toBeUndefined()
  })

  it("refuses a weekly cycle naming no day", () => {
    expect(findCycleProblem(draft({ cycle: "weekly" }))).toEqual({
      field: "weekdays",
      message: "Pick at least one weekday.",
    })
  })

  it("refuses an interval that is not a whole number of one or more", () => {
    for (const everyN of ["", "0", "-3", "abc", "1.5"]) {
      expect(
        findCycleProblem(draft({ cycle: "every_n_hours", everyN }))?.field,
      ).toBe("everyN")
    }
  })

  it("refuses an interval with no start date", () => {
    expect(
      findCycleProblem(
        draft({ cycle: "every_n_days", everyN: "14", anchorAt: "" }),
      ),
    ).toEqual({
      field: "anchorAt",
      message: "Pick the date this cycle starts from.",
    })
  })

  it("refuses an interval past the server's ceiling", () => {
    expect(
      findCycleProblem(draft({ cycle: "every_n_days", everyN: "3651" })),
    ).toEqual({ field: "everyN", message: "Enter 3650 or fewer." })
    expect(
      findCycleProblem(draft({ cycle: "every_n_days", everyN: "3650" })),
    ).toBeUndefined()
  })

  it("passes a complete interval", () => {
    expect(
      findCycleProblem(draft({ cycle: "every_n_hours", everyN: "6" })),
    ).toBeUndefined()
  })
})
