import { describe, expect, it } from "vitest"

import type { ChargeLine, UsageEntry } from "@/client"
import { entry } from "@/tests/activity"
import {
  buildTokenComposition,
  computeToolCost,
  describeAttempt,
  describeAttemptOutcome,
  formatElapsed,
  formatLatencyCell,
  formatToolUsage,
  formatUSD,
  getActivityRowClassName,
  indexGroupOutcomes,
  listToolUsage,
  readPositive,
  resolveExtentWindow,
  resolveWindow,
  sortChargeLines,
  sortPlanRows,
} from "./activityModel"

/** A fixed clock, so a rolling preset's start is an exact instant to compare. */
const NOW = Date.parse("2026-01-15T12:00:00.000Z")

/** A row of one routing group, in the shape the writers emit per attempt. */
function attempt(overrides: Partial<UsageEntry>): UsageEntry {
  return entry({
    request_group_id: "grp-1",
    policy_name: "cheap-first",
    attempt_count: 2,
    ...overrides,
  })
}

describe("resolveWindow", () => {
  it("prefers explicit bounds over the preset", () => {
    expect(
      resolveWindow("24h", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", NOW),
    ).toEqual({ start: "2026-01-01T00:00:00Z", end: "2026-01-02T00:00:00Z" })
  })

  it("keeps a half-open range half-open", () => {
    expect(resolveWindow("24h", "2026-01-01T00:00:00Z", "", NOW)).toEqual({
      start: "2026-01-01T00:00:00Z",
      end: undefined,
    })
  })

  it("anchors a rolling preset to the clock it is handed", () => {
    expect(resolveWindow("24h", "", "", NOW)).toEqual({
      start: "2026-01-14T12:00:00.000Z",
      end: undefined,
    })
  })

  it("leaves the unbounded All preset open at both ends", () => {
    expect(resolveWindow("all", "", "", NOW)).toEqual({
      start: undefined,
      end: undefined,
    })
  })

  it("leaves an empty custom range open rather than falling back to a preset", () => {
    expect(resolveWindow("custom", "", "", NOW)).toEqual({})
  })

  it("falls back to the default preset for a range this page does not offer", () => {
    // A Usage-page key carried across in a hand-edited URL: 90d is a preset
    // there and not here, so the window has to be the Activity default.
    expect(resolveWindow("90d", "", "", NOW)).toEqual(
      resolveWindow("24h", "", "", NOW),
    )
  })
})

describe("resolveExtentWindow", () => {
  it("matches the list window for a bounded preset", () => {
    expect(resolveExtentWindow("7d", NOW)).toEqual(
      resolveWindow("7d", "", "", NOW),
    )
  })

  it("gives the unbounded All preset an explicit year-long start", () => {
    // Without it the summary endpoint applies its hidden 30-day default and the
    // bars silently show a rolling month under a caption reading "All time".
    expect(resolveExtentWindow("all", NOW)).toEqual({
      start: "2025-01-15T12:00:00.000Z",
    })
  })

  it("gives the custom sentinel the same year-long start", () => {
    expect(resolveExtentWindow("custom", NOW)).toEqual({
      start: "2025-01-15T12:00:00.000Z",
    })
  })
})

describe("readPositive", () => {
  it("passes a positive number through", () => {
    expect(readPositive(42)).toBe(42)
  })

  it("floors anything a meter could carry that is not one", () => {
    // Meters arrive as unknown JSON, so a string, a negative, a NaN or an
    // Infinity all have to read as no usage rather than reaching the arithmetic.
    for (const value of [
      0,
      -1,
      Number.NaN,
      Number.POSITIVE_INFINITY,
      "5",
      null,
      undefined,
      {},
    ]) {
      expect(readPositive(value)).toBe(0)
    }
  })
})

describe("buildTokenComposition", () => {
  it("prefers the normalized meters over the raw columns", () => {
    // The meters say the cache buckets sit inside total_input_tokens; the raw
    // columns on the same row say something else, and must not be consulted.
    const composition = buildTokenComposition(
      entry({
        prompt_tokens: 9,
        completion_tokens: 9,
        cache_read_tokens: 9,
        cache_write_tokens: 9,
        billing_meters: {
          total_input_tokens: 1000,
          cache_read_tokens: 600,
          cache_write_tokens: 100,
          completion_tokens: 200,
        },
      }),
    )
    expect(composition).toEqual({
      fresh: 300,
      cacheRead: 600,
      cacheWrite: 100,
      output: 200,
      total: 1200,
    })
  })

  it("falls back per key, not per object", () => {
    // A row can carry a tools meter while its tokens were never metered.
    // Keying off the object's presence would read every token as 0.
    const composition = buildTokenComposition(
      entry({
        prompt_tokens: 400,
        completion_tokens: 100,
        cache_read_tokens: null,
        cache_write_tokens: null,
        billing_meters: { tools: { web_search: { billed: 1 } } },
      }),
    )
    expect(composition).toEqual({
      fresh: 400,
      cacheRead: 0,
      cacheWrite: 0,
      output: 100,
      total: 500,
    })
  })

  it("clamps fresh input rather than reporting a negative split", () => {
    const composition = buildTokenComposition(
      entry({
        prompt_tokens: 100,
        completion_tokens: 0,
        cache_read_tokens: 400,
        cache_write_tokens: null,
        billing_meters: null,
      }),
    )
    expect(composition).toMatchObject({ fresh: 0, cacheRead: 400, total: 400 })
  })

  it("answers null for a row that carries no usage at all", () => {
    expect(
      buildTokenComposition(
        entry({
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
          billing_meters: null,
        }),
      ),
    ).toBeNull()
  })
})

describe("listToolUsage", () => {
  it("reads the reserved tools namespace, heaviest first", () => {
    expect(
      listToolUsage(
        entry({
          billing_meters: {
            total_input_tokens: 10,
            tools: {
              web_search: { billed: 2, errors: 1, unit_rate: 0.01 },
              code_execution: { billed: 5 },
            },
          },
        }),
      ),
    ).toEqual([
      { tool: "code_execution", billed: 5, errors: 0, unitRate: null },
      { tool: "web_search", billed: 2, errors: 1, unitRate: 0.01 },
    ])
  })

  it("drops a tool that neither billed nor failed", () => {
    expect(
      listToolUsage(
        entry({ billing_meters: { tools: { web_fetch: { billed: 0 } } } }),
      ),
    ).toEqual([])
  })

  it("is empty for a row with no meters", () => {
    expect(listToolUsage(entry({ billing_meters: null }))).toEqual([])
  })
})

describe("formatToolUsage", () => {
  it("de-underscores the name and names the failures separately", () => {
    expect(
      formatToolUsage({
        tool: "web_search",
        billed: 3,
        errors: 1,
        unitRate: null,
      }),
    ).toBe("web search ×3, 1 failed")
  })

  it("drops the count for a tool that only failed", () => {
    expect(
      formatToolUsage({
        tool: "web_fetch",
        billed: 0,
        errors: 2,
        unitRate: null,
      }),
    ).toBe("web fetch, 2 failed")
  })
})

describe("computeToolCost", () => {
  it("bills each tool at the rate stored with the row", () => {
    expect(
      computeToolCost(
        entry({
          billing_meters: {
            tools: {
              web_search: { billed: 3, unit_rate: 0.01 },
              web_fetch: { billed: 2, unit_rate: 0.005 },
            },
          },
        }),
      ),
    ).toBeCloseTo(0.04, 10)
  })

  it("excludes failed calls, which are not billed", () => {
    expect(
      computeToolCost(
        entry({
          billing_meters: {
            tools: { web_search: { billed: 1, errors: 4, unit_rate: 0.01 } },
          },
        }),
      ),
    ).toBeCloseTo(0.01, 10)
  })

  it("answers null when any billed tool has no stored rate", () => {
    // Null is "unpriced", which the detail panel says in words. Summing the
    // priced ones would report a cost that is quietly short.
    expect(
      computeToolCost(
        entry({
          billing_meters: {
            tools: {
              web_search: { billed: 3, unit_rate: 0.01 },
              code_execution: { billed: 1 },
            },
          },
        }),
      ),
    ).toBeNull()
  })

  it("answers zero for a row with no tool calls", () => {
    expect(computeToolCost(entry({ billing_meters: null }))).toBe(0)
  })
})

describe("indexGroupOutcomes", () => {
  it("indexes the served target of a group by its outcome row", () => {
    const rows = [
      attempt({ id: "a", status: "absorbed", attempt_position: 1 }),
      attempt({
        id: "b",
        status: "success",
        attempt_position: 2,
        provider: "openai",
        model: "gpt-4o-mini",
      }),
    ]
    expect(indexGroupOutcomes(rows).get("grp-1")).toEqual({
      servedBy: "openai:gpt-4o-mini",
      servedPosition: 2,
    })
  })

  it("records a terminal failure as a group with no server", () => {
    const rows = [
      attempt({ id: "a", status: "absorbed", attempt_position: 1 }),
      attempt({ id: "b", status: "error", attempt_position: 2 }),
    ]
    expect(indexGroupOutcomes(rows).get("grp-1")).toEqual({
      servedBy: null,
      servedPosition: null,
    })
  })

  it("has no entry for a group whose rows are all absorbed", () => {
    // Which is exactly the page the status=absorbed filter shows, and why the
    // page looks the missing outcomes up separately.
    const rows = [attempt({ id: "a", status: "absorbed", attempt_position: 1 })]
    expect(indexGroupOutcomes(rows).has("grp-1")).toBe(false)
  })

  it("ignores a row that belongs to no group", () => {
    expect(indexGroupOutcomes([entry({ request_group_id: null })]).size).toBe(0)
  })
})

describe("describeAttempt", () => {
  const served = { servedBy: "openai:gpt-4o", servedPosition: 2 }

  it("names the model that served in an absorbed attempt's place", () => {
    expect(
      describeAttempt(
        attempt({ status: "absorbed", attempt_position: 1 }),
        served,
      ),
    ).toBe("attempt 1 of 2 failed, served by openai:gpt-4o")
  })

  it("says the request ended in an error when the known outcome served nothing", () => {
    expect(
      describeAttempt(attempt({ status: "absorbed", attempt_position: 1 }), {
        servedBy: null,
        servedPosition: null,
      }),
    ).toBe("attempt 1 of 2 failed, and the request ended in an error")
  })

  it("says only that it fell back when the outcome is unknown", () => {
    expect(
      describeAttempt(
        attempt({ status: "absorbed", attempt_position: 1 }),
        null,
      ),
    ).toBe("attempt 1 of 2 failed, fell back")
  })

  it("distinguishes a walk that stopped early from an exhausted plan", () => {
    expect(
      describeAttempt(attempt({ status: "error", attempt_position: 1 }), null),
    ).toBe("attempt 1 of 2 failed, no further candidate tried")
    expect(
      describeAttempt(attempt({ status: "error", attempt_position: 2 }), null),
    ).toBe("attempt 2 of 2 failed, plan exhausted")
  })

  it("carries the selection reason into a successful attempt's sentence", () => {
    expect(
      describeAttempt(
        attempt({
          status: "success",
          attempt_position: 2,
          selection_reason: "on_failure",
        }),
        served,
      ),
    ).toBe("served on attempt 2 of 2 (a fallback candidate)")
  })

  it("says only why the candidate was picked when the plan has one target", () => {
    expect(
      describeAttempt(
        attempt({
          status: "success",
          attempt_count: 1,
          attempt_position: 1,
          selection_reason: "static",
        }),
        null,
      ),
    ).toBe("the policy's only target")
  })

  it("answers null for a row carrying neither a plan nor a reason", () => {
    expect(
      describeAttempt(
        attempt({ attempt_count: null, attempt_position: null }),
        null,
      ),
    ).toBeNull()
  })
})

describe("describeAttemptOutcome", () => {
  it("names the status code where the attempt has one", () => {
    expect(
      describeAttemptOutcome(attempt({ status: "absorbed", status_code: 429 })),
    ).toBe("failed 429, fell back")
    expect(
      describeAttemptOutcome(attempt({ status: "error", status_code: 500 })),
    ).toBe("failed 500")
  })

  it("omits a status code the row does not carry", () => {
    expect(
      describeAttemptOutcome(
        attempt({ status: "absorbed", status_code: null }),
      ),
    ).toBe("failed, fell back")
    expect(
      describeAttemptOutcome(attempt({ status: "error", status_code: null })),
    ).toBe("failed")
  })
})

describe("sortPlanRows", () => {
  it("orders by attempt position, not by the order it was handed", () => {
    const rows = [
      attempt({ id: "b", attempt_position: 2 }),
      attempt({ id: "a", attempt_position: 1 }),
    ]
    expect(sortPlanRows(rows).map((row) => row.id)).toEqual(["a", "b"])
  })

  it("breaks a tie on the timestamp, for rows written before the column existed", () => {
    const rows = [
      attempt({
        id: "late",
        attempt_position: null,
        timestamp: "2026-01-02T00:00:00Z",
      }),
      attempt({
        id: "early",
        attempt_position: null,
        timestamp: "2026-01-01T00:00:00Z",
      }),
    ]
    expect(sortPlanRows(rows).map((row) => row.id)).toEqual(["early", "late"])
  })

  it("leaves the rows it was given alone", () => {
    const rows = [
      attempt({ id: "b", attempt_position: 2 }),
      attempt({ id: "a", attempt_position: 1 }),
    ]
    sortPlanRows(rows)
    expect(rows.map((row) => row.id)).toEqual(["b", "a"])
  })
})

describe("sortChargeLines", () => {
  it("puts token lines before tool lines, each keeping its written order", () => {
    const lines = [
      { meter: "web_search", units: 2, unit_rate: 0.01, cost: 0.02 },
      { meter: "completion_tokens", units: 10, rate_per_million: 1, cost: 1 },
      { meter: "total_input_tokens", units: 20, rate_per_million: 1, cost: 2 },
    ] as unknown as ChargeLine[]
    expect(sortChargeLines(lines).map((line) => line.meter)).toEqual([
      "completion_tokens",
      "total_input_tokens",
      "web_search",
    ])
  })

  it("leaves the array it was given alone", () => {
    const lines = [
      { meter: "web_search", units: 1, unit_rate: 0.01, cost: 0.01 },
      { meter: "completion_tokens", units: 1, rate_per_million: 1, cost: 1 },
    ] as unknown as ChargeLine[]
    sortChargeLines(lines)
    expect(lines.map((line) => line.meter)).toEqual([
      "web_search",
      "completion_tokens",
    ])
  })
})

describe("the relocated formatters", () => {
  // These disagree with shared/helpers/format.ts on the absent-value spelling,
  // which is the disagreement otari#1335 tracks; pin it so a later swap is a
  // deliberate change rather than a silent one.
  it("spell an absent money or token value as an em dash", () => {
    expect(formatUSD(null)).toBe("—")
    expect(formatUSD(0)).toBe("$0.00")
    expect(formatLatencyCell(null)).toBe("—")
    expect(formatLatencyCell(842)).toBe("842 ms")
  })

  it("count elapsed wall-clock time in whole seconds, then minutes", () => {
    expect(formatElapsed(0)).toBe("0s")
    expect(formatElapsed(59_999)).toBe("59s")
    expect(formatElapsed(60_000)).toBe("1m 00s")
    expect(formatElapsed(125_000)).toBe("2m 05s")
  })
})

describe("getActivityRowClassName", () => {
  it("tints an absorbed attempt as caution, not as a failure", () => {
    // A routing policy recovered from it, so the request succeeded.
    expect(getActivityRowClassName(entry({ status: "absorbed" }))).toBe(
      "bg-warning-subtle",
    )
    expect(getActivityRowClassName(entry({ status: "error" }))).toBe(
      "bg-danger-subtle",
    )
    expect(
      getActivityRowClassName(entry({ status: "success" })),
    ).toBeUndefined()
  })
})
