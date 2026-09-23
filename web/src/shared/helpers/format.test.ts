import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"

import { describe, expect, it } from "vitest"

import {
  deltaFraction,
  formatCost,
  formatLatency,
  formatNumber,
  formatPct,
  formatRate,
  formatRelative,
  formatReleaseDate,
  formatScore,
  formatTokens,
  formatUnitRate,
  formatUsd,
  formatUsdHeadline,
} from "@/shared/helpers/format"

describe("formatNumber", () => {
  it("groups thousands and handles nullish values", () => {
    expect(formatNumber(1234567)).toBe("1,234,567")
    expect(formatNumber(null)).toBe("0")
    expect(formatNumber(undefined)).toBe("0")
  })
})

describe("formatUsd", () => {
  it("renders compact cents for aggregate tiles", () => {
    expect(formatUsd(0)).toBe("$0.00")
    expect(formatUsd(1240.5)).toBe("$1,240.50")
    // No sub-cent precision here (unlike formatCost): tiles stay readable.
    expect(formatUsd(0.004)).toBe("$0.00")
  })
})

describe("formatUsdHeadline", () => {
  it("drops the cents once they stop carrying anything", () => {
    expect(formatUsdHeadline(2390.99)).toBe("$2,391")
    expect(formatUsdHeadline(100)).toBe("$100")
    expect(formatUsdHeadline(-2390.99)).toBe("-$2,391")
  })

  it("keeps them below $100, where they are a quarter of the number", () => {
    expect(formatUsdHeadline(99.99)).toBe("$99.99")
    expect(formatUsdHeadline(4.2)).toBe("$4.20")
  })
})

describe("formatScore", () => {
  it("keeps a vendor's scale and trims it to three places", () => {
    expect(formatScore(0.97)).toBe("0.97")
    expect(formatScore(0.123456)).toBe("0.123")
    expect(formatScore(87)).toBe("87")
  })
})

describe("formatTokens", () => {
  it("compacts at the k and M boundaries", () => {
    expect(formatTokens(512)).toBe("512")
    expect(formatTokens(999)).toBe("999")
    expect(formatTokens(1000)).toBe("1.0k")
    expect(formatTokens(84_200)).toBe("84.2k")
    expect(formatTokens(999_999)).toBe("1000.0k")
    expect(formatTokens(1_000_000)).toBe("1.0M")
    expect(formatTokens(12_400_000)).toBe("12.4M")
  })
})

describe("formatPct", () => {
  it("renders one decimal place", () => {
    expect(formatPct(0)).toBe("0.0%")
    expect(formatPct(0.021)).toBe("2.1%")
    expect(formatPct(1.25)).toBe("125.0%")
  })
})

describe("deltaFraction", () => {
  it("guards divide-by-zero and the unknown-previous case", () => {
    // No comparable baseline -> null (hides the delta), never Infinity/NaN.
    expect(deltaFraction(10, 0)).toBeNull()
    expect(deltaFraction(10, undefined)).toBeNull()
  })
  it("computes signed period-over-period change", () => {
    expect(deltaFraction(150, 100)).toBeCloseTo(0.5)
    expect(deltaFraction(80, 100)).toBeCloseTo(-0.2)
    expect(deltaFraction(100, 100)).toBe(0)
  })
})

describe("formatCost", () => {
  it("uses extra precision for sub-cent amounts", () => {
    expect(formatCost(0.0001)).toBe("$0.0001")
  })

  it("keeps four decimals above a cent too", () => {
    // The case the Activity log is read for: 2.34 cents is a real per-request
    // cost, and cents-only precision renders it as "$0.02" and drops the two
    // digits an operator opened the row to see.
    expect(formatCost(0.0234)).toBe("$0.0234")
    expect(formatCost(1.23456)).toBe("$1.2346")
  })

  it("uses two decimals for whole amounts", () => {
    expect(formatCost(12.5)).toBe("$12.50")
    expect(formatCost(null)).toBe("$0.00")
  })
})

describe("formatUnitRate", () => {
  it("falls back to significant digits below what four decimals can show", () => {
    // A per-call rate is routinely smaller than a per-million one. Without this
    // a real charge renders as "$0.0000" and reads as free.
    expect(formatUnitRate(0.00002)).toBe("$0.00002")
    expect(formatUnitRate(0.000001234)).toBe("$0.00000123")
  })

  it("uses the ordinary cost precision at or above a hundredth of a cent", () => {
    expect(formatUnitRate(0.0001)).toBe("$0.0001")
    expect(formatUnitRate(0.5)).toBe("$0.50")
  })

  it("renders zero as money rather than as significant digits", () => {
    expect(formatUnitRate(0)).toBe("$0.00")
  })
})

describe("formatLatency", () => {
  it("reads sub-second durations in whole milliseconds", () => {
    expect(formatLatency(820)).toBe("820 ms")
    expect(formatLatency(820.4)).toBe("820 ms")
  })

  it("switches to seconds at a thousand, with two decimals throughout", () => {
    expect(formatLatency(1000)).toBe("1.00 s")
    expect(formatLatency(15_000)).toBe("15.00 s")
  })

  it("returns undefined rather than a placeholder when nothing was recorded", () => {
    // Each surface decides: a table cell renders the em dash that keeps the
    // column aligned, a stat card drops the figure instead.
    expect(formatLatency(null)).toBeUndefined()
    expect(formatLatency(undefined)).toBeUndefined()
  })
})

describe("formatRate", () => {
  it("keeps the fourth decimal a published rate can carry", () => {
    expect(formatRate(0.075)).toBe("$0.075")
    expect(formatRate(0.0125)).toBe("$0.0125")
    expect(formatRate(0.037)).toBe("$0.037")
  })

  it("still reads a whole-dollar rate as money", () => {
    expect(formatRate(3)).toBe("$3.00")
    expect(formatRate(15)).toBe("$15.00")
  })
})

describe("formatReleaseDate", () => {
  it("renders a compact month and year without timezone drift", () => {
    expect(formatReleaseDate("2024-05-13")).toBe("May 2024")
    expect(formatReleaseDate("2025-01-01")).toBe("Jan 2025")
    expect(formatReleaseDate("2023-12")).toBe("Dec 2023")
  })

  it("falls back gracefully for missing or unparseable values", () => {
    expect(formatReleaseDate(null)).toBe("—")
    expect(formatReleaseDate(undefined)).toBe("—")
    expect(formatReleaseDate("someday")).toBe("someday")
    expect(formatReleaseDate("2024-13")).toBe("2024")
  })
})

describe("formatRelative", () => {
  const now = Date.parse("2026-01-01T12:00:00Z")

  it("describes past timestamps", () => {
    // Compact is the product's voice: "30s ago", not "30 seconds ago". The
    // difference is 10px of table lane, which is why the copy is the fix rather
    // than the column width.
    expect(formatRelative("2026-01-01T11:59:30Z", now)).toBe("30s ago")
    expect(formatRelative("2026-01-01T10:00:00Z", now)).toBe("2h ago")
    expect(formatRelative("2025-12-30T12:00:00Z", now)).toBe("2d ago")
    // A clock a little ahead of the server reads as the present rather than as
    // a request that has not happened yet.
    expect(formatRelative("2026-01-01T12:00:05Z", now)).toBe("just now")
  })

  /** An ISO timestamp `days` before `now`, so a boundary reads as its number. */
  const daysAgo = (days: number) =>
    new Date(now - days * 86_400_000).toISOString()

  it("steps up to months and years rather than counting days forever", () => {
    // `formatRelative` runs on long-lived rows too: a claimed email domain, an
    // account's last sign-in, a price's last edit. Without these buckets a
    // domain claimed a year and a half ago read as "548d ago".
    expect(formatRelative(daysAgo(29), now)).toBe("29d ago")
    expect(formatRelative(daysAgo(30), now)).toBe("1mo ago")
    expect(formatRelative(daysAgo(45), now)).toBe("1mo ago")
    expect(formatRelative(daysAgo(60), now)).toBe("2mo ago")
    expect(formatRelative(daysAgo(359), now)).toBe("11mo ago")
    expect(formatRelative(daysAgo(360), now)).toBe("1y ago")
    expect(formatRelative(daysAgo(548), now)).toBe("1y ago")
    expect(formatRelative(daysAgo(730), now)).toBe("2y ago")
  })

  it("returns 'never' for missing timestamps", () => {
    expect(formatRelative(null, now)).toBe("never")
  })
})

/**
 * The locale sweep, over the whole of `src`.
 *
 * The cases above are the formatters this module owns. This one is about the
 * ones it does not: a page that builds its own `Intl` formatter, or calls
 * `toLocaleString()` on a number, renders a figure in whatever locale the
 * browser reports while the words around it stay English. Three pages had done
 * exactly that, so the same session showed a spend as "1.234,56 $" on one page
 * and "$1,234.56" on another.
 *
 * Unit cases cannot catch it, because the test runner's own locale is the one
 * that makes the wrong code look right. Reading the source can.
 *
 * Dates are deliberately out of scope. `toLocaleDateString` and
 * `toLocaleTimeString` are still unpinned here and in `formatDateTime` itself,
 * and whether a timestamp should read as US or as the reader's is a decision
 * nobody has taken. Numbers and money are decided: the dashboard bills in USD.
 */
describe("formatters are locale-pinned", () => {
  const SRC = join(process.cwd(), "src")

  function sourceFiles(dir: string): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const full = join(dir, entry.name)
      if (entry.isDirectory()) return sourceFiles(full)
      if (!/\.tsx?$/.test(entry.name)) return []
      if (entry.name.includes(".test.") || entry.name.includes(".stories."))
        return []
      return [full]
    })
  }

  /** Relative to `src`, so a failure names the file the way an import does. */
  function relative(file: string): string {
    return file.slice(SRC.length + 1)
  }

  it("covers the source tree", () => {
    // A guard on the guard: a wrong root finds no files and passes.
    expect(sourceFiles(SRC).length).toBeGreaterThan(100)
  })

  it("pins every Intl formatter to en-US", () => {
    // `new` is optional because these are callable as factories, and the first
    // argument is matched even when empty: `new Intl.NumberFormat()` takes the
    // runtime locale exactly as `(undefined)` does, and the first version of
    // this pattern required both and so waved each of them through.
    const offenders = sourceFiles(SRC).flatMap((file) => {
      const source = readFileSync(file, "utf8")
      return [...source.matchAll(/(?:new\s+)?Intl\.\w+\(\s*([^,)]*)/g)]
        .filter((match) => match[1].trim() !== '"en-US"')
        .map((match) => `${relative(file)}: Intl.…(${match[1].trim()})`)
    })
    expect(offenders).toEqual([])
  })

  it("formats through the helper rather than through the browser", () => {
    // A bare `toLocaleString()` is the same bug in one call: it is the
    // browser's locale, not ours. `formatNumber` is the replacement for a
    // number, and it lives in `design-system/helpers` so that layer can reach
    // it too. The pattern cannot see what it was called on, so the files below
    // are listed rather than matched: each one is a date, which this sweep
    // deliberately does not decide.
    const offenders = sourceFiles(SRC).flatMap((file) => {
      const source = readFileSync(file, "utf8")
      return [...source.matchAll(/\.toLocaleString\(\s*\)/g)].map(() =>
        relative(file),
      )
    })
    expect([...new Set(offenders)].sort()).toEqual([
      // The two date call sites this sweep does not decide. Both render a
      // timestamp, where the open question is US versus the reader's own, and
      // both are a `new Date(...)` rather than a number.
      "features/keys/KeysPage.tsx",
      "features/overview/OverviewPage.tsx",
      "shared/helpers/format.ts",
    ])
  })
})
