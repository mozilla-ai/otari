import { describe, expect, it } from "vitest"
import { formatDuration } from "@/features/traces/format"

describe("formatDuration", () => {
  it("keeps the latency format under a minute", () => {
    expect(formatDuration(850)).toBe("850 ms")
    expect(formatDuration(21_310)).toBe("21.31 s")
  })

  it("counts minutes and seconds under an hour", () => {
    expect(formatDuration(125_000)).toBe("2m 05s")
    expect(formatDuration(59 * 60_000 + 59_000)).toBe("59m 59s")
  })

  it("counts hours and minutes from an hour up, so a long session stays short", () => {
    expect(formatDuration(3_600_000)).toBe("1h 00m")
    expect(formatDuration((1325 * 60 + 40) * 1000)).toBe("22h 06m")
  })
})
