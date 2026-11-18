import { describe, expect, it } from "vitest"
import { windowTotals } from "./SessionsPage"

describe("the Sessions chart's count", () => {
  it("is the count the server read", () => {
    expect(windowTotals({ data: { count: 51 }, isError: false })).toEqual({
      count: 51,
      failed: null,
    })
  })

  it("holds while the count is on its way", () => {
    expect(windowTotals({ isError: false })).toBeNull()
  })

  // Null would hold the skeletons for good; left out, the header reads the bars.
  it("gives way to the bars when the count could not be read", () => {
    expect(windowTotals({ isError: true })).toBeUndefined()
  })
})
