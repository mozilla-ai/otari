import { describe, expect, it } from "vitest"

import {
  setupFailureCopy,
  UNKNOWN_FAILURE,
} from "@/features/onboarding/setupFailureCopy"

describe("setupFailureCopy", () => {
  it("names the cause and the page that fixes it, per category", () => {
    expect(setupFailureCopy("policy")).toEqual({
      cause:
        "A budget, a model allow-list, or a rate limit rejected the request.",
      hint: { label: "Open budgets", to: "/budgets" },
    })
  })

  it("offers no next step for a provider outage, which nothing here fixes", () => {
    expect(setupFailureCopy("upstream").hint).toBeUndefined()
    expect(setupFailureCopy("timeout").hint).toBeUndefined()
  })

  it("falls back when the category is absent or one this dashboard does not know", () => {
    expect(setupFailureCopy(null)).toBe(UNKNOWN_FAILURE)
    expect(setupFailureCopy(undefined)).toBe(UNKNOWN_FAILURE)
    // A category a newer gateway sends and this build has no copy for. Cast
    // because the generated union is exactly the set the current gateway sends,
    // which is the situation being tested.
    expect(
      setupFailureCopy(
        "something_new" as Parameters<typeof setupFailureCopy>[0],
      ),
    ).toBe(UNKNOWN_FAILURE)
  })

  it("says nothing about the provider's own error text", () => {
    // The gateway sends a category and never the upstream payload, so the copy
    // here is the whole of what an operator reads. A sentence that promised
    // details would be promising something that never arrives.
    for (const category of [
      "invalid_request",
      "configuration",
      "policy",
      "upstream",
      "timeout",
      "internal",
    ] as const) {
      expect(setupFailureCopy(category).cause).not.toMatch(
        /error message|detail/i,
      )
    }
  })
})
