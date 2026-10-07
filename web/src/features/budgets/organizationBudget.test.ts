import { describe, expect, it } from "vitest"

import {
  hasNoLimit,
  limitLabel,
  scopeLabel,
  tightestUsage,
} from "@/features/budgets/organizationBudget"
import { workspace } from "@/tests/fixtures"

const caps = (
  overrides: Partial<{
    max_budget: number | null
    token_limit: number | null
    request_limit: number | null
  }> = {},
) => ({
  max_budget: null,
  token_limit: null,
  request_limit: null,
  ...overrides,
})

describe("limitLabel", () => {
  it("says a budget capping nothing is no limit, not zero", () => {
    // A budget with no cap on any axis admits every request, which is the
    // opposite of what "$0.00" reads as.
    expect(limitLabel(caps())).toBe("No limit")
    expect(hasNoLimit(caps())).toBe(true)
  })

  it("formats a real limit as money", () => {
    expect(limitLabel(caps({ max_budget: 250 }))).toContain("250")
  })

  it("keeps a zero limit distinct from an absent one", () => {
    // Zero is a real cap that refuses everything, and an operator can set it.
    expect(limitLabel(caps({ max_budget: 0 }))).not.toBe("No limit")
    expect(hasNoLimit(caps({ max_budget: 0 }))).toBe(false)
  })

  it("names a token-only cap rather than reading as unlimited", () => {
    // The case the dollar-only label got wrong: this budget refuses requests.
    expect(limitLabel(caps({ token_limit: 1_000_000 }))).toBe(
      "1,000,000 tokens",
    )
    expect(hasNoLimit(caps({ token_limit: 1_000_000 }))).toBe(false)
  })

  it("names a request-only cap", () => {
    expect(limitLabel(caps({ request_limit: 500 }))).toBe("500 requests")
  })

  it("names every cap a budget holds", () => {
    const label = limitLabel(
      caps({ max_budget: 25, token_limit: 1_000, request_limit: 10 }),
    )

    expect(label).toContain("25")
    expect(label).toContain("1,000 tokens")
    expect(label).toContain("10 requests")
  })
})

describe("scopeLabel", () => {
  const context = {
    organizationName: "Acme",
    workspaces: [workspace({ name: "Engineering" })],
  }

  it("names the organization and says it is the whole of it", () => {
    expect(
      scopeLabel(
        { scope_type: "organization", scope_id: "irrelevant" },
        context,
      ),
    ).toBe("Acme (whole organization)")
  })

  it("resolves a workspace id to its name", () => {
    expect(
      scopeLabel(
        { scope_type: "workspace", scope_id: workspace().id },
        context,
      ),
    ).toBe("Engineering (workspace)")
  })

  it("does not invent a name for a workspace it has not loaded", () => {
    // The roster read can fail or still be in flight, and a ceiling is real
    // either way. "A workspace" is less than the truth and none of it is wrong.
    expect(
      scopeLabel(
        {
          scope_type: "workspace",
          scope_id: "77777777-7777-7777-7777-777777777777",
        },
        context,
      ),
    ).toBe("A workspace")
  })

  it("names a membership or a key by kind and a short id", () => {
    // Neither has a name this page has read, and a membership id is not a
    // person's name. The kind plus enough id to match on is what it can say.
    expect(
      scopeLabel(
        {
          scope_type: "workspace_member",
          scope_id: "12345678-9999-9999-9999-999999999999",
        },
        context,
      ),
    ).toBe("A workspace member (12345678…)")
    expect(
      scopeLabel({ scope_type: "api_token", scope_id: "sk-abc" }, context),
    ).toBe("An API key (sk-abc)")
  })

  it("falls back to the raw kind for a scope a newer gateway added", () => {
    expect(
      scopeLabel({ scope_type: "something_new", scope_id: "x" }, context),
    ).toBe("something_new")
  })
})

describe("tightestUsage", () => {
  const usage = (
    overrides: Partial<{
      current_spend: number
      reserved_spend: number
      current_tokens: number
      reserved_tokens: number
      current_requests: number
      reserved_requests: number
    }> = {},
  ) => ({
    current_spend: 0,
    reserved_spend: 0,
    current_tokens: 0,
    reserved_tokens: 0,
    current_requests: 0,
    reserved_requests: 0,
    ...overrides,
  })

  it("is the entity closest to its limit, not the sum or the average", () => {
    const platform = usage({ current_spend: 440 })
    const result = tightestUsage(caps({ max_budget: 500 }), [
      usage({ current_spend: 120 }),
      platform,
      usage({ current_spend: 30 }),
    ])
    expect(result?.used).toBeCloseTo(0.88)
    expect(result?.ceiling).toBe(platform)
  })

  it("counts held spend, and whichever capped axis is furthest along", () => {
    const result = tightestUsage(caps({ max_budget: 100, request_limit: 10 }), [
      usage({ current_spend: 10, reserved_spend: 10, current_requests: 9 }),
    ])
    expect(result?.used).toBeCloseTo(0.9)
  })

  it("is over for any use of a zero cap", () => {
    expect(
      tightestUsage(caps({ max_budget: 0 }), [usage({ current_spend: 1 })])
        ?.used,
    ).toBe(Infinity)
  })

  it("has nothing to measure without a cap or without entities", () => {
    expect(tightestUsage(caps(), [usage({ current_spend: 5 })])).toBeNull()
    expect(tightestUsage(caps({ max_budget: 500 }), [])).toBeNull()
  })
})
