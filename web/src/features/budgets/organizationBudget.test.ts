import { describe, expect, it } from "vitest"

import type { OrganizationBudget } from "@/client"
import {
  budgetLabel,
  hasNoLimit,
  limitLabel,
  PERIOD_OPTIONS,
  periodLabel,
  periodValue,
  scopeLabel,
} from "@/features/budgets/organizationBudget"
import { workspace } from "@/tests/fixtures"

function budget(
  overrides: Partial<OrganizationBudget> = {},
): OrganizationBudget {
  return {
    budget_id: "abcd1234-2222-3333-4444-555555555555",
    organization_id: "11111111-1111-1111-1111-111111111111",
    name: null,
    max_budget: 250,
    token_limit: null,
    request_limit: null,
    budget_duration_sec: null,
    reset_alignment: "calendar_month",
    ceiling_count: 0,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    ...overrides,
  }
}

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

describe("periodLabel", () => {
  it("names each calendar boundary with the instant it resets on", () => {
    expect(
      periodLabel({
        reset_alignment: "calendar_day",
        budget_duration_sec: null,
      }),
    ).toContain("UTC midnight")
    expect(
      periodLabel({
        reset_alignment: "calendar_week",
        budget_duration_sec: null,
      }),
    ).toContain("Monday")
    expect(
      periodLabel({
        reset_alignment: "calendar_month",
        budget_duration_sec: null,
      }),
    ).toContain("1st")
  })

  it("says a duration is measured from the last reset, not from a clock time", () => {
    // The distinction the page exists to keep honest: 86400 does not mean
    // "resets at midnight", it means "at least 24 hours, restarted on next use".
    const label = periodLabel({
      reset_alignment: null,
      budget_duration_sec: 86_400,
    })
    expect(label).toContain("1 day")
    expect(label).toContain("from the last reset")
  })

  it("falls back to seconds for a duration that is not whole hours", () => {
    expect(
      periodLabel({ reset_alignment: null, budget_duration_sec: 90 }),
    ).toBe("Every 90s, from the last reset")
  })

  it("says never when a budget carries no period at all", () => {
    expect(
      periodLabel({ reset_alignment: null, budget_duration_sec: null }),
    ).toBe("Never")
  })

  it("prefers the alignment when a row somehow carries both", () => {
    // The database refuses the pair, so this is unreachable through either
    // surface; asserted because the alignment is the one that actually governs
    // the reset in `_aligned_window`, so a row read from a hand-edited database
    // should not be described by the field that is ignored.
    expect(
      periodLabel({
        reset_alignment: "calendar_month",
        budget_duration_sec: 86_400,
      }),
    ).toContain("1st")
  })
})

describe("periodValue", () => {
  it("opens the form on the budget's own boundary", () => {
    expect(periodValue(budget({ reset_alignment: "calendar_day" }))).toBe(
      "calendar_day",
    )
  })

  it("defaults a new budget to monthly", () => {
    expect(periodValue(undefined)).toBe("calendar_month")
  })

  it("opens on no reset for a duration the form cannot express", () => {
    // Rather than proposing a boundary the budget never had. The dialog warns
    // that saving replaces the duration, which is the honest version of this.
    expect(
      periodValue(
        budget({ reset_alignment: null, budget_duration_sec: 86_400 }),
      ),
    ).toBe("none")
  })

  it("only ever returns a value the picker offers", () => {
    const offered = PERIOD_OPTIONS.map((option) => option.value)
    for (const alignment of [
      "calendar_day",
      "calendar_week",
      "calendar_month",
    ]) {
      expect(offered).toContain(
        periodValue(budget({ reset_alignment: alignment })),
      )
    }
    expect(offered).toContain(periodValue(undefined))
  })
})

describe("budgetLabel", () => {
  it("uses the name when there is one", () => {
    expect(budgetLabel(budget({ name: "Engineering monthly" }))).toBe(
      "Engineering monthly",
    )
  })

  it("falls back to the head of the id, which is what an operator can match on", () => {
    expect(budgetLabel(budget({ name: null }))).toBe("abcd1234")
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
