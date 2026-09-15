import { describe, expect, it } from "vitest"

import type { LabelableBudget } from "@/features/budgets/budgetLabel"
import {
  budgetLabel,
  budgetLabeler,
  hasBudgetName,
  shortBudgetId,
} from "@/features/budgets/budgetLabel"

function budget(overrides: Partial<LabelableBudget> = {}): LabelableBudget {
  return {
    budget_id: "abcd1234-2222-3333-4444-555555555555",
    name: null,
    max_budget: 250,
    token_limit: null,
    request_limit: null,
    budget_duration_sec: null,
    reset_alignment: "calendar_month",
    ...overrides,
  }
}

describe("hasBudgetName", () => {
  it("reads a name that is absent or blank as no name at all", () => {
    expect(hasBudgetName({ name: null })).toBe(false)
    expect(hasBudgetName({ name: "   " })).toBe(false)
    expect(hasBudgetName({ name: "Engineering" })).toBe(true)
  })
})

describe("shortBudgetId", () => {
  it("takes the head of the id", () => {
    expect(shortBudgetId("abcd1234-2222-3333-4444-555555555555")).toBe(
      "abcd1234",
    )
  })
})

describe("budgetLabel", () => {
  it("uses the name when there is one", () => {
    expect(budgetLabel(budget({ name: "Engineering monthly" }))).toBe(
      "Engineering monthly",
    )
  })

  it("trims a name rather than rendering its whitespace", () => {
    expect(budgetLabel(budget({ name: "  Engineering  " }))).toBe("Engineering")
  })

  it("names an unnamed budget by what it caps and how often", () => {
    expect(budgetLabel(budget({ name: null }))).toBe("$250.00 / month")
  })

  it("carries every axis a budget caps, not the dollar one alone", () => {
    expect(
      budgetLabel(budget({ max_budget: null, token_limit: 1_000_000 })),
    ).toBe("1,000,000 tokens / month")
  })

  it("counts a rolling period in days", () => {
    expect(
      budgetLabel(
        budget({ reset_alignment: null, budget_duration_sec: 7 * 86_400 }),
      ),
    ).toBe("$250.00 / 7 days")
  })

  it("says no period for a budget that caps nothing", () => {
    expect(
      budgetLabel(
        budget({ max_budget: null, reset_alignment: "calendar_day" }),
      ),
    ).toBe("No limit")
  })

  it("drops the period from a budget that never resets", () => {
    expect(
      budgetLabel(budget({ reset_alignment: null, budget_duration_sec: null })),
    ).toBe("$250.00")
  })
})

describe("budgetLabeler", () => {
  it("qualifies two unnamed budgets that would otherwise read alike", () => {
    const first = budget({ budget_id: "04f2f38a-1111-1111-1111-111111111111" })
    const second = budget({ budget_id: "9b71c0de-2222-2222-2222-222222222222" })
    const nameBudget = budgetLabeler([first, second])
    expect(nameBudget(first)).toBe("$250.00 / month (04f2f38a)")
    expect(nameBudget(second)).toBe("$250.00 / month (9b71c0de)")
  })

  it("leaves an unnamed budget unqualified where nothing reads like it", () => {
    const unnamed = budget({
      budget_id: "04f2f38a-1111-1111-1111-111111111111",
    })
    const nameBudget = budgetLabeler([
      unnamed,
      budget({
        budget_id: "9b71c0de-2222-2222-2222-222222222222",
        max_budget: 50,
      }),
      // A named budget never collides, whatever it caps.
      budget({
        budget_id: "c0ffee00-3333-3333-3333-333333333333",
        name: "Engineering monthly",
      }),
    ])
    expect(nameBudget(unnamed)).toBe("$250.00 / month")
  })

  it("does not qualify a budget against itself", () => {
    const only = budget()
    expect(budgetLabeler([only])(only)).toBe("$250.00 / month")
  })

  it("labels a budget the list does not carry", () => {
    const nameBudget = budgetLabeler([])
    expect(nameBudget(budget({ name: "Engineering" }))).toBe("Engineering")
    expect(nameBudget(budget())).toBe("$250.00 / month")
  })
})
