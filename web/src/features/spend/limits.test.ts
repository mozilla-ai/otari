import { describe, expect, it } from "vitest"
import type { OrganizationMember, OrganizationSpendCeiling } from "@/client"
import {
  limitLine,
  limitsOn,
  limitsOnPerson,
  NO_LIMITS,
  perWords,
  usedShare,
  wholePct,
} from "@/features/spend/limits"

function ceiling(
  overrides: Partial<OrganizationSpendCeiling>,
): OrganizationSpendCeiling {
  return {
    id: "c",
    budget_id: "b",
    scope_type: "org_member",
    scope_id: "m1",
    max_budget: 100,
    current_spend: 0,
    provider_key_id: null,
    reset_alignment: null,
    budget_duration_sec: null,
    ...overrides,
  } as OrganizationSpendCeiling
}

const member = {
  organization_member_id: "m1",
  full_name: "Wei Zhang",
  workspaces: [
    {
      workspace_id: "w1",
      workspace_member_id: "wm1",
      workspace_name: "Platform",
    },
    { workspace_id: "w2", workspace_member_id: "wm2", workspace_name: "Data" },
  ],
} as unknown as OrganizationMember

describe("perWords", () => {
  it("reads a calendar boundary as a word", () => {
    expect(perWords(ceiling({ reset_alignment: "calendar_week" }))).toBe(
      "a week",
    )
    expect(perWords(ceiling({ reset_alignment: "calendar_month" }))).toBe(
      "a month",
    )
  })

  it("reads a duration under a day in hours", () => {
    expect(perWords(ceiling({ budget_duration_sec: 3_600 }))).toBe("an hour")
    expect(perWords(ceiling({ budget_duration_sec: 6 * 3_600 }))).toBe(
      "every 6 hours",
    )
  })

  it("reads a rolling duration by its length", () => {
    expect(perWords(ceiling({ budget_duration_sec: 86_400 }))).toBe("a day")
    expect(perWords(ceiling({ budget_duration_sec: 7 * 86_400 }))).toBe(
      "a week",
    )
    expect(perWords(ceiling({ budget_duration_sec: 30 * 86_400 }))).toBe(
      "a month",
    )
    expect(perWords(ceiling({ budget_duration_sec: 10 * 86_400 }))).toBe(
      "every 10 days",
    )
  })

  it("says nothing for a limit that never resets", () => {
    expect(perWords(ceiling({}))).toBe("")
  })
})

describe("limitsOn", () => {
  it("finds a scope's own limit", () => {
    const limits = limitsOn(
      [ceiling({ scope_type: "api_token", scope_id: "k1", current_spend: 2 })],
      "api_token",
      "k1",
    )
    expect(limits.count).toBe(1)
    expect(limits.own?.spent).toBe(2)
    expect(limits.binding).toBe(limits.own)
  })

  it("has none for another scope, a missing id, or no cap", () => {
    const list = [ceiling({ scope_type: "api_token", scope_id: "k1" })]
    expect(limitsOn(list, "api_token", "k2")).toBe(NO_LIMITS)
    expect(limitsOn(list, "workspace", "k1")).toBe(NO_LIMITS)
    expect(limitsOn(list, "api_token", undefined)).toBe(NO_LIMITS)
    expect(limitsOn([ceiling({ max_budget: null })], "org_member", "m1")).toBe(
      NO_LIMITS,
    )
  })

  it("reads a $0 limit as a limit that is used up", () => {
    const limits = limitsOn([ceiling({ max_budget: 0 })], "org_member", "m1")
    expect(limits.own?.max).toBe(0)
    expect(limits.binding).toBe(limits.own)
    expect(usedShare({ spent: 0, max: 0 })).toBe(1)
  })

  it("leaves out a limit narrowed to one provider key", () => {
    expect(
      limitsOn([ceiling({ provider_key_id: "pk" })], "org_member", "m1"),
    ).toBe(NO_LIMITS)
  })
})

describe("limitsOnPerson", () => {
  it("binds on whichever limit is closest to its cap", () => {
    const limits = limitsOnPerson(
      [
        ceiling({ id: "own", max_budget: 100, current_spend: 10 }),
        ceiling({
          id: "platform",
          scope_type: "workspace_member",
          scope_id: "wm1",
          max_budget: 40,
          current_spend: 30,
        }),
      ],
      member,
    )
    expect(limits.count).toBe(2)
    expect(limits.own?.ceiling.id).toBe("own")
    expect(limits.binding?.ceiling.id).toBe("platform")
    expect(limits.binding?.origin).toBe("in Platform")
  })

  it("has no own limit when only a workspace default reaches them", () => {
    const limits = limitsOnPerson(
      [
        ceiling({
          scope_type: "workspace_member",
          scope_id: "wm2",
          max_budget: 25,
        }),
      ],
      member,
    )
    expect(limits.own).toBeNull()
    expect(limits.binding?.origin).toBe("in Data")
  })

  it("ignores limits on other people", () => {
    expect(
      limitsOnPerson(
        [
          ceiling({ scope_id: "m2" }),
          ceiling({ scope_type: "workspace_member", scope_id: "wm9" }),
        ],
        member,
      ),
    ).toBe(NO_LIMITS)
  })
})

describe("limitLine", () => {
  it("says the whole limit in one line", () => {
    const { own } = limitsOn(
      [
        ceiling({
          max_budget: 30,
          current_spend: 25.62,
          budget_duration_sec: 7 * 86_400,
        }),
      ],
      "org_member",
      "m1",
    )
    expect(own && limitLine(own)).toBe("$25.62 of $30.00 a week")
  })
})

describe("wholePct", () => {
  it("rounds to a whole number", () => {
    expect(wholePct(0.214)).toBe("21%")
    expect(wholePct(1)).toBe("100%")
    expect(wholePct(0)).toBe("0%")
  })

  it("never rounds a real share away to zero", () => {
    expect(wholePct(0.0004)).toBe("<1%")
  })
})
