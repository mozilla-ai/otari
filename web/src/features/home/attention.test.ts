import { describe, expect, it } from "vitest"
import type { OrganizationSpendCeiling } from "@/client"
import {
  type AttentionInput,
  type AttentionSource,
  attention,
} from "@/features/home/attention"
import type { LimitTarget } from "@/features/spend/limits"

const DAY = 86_400_000
const NOW = Date.parse("2026-11-11T12:00:00Z")
const iso = (ms: number) => new Date(ms).toISOString()

let counter = 0
function ceiling(
  overrides: Partial<OrganizationSpendCeiling>,
): OrganizationSpendCeiling {
  counter += 1
  return {
    id: `c${counter}`,
    budget_id: "b",
    scope_type: "org_member",
    scope_id: "m1",
    max_budget: 100,
    current_spend: 0,
    provider_key_id: null,
    reset_alignment: "calendar_month",
    budget_duration_sec: null,
    period_start: iso(NOW - 10 * DAY),
    period_end: iso(NOW + 20 * DAY),
    ...overrides,
  } as OrganizationSpendCeiling
}

const target = (scope: LimitTarget["scope"], scopeId: string): LimitTarget => ({
  scope,
  scopeId,
  name: scopeId,
  own: null,
})

function input(overrides: Partial<AttentionInput> = {}): AttentionInput {
  return {
    now: NOW,
    states: {
      limits: "ready",
      refusals: "ready",
      failures: "ready",
      account: "ready",
    },
    ceilings: [],
    subjects: new Map([
      ["m1", "Wei Zhang"],
      ["m2", "Ana Ruiz"],
      ["w1", "The Platform workspace"],
      ["k1", "The key ci"],
    ]),
    workspaceNames: new Map([["w1", "Platform"]]),
    targets: new Map(),
    peopleOf: new Map(),
    account: { sourceName: "The balance" },
    day: { requests: 100, errors: [] },
    refusals: [],
    ...overrides,
  }
}

const texts = (overrides: Partial<AttentionInput>) =>
  attention(input(overrides)).items.map((one) => one.text)

// What an edition that holds the organization's money contributes
// (`overlayHomeAccount.tsx`). Nothing in this build: these pin how a row it
// gives is ranked and carried, not what it says.
const ACCOUNT_ROW = {
  id: "balance",
  isUrgent: false,
  text: "The balance lasts about 6 days at this week’s pace.",
  action: "Add funds",
}

describe("type 1, an edition's account row", () => {
  it("is carried as given, with its own control", () => {
    const [item] = attention(
      input({
        account: { sourceName: "The balance", attention: ACCOUNT_ROW },
      }),
    ).items
    expect(item).toEqual({
      id: "balance",
      isUrgent: false,
      text: "The balance lasts about 6 days at this week’s pace.",
      action: { label: "", node: "Add funds" },
    })
  })

  it("is not there when nothing is contributed", () => {
    expect(texts({})).toEqual([])
  })
})

describe("type 2, a limit reached", () => {
  const reached = (overrides: Partial<OrganizationSpendCeiling> = {}) =>
    attention(
      input({
        ceilings: [ceiling({ current_spend: 100, ...overrides })],
        targets: new Map(),
      }),
    ).items[0]

  it("says when it resets, in days", () => {
    expect(reached()?.text).toBe(
      "Wei Zhang is at the limit of $100.00 a month. Requests are refused until it resets in 20 days.",
    )
    expect(reached()?.isUrgent).toBe(true)
  })

  it("says tomorrow for one day", () => {
    expect(reached({ period_end: iso(NOW + 0.5 * DAY) })?.text).toMatch(
      /until it resets tomorrow\.$/,
    )
  })

  it("never calls a limit it cannot name the organization's", () => {
    const who = (scope_type: string, scope_id: string) =>
      reached({ scope_type, scope_id })?.text.split(" is at the limit")[0]
    expect(who("api_token", "k-teammate")).toBe("An API key")
    expect(who("org_member", "m-gone")).toBe("Someone")
    expect(who("workspace_member", "wm-gone")).toBe("Someone")
    expect(who("workspace", "w-unread")).toBe("A workspace")
    expect(who("organization", "org-1")).toBe("The organization")
  })

  it("counts a $0 limit as reached, before anything is spent", () => {
    expect(reached({ max_budget: 0, current_spend: 0 })?.text).toBe(
      "Wei Zhang is at the limit of $0.00 a month. Requests are refused until it resets in 20 days.",
    )
  })

  it("says until the limit is raised when it never resets", () => {
    expect(reached({ period_end: null, period_start: null })?.text).toMatch(
      /until the limit is raised\.$/,
    )
  })

  it("opens the dialog where Home has the row, and Rules where it has none", () => {
    const withRow = attention(
      input({
        ceilings: [ceiling({ id: "cx", current_spend: 100 })],
        targets: new Map([["cx", target("org_member", "m1")]]),
      }),
    ).items[0]
    expect(withRow?.action.label).toBe("Raise the limit")
    expect(withRow?.action.limit?.scopeId).toBe("m1")
    expect(reached()?.action).toEqual({ label: "Open Rules", to: "/rules" })
  })

  it("is not reached once its period has ended", () => {
    expect(
      texts({
        ceilings: [
          ceiling({ current_spend: 100, period_end: iso(NOW - 3 * DAY) }),
        ],
      }),
    ).toEqual([])
  })

  it("ignores a limit on a provider key", () => {
    expect(
      texts({
        ceilings: [ceiling({ current_spend: 100, provider_key_id: "p1" })],
      }),
    ).toEqual([])
  })
})

describe("type 3, refused", () => {
  const refusals = (
    rows: { userId: string | null; name: string; requests: number }[],
    extra: Partial<AttentionInput> = {},
  ) => attention(input({ refusals: rows, ...extra })).items[0]

  it("names one person and links to their refusals", () => {
    const item = refusals([{ userId: "u1", name: "Ana Ruiz", requests: 4 }])
    expect(item?.text).toBe(
      "The gateway refused 4 requests from Ana Ruiz in the last 24 hours.",
    )
    expect(item?.isUrgent).toBe(true)
    expect(item?.action).toEqual({
      label: "See the refusals",
      to: "/requests",
      // The refused requests only, not every failure in the window.
      search: { status_code: "403", range: "24h", user_id: "u1" },
    })
  })

  it("says a request for one", () => {
    expect(
      refusals([{ userId: "u1", name: "Ana Ruiz", requests: 1 }])?.text,
    ).toBe("The gateway refused a request from Ana Ruiz in the last 24 hours.")
  })

  it("counts people when there are several, with no user filter", () => {
    const item = refusals([
      { userId: "u1", name: "Ana Ruiz", requests: 4 },
      { userId: "u2", name: "Wei Zhang", requests: 3 },
    ])
    expect(item?.text).toBe(
      "The gateway refused 7 requests from 2 people in the last 24 hours.",
    )
    expect(item?.action.search).toEqual({ status_code: "403", range: "24h" })
  })

  it("names nobody when the requests had no person", () => {
    expect(
      refusals([{ userId: null, name: "No person", requests: 9 }])?.text,
    ).toBe("The gateway refused 9 requests in the last 24 hours.")
  })

  it("skips a person who already has a limit-reached row", () => {
    const items = attention(
      input({
        ceilings: [ceiling({ current_spend: 100 })],
        peopleOf: new Map([["m1", ["u1"]]]),
        refusals: [{ userId: "u1", name: "Wei Zhang", requests: 6 }],
      }),
    ).items
    expect(items.map((one) => one.id)).not.toContain("refused")
    expect(items).toHaveLength(1)
  })

  it("keeps the others when only one person is skipped", () => {
    const items = attention(
      input({
        ceilings: [ceiling({ current_spend: 100 })],
        peopleOf: new Map([["m1", ["u1"]]]),
        refusals: [
          { userId: "u1", name: "Wei Zhang", requests: 6 },
          { userId: "u2", name: "Ana Ruiz", requests: 2 },
        ],
      }),
    ).items
    expect(items.map((one) => one.text)).toContain(
      "The gateway refused 2 requests from Ana Ruiz in the last 24 hours.",
    )
  })
})

describe("type 4, failures", () => {
  const failing = (
    requests: number,
    errors: { errorClass: string; requests: number }[],
  ) => attention(input({ day: { requests, errors } })).items[0]

  it("counts only server-side classes", () => {
    expect(
      failing(100, [
        { errorClass: "client_error", requests: 40 },
        { errorClass: "auth", requests: 40 },
      ]),
    ).toBeUndefined()
  })

  it("needs five errors", () => {
    expect(
      failing(10, [{ errorClass: "provider_error", requests: 4 }]),
    ).toBeUndefined()
    expect(
      failing(10, [{ errorClass: "provider_error", requests: 5 }])?.isUrgent,
    ).toBe(false)
  })

  it("needs a rate of 2%", () => {
    expect(
      failing(1000, [{ errorClass: "rate_limit", requests: 19 }]),
    ).toBeUndefined()
    expect(
      failing(1000, [{ errorClass: "rate_limit", requests: 20 }])?.text,
    ).toBe("20 of 1,000 requests failed in the last 24 hours (2.0%).")
  })

  it("is urgent at 10% with 20 requests, and only a warning with fewer", () => {
    expect(
      failing(20, [{ errorClass: "pricing", requests: 5 }])?.isUrgent,
    ).toBe(true)
    expect(
      failing(19, [{ errorClass: "pricing", requests: 5 }])?.isUrgent,
    ).toBe(false)
    expect(
      failing(200, [{ errorClass: "unknown", requests: 19 }])?.isUrgent,
    ).toBe(false)
  })

  it("adds the counted classes together and links to the failures", () => {
    const item = failing(50, [
      { errorClass: "provider_error", requests: 3 },
      { errorClass: "rate_limit", requests: 3 },
      { errorClass: "client_error", requests: 30 },
    ])
    expect(item?.text).toBe(
      "6 of 50 requests failed in the last 24 hours (12.0%).",
    )
    expect(item?.action).toEqual({
      label: "See the failures",
      to: "/requests",
      search: { status: "error", range: "24h" },
    })
  })
})

describe("type 5, pace", () => {
  const month = {
    period_start: iso(NOW - 10 * DAY),
    period_end: iso(NOW + 20 * DAY),
  }
  const date = (ms: number) =>
    new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" }).format(
      ms,
    )

  it("projects an organization limit and offers to change it", () => {
    // 50 spent in 10 of 30 days: 150 projected on 100, 40 left at 5 a day.
    const [item] = attention(
      input({
        ceilings: [
          ceiling({
            id: "co",
            scope_type: "organization",
            scope_id: "org",
            current_spend: 50,
            ...month,
          }),
        ],
        targets: new Map([["co", target("organization", "org")]]),
      }),
    ).items
    expect(item?.text).toBe(
      `At this pace the team budget of $100.00 a month runs out around ${date(NOW + 10 * DAY)}, 10 days before it resets.`,
    )
    expect(item?.isUrgent).toBe(false)
    expect(item?.action.label).toBe("Change the limit")
    expect(item?.action.limit?.scope).toBe("organization")
  })

  it("names the workspace and offers to raise its limit", () => {
    const [item] = attention(
      input({
        ceilings: [
          ceiling({
            id: "cw",
            scope_type: "workspace",
            scope_id: "w1",
            current_spend: 50,
            ...month,
          }),
        ],
        targets: new Map([["cw", target("workspace", "w1")]]),
      }),
    ).items
    expect(item?.text).toMatch(
      /^At this pace the Platform workspace’s limit of \$100\.00 a month runs out around /,
    )
    expect(item?.action.label).toBe("Raise the limit")
  })

  it("is only for organizations and workspaces", () => {
    expect(
      texts({ ceilings: [ceiling({ current_spend: 50, ...month })] }),
    ).toEqual([])
  })

  it("needs a period of a week", () => {
    expect(
      texts({
        ceilings: [
          ceiling({
            scope_type: "organization",
            current_spend: 60,
            period_start: iso(NOW - 3 * DAY),
            period_end: iso(NOW + 3 * DAY),
          }),
        ],
      }),
    ).toEqual([])
  })

  it("needs a tenth of the period gone", () => {
    expect(
      texts({
        ceilings: [
          ceiling({
            scope_type: "organization",
            current_spend: 60,
            period_start: iso(NOW - 2 * DAY),
            period_end: iso(NOW + 28 * DAY),
          }),
        ],
      }),
    ).toEqual([])
  })

  it("is not raised when the pace stays under the maximum", () => {
    expect(
      texts({
        ceilings: [
          ceiling({ scope_type: "organization", current_spend: 30, ...month }),
        ],
      }),
    ).toEqual([])
  })

  it("gives way to near-limit at 80% used", () => {
    const items = attention(
      input({
        ceilings: [
          ceiling({ scope_type: "organization", current_spend: 85, ...month }),
        ],
      }),
    ).items
    expect(items).toHaveLength(1)
    expect(items[0]?.text).toMatch(/has used 85% of/)
  })
})

describe("type 6, near a limit", () => {
  it("says how far along it is and offers to raise it", () => {
    const [item] = attention(
      input({
        ceilings: [ceiling({ id: "cn", current_spend: 85 })],
        targets: new Map([["cn", target("org_member", "m1")]]),
      }),
    ).items
    expect(item?.text).toBe(
      "Wei Zhang has used 85% of $100.00 a month. Resets in 20 days.",
    )
    expect(item?.isUrgent).toBe(false)
    expect(item?.action.label).toBe("Raise the limit")
  })

  it("is not near below 80%", () => {
    expect(texts({ ceilings: [ceiling({ current_spend: 79 })] })).toEqual([])
  })

  it("ignores a limit whose period ended, however far along it was", () => {
    expect(
      texts({
        ceilings: [ceiling({ current_spend: 85, period_end: iso(NOW - DAY) })],
      }),
    ).toEqual([])
  })
})

describe("order and fold", () => {
  it("puts urgent before warnings, then keeps the table's order", () => {
    const result = attention(
      input({
        account: { sourceName: "The balance", attention: ACCOUNT_ROW },
        ceilings: [
          ceiling({ id: "near", current_spend: 90 }),
          ceiling({ id: "full", scope_id: "m2", current_spend: 100 }),
        ],
        refusals: [{ userId: "u9", name: "Eli", requests: 2 }],
        day: {
          requests: 100,
          errors: [{ errorClass: "unknown", requests: 6 }],
        },
      }),
    )
    expect(result.items.map((one) => one.id)).toEqual([
      "full",
      "refused",
      "balance",
      "failures",
      "near",
    ])
  })

  it("sorts limits organization, workspace, person, key, then by how far along", () => {
    const result = attention(
      input({
        ceilings: [
          ceiling({
            id: "key",
            scope_type: "api_token",
            scope_id: "k1",
            current_spend: 99,
          }),
          ceiling({ id: "person-low", current_spend: 81 }),
          ceiling({ id: "person-high", scope_id: "m2", current_spend: 95 }),
          ceiling({
            id: "workspace",
            scope_type: "workspace",
            scope_id: "w1",
            current_spend: 82,
          }),
          ceiling({
            id: "org",
            scope_type: "organization",
            scope_id: "o",
            current_spend: 83,
          }),
        ],
      }),
    )
    expect([...result.items, ...result.folded].map((one) => one.id)).toEqual([
      "org",
      "workspace",
      "person-high",
      "person-low",
      "key",
    ])
  })

  it("shows three near-limit rows and folds the rest", () => {
    const result = attention(
      input({
        ceilings: [81, 82, 83, 84, 85].map((spent, index) =>
          ceiling({
            id: `n${index}`,
            scope_id: `m${index + 10}`,
            current_spend: spent,
          }),
        ),
      }),
    )
    expect(result.items).toHaveLength(3)
    expect(result.folded).toHaveLength(2)
    expect(result.foldText).toBe("2 more limits are past 80%.")
    expect(result.folded.map((one) => one.id)).toEqual(["n1", "n0"])
  })

  it("does not fold at three, and never folds urgent rows", () => {
    const three = attention(
      input({
        ceilings: [81, 82, 83].map((spent, index) =>
          ceiling({ id: `n${index}`, current_spend: spent }),
        ),
      }),
    )
    expect(three.folded).toEqual([])
    const urgent = attention(
      input({
        ceilings: [1, 2, 3, 4, 5].map((index) =>
          ceiling({ id: `f${index}`, current_spend: 100 }),
        ),
      }),
    )
    expect(urgent.items).toHaveLength(5)
    expect(urgent.folded).toEqual([])
  })

  it("says one more limit in the singular", () => {
    const result = attention(
      input({
        ceilings: [81, 82, 83, 84].map((spent, index) =>
          ceiling({ id: `n${index}`, current_spend: spent }),
        ),
      }),
    )
    expect(result.foldText).toBe("1 more limit is past 80%.")
  })
})

describe("the ended-period rule", () => {
  it("counts an ended period as nothing spent: no row at all", () => {
    const ended = ceiling({
      current_spend: 25.62,
      max_budget: 30,
      period_end: iso(NOW - 3 * DAY),
    })
    const result = attention(input({ ceilings: [ended] }))
    expect(result.items).toEqual([])
    expect(result.quiet?.lead).toBe("Nothing needs you right now.")
  })

  it("still counts a period that ends later today", () => {
    expect(
      texts({
        ceilings: [
          ceiling({ current_spend: 100, period_end: iso(NOW + 3600_000) }),
        ],
      }),
    ).toHaveLength(1)
  })
})

describe("states", () => {
  it("is loading until every source has answered", () => {
    const sources: AttentionSource[] = [
      "limits",
      "refusals",
      "failures",
      "account",
    ]
    for (const source of sources) {
      const states = {
        limits: "ready",
        refusals: "ready",
        failures: "ready",
        account: "ready",
        [source]: "loading",
      } as AttentionInput["states"]
      const result = attention(input({ states }))
      expect(result.isLoading).toBe(true)
      expect(result.quiet).toBeUndefined()
    }
    expect(attention(input()).isLoading).toBe(false)
  })

  it("names a source that failed, last, and does not claim all is well", () => {
    const result = attention(
      input({
        states: {
          limits: "failed",
          refusals: "ready",
          failures: "ready",
          account: "ready",
        },
      }),
    )
    expect(result.notChecked).toEqual([
      "Limits could not be checked, so this list may be incomplete.",
    ])
    expect(result.quiet).toBeUndefined()
  })

  it("still lists what it could read when another source failed", () => {
    const result = attention(
      input({
        states: {
          limits: "ready",
          refusals: "failed",
          failures: "ready",
          account: "failed",
        },
        ceilings: [ceiling({ current_spend: 100 })],
      }),
    )
    expect(result.items).toHaveLength(1)
    expect(result.notChecked).toEqual([
      "Refusals could not be checked, so this list may be incomplete.",
      "The balance could not be checked, so this list may be incomplete.",
    ])
  })
})

describe("nothing qualifies", () => {
  it("says so", () => {
    expect(attention(input()).quiet).toEqual({
      lead: "Nothing needs you right now.",
      detail:
        "No limit is past 80% and nothing was refused in the last 24 hours.",
    })
  })

  it("adds the clause an edition contributes", () => {
    expect(
      attention(
        input({
          account: {
            sourceName: "The balance",
            quietClause: "the balance tops up on its own",
          },
        }),
      ).quiet?.detail,
    ).toBe(
      "No limit is past 80%, nothing was refused in the last 24 hours, and the balance tops up on its own.",
    )
  })
})
