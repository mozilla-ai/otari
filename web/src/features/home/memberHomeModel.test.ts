import { describe, expect, it } from "vitest"
import {
  hasActiveKey,
  isMemberLimitRefusal,
  isSpendRefusal,
  LOWER_LIMIT_NOTE,
  lowerLimitNote,
  type MemberAttentionInput,
  type MemberAttentionKey,
  memberAttention,
  memberLimit,
  memberLimitLabel,
  ownReadState,
  rosterRowOf,
  UNSEEN_LIMIT_NOTE,
  unseenLimitLines,
} from "@/features/home/memberHomeModel"

const PLATFORM = "ws-platform"
const DATA = "ws-data"
const DAY = 86_400_000
// Noon, so a day either side stays on its calendar day in any time zone.
const NOW = new Date(2026, 10, 11, 12).getTime()

const monthly = {
  budget_id: "b-default",
  max_budget: 1,
  provider_key_id: null,
  reset_alignment: "calendar_month",
  budget_duration_sec: null,
} as const

const placed = (workspaceId: string, budgetId: string, max: number | null) => ({
  workspaces: [
    {
      workspace_id: workspaceId,
      workspace_member_id: `wm-${workspaceId}`,
      workspace_name: workspaceId,
      role: "member",
      ceiling: { id: `c-${budgetId}`, budget_id: budgetId, max_budget: max },
    },
  ],
})

// A Sunday, so the week it is in began the Monday before.
const SUNDAY = Date.parse("2026-11-15T18:00:00Z")
const MONTH_START = "2026-11-01T00:00:00.000Z"

describe("memberLimit", () => {
  it("starts the default's period where the gateway does, on UTC calendar lines", () => {
    const at = (alignment: string) =>
      memberLimit({
        member: undefined,
        workspaceId: PLATFORM,
        defaults: [{ ...monthly, reset_alignment: alignment }],
        now: SUNDAY,
      })
    expect(at("calendar_month")).toMatchObject({ periodStart: MONTH_START })
    expect(at("calendar_week")).toMatchObject({
      periodStart: "2026-11-09T00:00:00.000Z",
    })
    expect(at("calendar_day")).toMatchObject({
      periodStart: "2026-11-15T00:00:00.000Z",
    })
  })

  it("knows no period start for a default that rolls from a request", () => {
    expect(
      memberLimit({
        member: undefined,
        workspaceId: PLATFORM,
        defaults: [
          { ...monthly, reset_alignment: null, budget_duration_sec: 86_400 },
        ],
        now: SUNDAY,
      }),
    ).toMatchObject({ kind: "default", periodStart: "" })
  })

  it("reads the roster's amount as set by an owner or admin", () => {
    expect(
      memberLimit({
        member: placed(PLATFORM, "b-own", 40),
        workspaceId: PLATFORM,
        defaults: [monthly],
        now: SUNDAY,
      }),
    ).toEqual({ kind: "own", amount: 40 })
  })

  it("reads a roster ceiling drawn from the team default as everyone's", () => {
    expect(
      memberLimit({
        member: placed(PLATFORM, "b-default", 1),
        workspaceId: PLATFORM,
        defaults: [monthly],
        now: SUNDAY,
      }),
    ).toEqual({
      kind: "default",
      amount: 1,
      per: "a month",
      periodStart: MONTH_START,
    })
  })

  it("falls back to the team default when the roster has no ceiling there", () => {
    expect(
      memberLimit({
        member: placed(DATA, "b-own", 25),
        workspaceId: PLATFORM,
        defaults: [monthly],
        now: SUNDAY,
      }),
    ).toEqual({
      kind: "default",
      amount: 1,
      per: "a month",
      periodStart: MONTH_START,
    })
  })

  it("ignores a default on one provider key", () => {
    expect(
      memberLimit({
        member: undefined,
        workspaceId: PLATFORM,
        defaults: [{ ...monthly, provider_key_id: "pk-1" }],
        now: SUNDAY,
      }),
    ).toEqual({ kind: "unseen" })
  })

  it("reads a $0 limit as a limit, on the roster or as the default", () => {
    expect(
      memberLimit({
        member: placed(PLATFORM, "b-own", 0),
        workspaceId: PLATFORM,
        defaults: [],
        now: SUNDAY,
      }),
    ).toEqual({ kind: "own", amount: 0 })
    expect(
      memberLimit({
        member: undefined,
        workspaceId: PLATFORM,
        defaults: [{ ...monthly, max_budget: 0 }],
        now: SUNDAY,
      }),
    ).toEqual({
      kind: "default",
      amount: 0,
      per: "a month",
      periodStart: MONTH_START,
    })
  })

  // A limit set on the person across the organization is on no roster row and
  // in no read a member may make, so what they can read never rules one out.
  it("says unseen, not none, when nothing the caller can read caps them", () => {
    expect(
      memberLimit({
        member: placed(PLATFORM, "b-own", null),
        workspaceId: PLATFORM,
        defaults: [],
        now: SUNDAY,
      }),
    ).toEqual({ kind: "unseen" })
  })
})

describe("rosterRowOf", () => {
  const rows = [
    { user_id: "u-1", attribution_user_id: "a-1", name: "one" },
    { user_id: null, attribution_user_id: "a-2", name: "two" },
  ]
  it("finds the caller by either id", () => {
    expect(rosterRowOf(rows, "u-1")?.name).toBe("one")
    expect(rosterRowOf(rows, "a-2")?.name).toBe("two")
  })
  it("finds nobody without an id", () => {
    expect(rosterRowOf(rows, undefined)).toBeUndefined()
  })
})

const key = (overrides: Partial<MemberAttentionKey>): MemberAttentionKey => ({
  id: "k-1",
  key_name: "nightly",
  key_suffix: "abcd",
  expires_at: null,
  is_active: true,
  disabled_reason: null,
  ...overrides,
})

const input = (
  overrides: Partial<MemberAttentionInput> = {},
): MemberAttentionInput => ({
  now: NOW,
  userId: "u-max",
  states: { refusals: "ready", failures: "ready", keys: "ready" },
  refused: 0,
  day: { requests: 7, errors: [{ errorClass: "client_error", requests: 1 }] },
  keys: [key({})],
  ...overrides,
})

describe("memberAttention", () => {
  it("is quiet when nothing qualifies, and a caller's own 400 is not a failure", () => {
    const result = memberAttention(
      input({
        day: {
          requests: 40,
          errors: [{ errorClass: "client_error", requests: 6 }],
        },
      }),
    )
    expect(result.items).toEqual([])
    expect(result.quiet).toBe(
      "Nothing needs you right now. None of your requests was refused in the last 24 hours, and none of your keys expires this week.",
    )
  })

  it("waits for every source before it says anything", () => {
    const result = memberAttention(
      input({
        states: { refusals: "loading", failures: "ready", keys: "ready" },
      }),
    )
    expect(result.isLoading).toBe(true)
    expect(result.quiet).toBeUndefined()
  })

  it("names a source it could not read rather than calling the list quiet", () => {
    const result = memberAttention(
      input({
        states: { refusals: "ready", failures: "failed", keys: "ready" },
      }),
    )
    expect(result.quiet).toBeUndefined()
    expect(result.notChecked).toEqual([
      "Failures could not be checked, so this list may be incomplete.",
    ])
  })

  it("counts the caller's refusals, opening Requests on their errors", () => {
    const one = memberAttention(input({ refused: 1 })).items[0]
    expect(one?.text).toBe(
      "1 of your requests was refused in the last 24 hours.",
    )
    expect(one?.action).toEqual({
      label: "See the refusals",
      to: "/requests",
      search: { status_code: "403", range: "24h", user_id: "u-max" },
    })
    expect(memberAttention(input({ refused: 3 })).items[0]?.text).toBe(
      "3 of your requests were refused in the last 24 hours.",
    )
  })

  it("counts failures past the owner's thresholds only", () => {
    const below = memberAttention(
      input({
        day: {
          requests: 100,
          errors: [{ errorClass: "provider_error", requests: 4 }],
        },
      }),
    )
    expect(below.items).toEqual([])
    const past = memberAttention(
      input({
        day: {
          requests: 40,
          errors: [{ errorClass: "provider_error", requests: 6 }],
        },
      }),
    ).items[0]
    expect(past?.text).toBe(
      "15.0% of your requests failed in the last 24 hours.",
    )
    expect(past?.isUrgent).toBe(true)
  })

  it("says when an own key expires this week, in calendar days", () => {
    const text = (at: number) =>
      memberAttention(
        input({ keys: [key({ expires_at: new Date(at).toISOString() })] }),
      ).items[0]?.text
    expect(text(NOW + 2 * 3_600_000)).toBe("Your key nightly expires today.")
    expect(text(NOW + DAY)).toBe("Your key nightly expires tomorrow.")
    expect(text(NOW + 3 * DAY)).toBe("Your key nightly expires in 3 days.")
    expect(text(NOW + 8 * DAY)).toBeUndefined()
  })

  it("says when an own key expired this past week, and opens expired keys", () => {
    const item = memberAttention(
      input({
        keys: [key({ expires_at: new Date(NOW - 2 * DAY).toISOString() })],
      }),
    ).items[0]
    expect(item?.text).toBe("Your key nightly expired on Nov 9.")
    expect(item?.action.search).toEqual({ state: "expired" })
    expect(
      memberAttention(
        input({
          keys: [key({ expires_at: new Date(NOW - 9 * DAY).toISOString() })],
        }),
      ).items,
    ).toEqual([])
  })

  it("says an operator switched a key off, once, whatever its expiry", () => {
    const items = memberAttention(
      input({
        keys: [
          key({
            key_name: null,
            disabled_reason: "operator",
            is_active: false,
            expires_at: new Date(NOW - DAY).toISOString(),
          }),
        ],
      }),
    ).items
    expect(items.map((one) => one.text)).toEqual([
      "Your key ending abcd was disabled by an operator.",
    ])
  })

  it("says a revoked key was revoked, not switched off", () => {
    const items = memberAttention(
      input({
        keys: [
          key({ key_name: null, disabled_reason: "revoked", is_active: false }),
        ],
      }),
    ).items
    expect(items.map((one) => one.text)).toEqual([
      "Your key ending abcd was revoked.",
    ])
  })

  it("puts what stops requests now before what is coming", () => {
    const items = memberAttention(
      input({
        keys: [
          key({ id: "soon", expires_at: new Date(NOW + DAY).toISOString() }),
        ],
        refused: 2,
      }),
    ).items
    expect(items.map((one) => one.id)).toEqual(["refused", "key-expiring:soon"])
  })
})

describe("ownReadState", () => {
  // A read of the caller's own requests is off until their id is known, so
  // React Query reports it as pending with no data, however long that lasts.
  const off = { data: undefined, isError: false }

  it("fails with the roster rather than waiting on a read that is never asked", () => {
    expect(
      ownReadState(
        { userId: undefined, isPending: false, isFailed: true },
        off,
      ),
    ).toBe("failed")
    const band = memberAttention({
      now: NOW,
      userId: undefined,
      states: {
        refusals: ownReadState(
          { userId: undefined, isPending: false, isFailed: true },
          off,
        ),
        failures: ownReadState(
          { userId: undefined, isPending: false, isFailed: true },
          off,
        ),
        keys: "ready",
      },
      refused: 0,
      day: { requests: 0, errors: [] },
      keys: [],
    })
    expect(band.isLoading).toBe(false)
    expect(band.notChecked).toEqual([
      "Refusals could not be checked, so this list may be incomplete.",
      "Failures could not be checked, so this list may be incomplete.",
    ])
  })

  it("answers for a caller with no billing id: nothing is filed under them", () => {
    expect(
      ownReadState(
        { userId: undefined, isPending: false, isFailed: false },
        off,
      ),
    ).toBe("ready")
  })

  it("waits while the id resolves, then reads the query", () => {
    expect(
      ownReadState(
        { userId: undefined, isPending: true, isFailed: false },
        off,
      ),
    ).toBe("loading")
    const me = { userId: "u1", isPending: false, isFailed: false }
    expect(ownReadState(me, off)).toBe("loading")
    expect(ownReadState(me, { data: undefined, isError: true })).toBe("failed")
    expect(ownReadState(me, { data: {}, isError: false })).toBe("ready")
  })
})

describe("hasActiveKey", () => {
  const key = (over: { is_active?: boolean; expires_at?: string | null }) => ({
    is_active: true,
    expires_at: null,
    ...over,
  })

  it("counts a key with no expiry, or one still ahead", () => {
    expect(hasActiveKey([key({})], NOW)).toBe(true)
    expect(
      hasActiveKey(
        [key({ expires_at: new Date(NOW + 1000).toISOString() })],
        NOW,
      ),
    ).toBe(true)
  })

  it("does not count a disabled key or one past its expiry", () => {
    expect(hasActiveKey([key({ is_active: false })], NOW)).toBe(false)
    expect(
      hasActiveKey(
        [key({ expires_at: new Date(NOW - 1000).toISOString() })],
        NOW,
      ),
    ).toBe(false)
  })
})

describe("the figure on a member's Home", () => {
  const WEEK_START = "2026-11-09T00:00:00.000Z"
  const WEEKLY = {
    kind: "default",
    amount: 40,
    per: "a week",
    periodStart: WEEK_START,
  } as const

  it("is named for what it is: the workspace's amount, or the person's own", () => {
    expect(memberLimitLabel(WEEKLY, "Platform")).toBe("Limit per person")
    expect(memberLimitLabel({ kind: "unseen" }, "Platform")).toBe("Your limit")
    expect(memberLimitLabel({ kind: "own", amount: 5 }, "Platform")).toBe(
      "Your limit in Platform",
    )
  })

  it("says a lower limit may apply when a spend refusal came with spend under the figure", () => {
    expect(
      lowerLimitNote({
        limit: WEEKLY,
        since: WEEK_START,
        spent: 0.48,
        spendRefusals: 3,
      }),
    ).toBe(LOWER_LIMIT_NOTE)
  })

  it("says nothing to a member who was not refused, or who spent the figure", () => {
    expect(
      lowerLimitNote({
        limit: WEEKLY,
        since: WEEK_START,
        spent: 0.48,
        spendRefusals: 0,
      }),
    ).toBe(undefined)
    expect(
      lowerLimitNote({
        limit: WEEKLY,
        since: WEEK_START,
        spent: 40,
        spendRefusals: 3,
      }),
    ).toBe(undefined)
    expect(
      lowerLimitNote({
        limit: { kind: "unseen" },
        since: WEEK_START,
        spent: 1,
        spendRefusals: 3,
      }),
    ).toBe(undefined)
  })

  it("compares only figures read over the limit's own period", () => {
    // A week of spend says nothing of a monthly limit that began weeks ago.
    expect(
      lowerLimitNote({
        limit: {
          kind: "default",
          amount: 40,
          per: "a month",
          periodStart: "2026-11-01T00:00:00.000Z",
        },
        since: WEEK_START,
        spent: 30,
        spendRefusals: 3,
      }),
    ).toBe(undefined)
    // Nor does a month of it say anything of this week.
    expect(
      lowerLimitNote({
        limit: WEEKLY,
        since: "2026-11-01T00:00:00.000Z",
        spent: 30,
        spendRefusals: 3,
      }),
    ).toBe(undefined)
  })

  it("says nothing for a default whose period rolls from a request", () => {
    expect(
      lowerLimitNote({
        limit: { ...WEEKLY, periodStart: "" },
        since: WEEK_START,
        spent: 0.48,
        spendRefusals: 3,
      }),
    ).toBe(undefined)
  })

  it("counts a refusal over budget, and not a blocked caller or a model outside a key's list", () => {
    expect(
      isSpendRefusal(
        "Member has $0.015145 of budget remaining, less than this",
      ),
    ).toBe(true)
    expect(isSpendRefusal("Model not allowed for this key")).toBe(false)
    expect(isSpendRefusal(null)).toBe(false)
  })
})

describe("a limit a member cannot read", () => {
  const UNSEEN = { kind: "unseen" } as const

  it("is a limit on the member only where the refusal names a member", () => {
    expect(isMemberLimitRefusal("Member has exceeded budget limit")).toBe(true)
    expect(
      isMemberLimitRefusal(
        "Member has $0.01 of budget remaining, less than this request's estimated cost",
      ),
    ).toBe(true)
    for (const other of [
      "Organization has exceeded budget limit",
      "Workspace has exceeded budget limit",
      "API key has exceeded budget limit",
      "User 'u-1' has exceeded budget limit",
      "Model not allowed for this key",
      null,
    ]) {
      expect(isMemberLimitRefusal(other)).toBe(false)
    }
  })

  it("says who can see it, and that it refused the member when it did", () => {
    const lines = (refusals: string[], words = "this month") =>
      unseenLimitLines({
        limit: UNSEEN,
        refusals,
        words,
        workspace: "Platform",
      })
    const unrefused = {
      subline: "None for everyone in Platform",
      note: UNSEEN_LIMIT_NOTE,
    }
    expect(lines([])).toEqual(unrefused)
    expect(lines(["Workspace has exceeded budget limit"])).toEqual(unrefused)
    expect(
      lines(["Member has exceeded budget limit"], "in the last 7 days"),
    ).toEqual({
      subline: "Only owners and admins see it",
      note: "A limit set on you refused requests in the last 7 days.",
    })
  })

  it("waits for the refusals, and says nothing over a limit the member can read", () => {
    expect(
      unseenLimitLines({
        limit: UNSEEN,
        refusals: undefined,
        words: "this month",
        workspace: "Platform",
      }),
    ).toBe(undefined)
    expect(
      unseenLimitLines({
        limit: { kind: "own", amount: 40 },
        refusals: ["Member has exceeded budget limit"],
        words: "this month",
        workspace: "Platform",
      }),
    ).toBe(undefined)
  })
})
