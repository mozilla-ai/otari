import { describe, expect, it } from "vitest"
import type {
  ApiKey,
  OrganizationMember,
  TraceSummary,
  UsageGroupRow,
} from "@/client"
import {
  activeKeysOf,
  type MonthRecord,
  personRows,
  rosterLimits,
} from "@/features/people/people"
import { NO_LIMITS } from "@/features/spend/limits"
import type { SpendRow } from "@/features/spend/useSpend"
import { apiKey, organizationMember } from "@/tests/fixtures"
import { traceSummary } from "@/tests/traceFixtures"

const PLATFORM = "ws-platform"
const MOBILE = "ws-mobile"

function member(
  id: string,
  overrides: Partial<OrganizationMember> = {},
): OrganizationMember {
  return organizationMember({
    organization_member_id: `om-${id}`,
    user_id: `login-${id}`,
    attribution_user_id: `user-${id}`,
    email: `${id}@example.com`,
    full_name: id.charAt(0).toUpperCase() + id.slice(1),
    role: "member",
    status: "active",
    created_at: "2026-11-01T00:00:00Z",
    workspaces: [
      {
        workspace_id: PLATFORM,
        workspace_name: "Platform",
        workspace_member_id: `wm-${id}`,
        role: "member",
      },
    ],
    ...overrides,
  })
}

/** A row of Home's People cut, keyed as `useSpend` keys a member's row. */
function spendRow(id: string, cost: number, tokens = 0): SpendRow {
  return {
    id: `person:om-${id}`,
    filterKey: `user-${id}`,
    name: id,
    cost,
    requests: 1,
    tokens,
    limits: NO_LIMITS,
    target: null,
  }
}

function usage(key: string, requests: number): UsageGroupRow {
  return { key, label: key, cost: 0, requests, tokens: 0, is_other: false }
}

function session(user: string, workspace = PLATFORM): TraceSummary {
  return traceSummary({ user_id: user, workspace_id: workspace })
}

function key(owner: string | null, isActive = true): ApiKey {
  return apiKey({
    id: `${owner}-${isActive}`,
    user_id: owner,
    is_active: isActive,
  })
}

const NOW = Date.parse("2026-11-17T12:00:00Z")

const EMPTY: MonthRecord = {
  now: NOW,
  spendRows: [],
  refusalsByUser: [],
  sessions: [],
  keys: [],
}

describe("personRows", () => {
  it("takes spend and tokens from Home's row for the same membership", () => {
    const [row] = personRows([member("wei")], {
      ...EMPTY,
      // Home's row for another membership is not theirs, even one recorded
      // under the same spender id (a shared service identity, say).
      spendRows: [
        { ...spendRow("liam", 99, 1), filterKey: "user-wei" },
        spendRow("wei", 13.16, 52_000),
      ],
    })
    expect(row).toMatchObject({ spend: 13.16, tokens: 52_000 })
    expect(row?.spendRow?.id).toBe("person:om-wei")
  })

  it("counts refusals and sessions by the id requests are recorded under", () => {
    const [row] = personRows([member("wei")], {
      ...EMPTY,
      refusalsByUser: [usage("user-wei", 3), usage("login-wei", 9)],
      sessions: [
        session("user-wei"),
        session("user-wei"),
        session("login-wei"),
      ],
    })
    expect(row).toMatchObject({ refusals: 3, sessions: 2 })
  })

  it("counts a person's active keys under either id, and not revoked ones", () => {
    const [row] = personRows([member("hana")], {
      ...EMPTY,
      keys: [
        key("user-hana"),
        key("login-hana"),
        key("user-hana", false),
        key("user-wei"),
        key(null),
      ],
    })
    expect(row?.apiKeys).toBe(2)
  })

  it("does not count a key past its expiry as active", () => {
    const expired = apiKey({
      id: "miao",
      user_id: "user-hana",
      is_active: true,
      expires_at: "2026-11-16T23:00:00Z",
    })
    const [row] = personRows([member("hana")], {
      ...EMPTY,
      keys: [key("user-hana"), expired],
    })
    expect(row?.apiKeys).toBe(1)
    expect(
      activeKeysOf([key("user-hana"), expired], member("hana"), NOW),
    ).toHaveLength(1)
  })

  it("marks only the columns whose read failed, and keeps the rest", () => {
    const [row] = personRows([member("max")], {
      ...EMPTY,
      spendRows: [spendRow("max", 4.5, 9_000)],
      refusalsByUser: [{ key: "user-max", requests: 3 } as UsageGroupRow],
      unread: { spend: true },
    })
    expect(row?.unread).toEqual({
      spend: true,
      sessions: false,
      refusals: false,
      keys: false,
    })
    expect(row?.refusals).toBe(3)
  })

  it("leaves the spend of someone folded into Home's Other row unknown, not zero", () => {
    const [ana, wei] = personRows([member("ana"), member("wei")], {
      ...EMPTY,
      spendRows: [
        spendRow("ana", 9),
        { ...spendRow("wei", 0), isFolded: true },
      ],
    })
    expect(ana?.name).toBe("Ana")
    expect(ana?.unread.spend).toBe(false)
    expect(wei?.name).toBe("Wei")
    expect(wei?.unread.spend).toBe(true)
    // Only the spend columns: their other reads stand.
    expect(wei?.unread.sessions).toBe(false)
  })

  it("leaves another person's key count unknown to someone who reads only their own", () => {
    const rows = personRows([member("max"), member("hana")], {
      ...EMPTY,
      keys: [key("user-max")],
      keyReaderMemberIds: new Set(["om-max"]),
    })
    expect(rows.map((row) => [row.name, row.apiKeys])).toEqual([
      ["Hana", undefined],
      ["Max", 1],
    ])
  })

  it("lists only the people placed in the workspace asked for", () => {
    const placed = (id: string, workspaceId: string) =>
      member(id, {
        workspaces: [
          {
            workspace_id: workspaceId,
            workspace_name: workspaceId,
            workspace_member_id: `wm-${id}`,
            role: "member",
          },
        ],
      })
    const everyone = [
      placed("wei", PLATFORM),
      placed("hana", MOBILE),
      member("liam", { workspaces: [] }),
      member("zoe", {
        workspaces: [
          ...(placed("zoe", PLATFORM).workspaces ?? []),
          ...(placed("zoe", MOBILE).workspaces ?? []),
        ],
      }),
    ]
    const names = (workspaceId?: string) =>
      personRows(everyone, { ...EMPTY, workspaceId })
        .map((row) => row.name)
        .sort()
    expect(names(PLATFORM)).toEqual(["Wei", "Zoe"])
    expect(names(MOBILE)).toEqual(["Hana", "Zoe"])
    expect(names("ws-empty")).toEqual([])
    expect(names()).toEqual(["Hana", "Liam", "Wei", "Zoe"])
  })

  it("leaves out sessions in another organization's workspaces", () => {
    const [row] = personRows([member("hana")], {
      ...EMPTY,
      sessions: [session("user-hana"), session("user-hana", "ws-elsewhere")],
      workspaceIds: new Set([PLATFORM, MOBILE]),
    })
    expect(row?.sessions).toBe(1)
  })

  it("reports nothing for someone outside the caller's workspaces", () => {
    const hana = member("hana", {
      workspaces: [
        {
          workspace_id: MOBILE,
          workspace_name: "Mobile",
          workspace_member_id: "wm-hana",
          role: "member",
        },
      ],
    })
    const rows = personRows([hana, member("wei")], {
      ...EMPTY,
      spendRows: [spendRow("hana", 27.65, 9), spendRow("wei", 1)],
      sessions: [session("user-hana", MOBILE)],
      visibleWorkspaceIds: new Set([PLATFORM]),
    })
    expect(rows.map((row) => [row.name, row.isOutsideView])).toEqual([
      ["Wei", false],
      ["Hana", true],
    ])
    expect(rows[1]).toMatchObject({
      spend: 0,
      tokens: 0,
      sessions: 0,
      spendRow: undefined,
    })
  })

  it("keeps active and invited people, invitations last, and drops the rest", () => {
    const rows = personRows(
      [
        member("ana", {
          status: "invited",
          organization_member_id: null,
          invitation_id: "inv-ana",
          user_id: null,
          attribution_user_id: null,
        }),
        member("gone", { status: "suspended" }),
        member("wei"),
        member("zainab"),
      ],
      {
        ...EMPTY,
        spendRows: [spendRow("zainab", 24.42), spendRow("wei", 13.16)],
      },
    )
    expect(rows.map((row) => [row.id, row.isInvited])).toEqual([
      ["om-zainab", false],
      ["om-wei", false],
      ["inv-ana", true],
    ])
  })
})

describe("rosterLimits", () => {
  it("lists the amount on the person in each team the roster gives, by team", () => {
    const placed = member("ethan", {
      workspaces: [
        {
          workspace_id: PLATFORM,
          workspace_name: "Platform",
          workspace_member_id: "wm-1",
          role: "member",
          ceiling: { id: "c-1", budget_id: "b-1", max_budget: 40 },
        },
        {
          workspace_id: "ws-data",
          workspace_name: "Data",
          workspace_member_id: "wm-2",
          role: "member",
          ceiling: { id: "c-2", budget_id: "b-2", max_budget: 25 },
        },
        {
          workspace_id: MOBILE,
          workspace_name: "Mobile",
          workspace_member_id: "wm-3",
          role: "member",
          ceiling: null,
        },
      ],
    })
    expect(rosterLimits(placed)).toEqual([
      { workspaceId: "ws-data", team: "Data", amount: 25 },
      { workspaceId: PLATFORM, team: "Platform", amount: 40 },
    ])
  })

  it("is empty for someone with no limit anywhere", () => {
    expect(rosterLimits(member("wei"))).toEqual([])
  })

  it("keeps a $0 limit, which refuses every request", () => {
    const blocked = member("ethan", {
      workspaces: [
        {
          workspace_id: PLATFORM,
          workspace_name: "Platform",
          workspace_member_id: "wm-1",
          role: "member",
          ceiling: { id: "c-1", budget_id: "b-1", max_budget: 0 },
        },
        {
          workspace_id: MOBILE,
          workspace_name: "Mobile",
          workspace_member_id: "wm-2",
          role: "member",
          ceiling: { id: "c-2", budget_id: "b-2", max_budget: null },
        },
      ],
    })
    expect(rosterLimits(blocked)).toEqual([
      { workspaceId: PLATFORM, team: "Platform", amount: 0 },
    ])
  })
})
