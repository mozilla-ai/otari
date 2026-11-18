import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { UsageGroupRow } from "@/client"
import { limitsOn, NO_LIMITS } from "@/features/spend/limits"
import { windowOf } from "@/features/spend/period"
import {
  keyRows,
  modelRows,
  peopleRows,
  providerRows,
  type Spend,
  useSpend,
  useWorkspaceSpend,
} from "@/features/spend/useSpend"
import { API_ROOT } from "@/shared/api/client"
import {
  apiKey,
  organizationContext,
  organizationMember,
  organizationSpendCeiling,
  usageTotals,
  workspace,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const group = (overrides: Partial<UsageGroupRow>): UsageGroupRow => ({
  key: null,
  label: null,
  cost: 1,
  tokens: 10,
  requests: 1,
  is_other: false,
  ...overrides,
})

/** The server's fold of its smallest rows, beside a real row with no key. */
const FOLD = group({ is_other: true, cost: 3 })
const NOBODY = group({ cost: 2 })

describe("the server's fold row", () => {
  it("is Other, with a key of its own, on every cut", () => {
    const models = modelRows([FOLD, NOBODY])
    expect(models.map((row) => [row.id, row.name])).toEqual([
      ["model::other", "Other"],
      ["model::none", "Unknown model"],
    ])
    const keys = keyRows(
      [FOLD, NOBODY],
      [],
      () => undefined,
      () => NO_LIMITS,
      false,
    )
    expect(keys.map((row) => [row.id, row.name])).toEqual([
      ["key:other", "Other"],
      ["key:none", "No key"],
    ])
    const people = peopleRows([FOLD, NOBODY], [], () => NO_LIMITS, false)
    expect(people.map((row) => [row.id, row.name])).toEqual([
      ["user:other", "Other"],
      ["user:none", "No person"],
    ])
    expect(new Set(people.map((row) => row.id)).size).toBe(people.length)
  })

  it("is never claimed by a member with no id of their own", () => {
    const member = organizationMember({
      user_id: null,
      attribution_user_id: null,
      status: "active",
    })
    const rows = peopleRows([NOBODY], [member], () => NO_LIMITS, false)
    expect(rows.find((row) => row.isMember)?.cost).toBe(0)
    expect(rows.find((row) => row.id === "user:none")?.cost).toBe(2)
  })
})

describe("providerRows", () => {
  const usage = [
    group({ key: "anthropic", cost: 3, requests: 2 }),
    group({ key: "openai", cost: 6, requests: 4 }),
    // Requests that recorded no provider: imported, or refused before one.
    group({ key: null, cost: 0.5, requests: 3 }),
    group({ key: "acme-proxy", cost: 0.25 }),
    FOLD,
  ]

  it("names each provider, biggest spend first, by the name the dashboard gives it", () => {
    const rows = providerRows(usage)
    expect(rows.map((row) => [row.id, row.name, row.filterKey])).toEqual([
      ["provider:openai", "OpenAI", "openai"],
      ["provider:anthropic", "Anthropic", "anthropic"],
      // The server's fold, at $3 like Anthropic: a tie goes by name.
      ["provider::other", "Other", null],
      ["provider::none", "Unknown provider", null],
      // An instance an operator named keeps the name they gave it.
      ["provider:acme-proxy", "acme-proxy", "acme-proxy"],
    ])
  })

  it("keeps the requests that recorded no provider as one row, so the rows add up to the total", () => {
    const rows = providerRows(usage)
    const unknown = rows.find((row) => row.id === "provider::none")
    expect(unknown?.detail).toBe("Not recorded")
    expect(unknown?.cost).toBe(0.5)
    expect(rows.reduce((sum, row) => sum + row.cost, 0)).toBe(12.75)
    const shares = rows.map((row) => row.cost / 12.75)
    expect(shares.reduce((sum, share) => sum + share, 0)).toBeCloseTo(1)
  })

  it("sets no limit on a provider and offers none to set", () => {
    for (const row of providerRows(usage)) {
      expect(row.limits).toBe(NO_LIMITS)
      expect(row.target).toBeNull()
    }
  })

  it("keeps an instance named like the aggregate rows apart from them", () => {
    const rows = providerRows([
      group({ key: "other", cost: 2 }),
      group({ key: "none", cost: 1 }),
      group({ key: null, cost: 0.5 }),
      FOLD,
    ])
    const ids = rows.map((row) => row.id)
    expect(new Set(ids).size).toBe(ids.length)
    expect(rows.find((row) => row.id === "provider:other")?.filterKey).toBe(
      "other",
    )
  })

  it("leaves out a provider with no requests and no spend", () => {
    const rows = providerRows([
      group({ key: "openai", cost: 1 }),
      group({ key: "mistral", cost: 0, requests: 0 }),
    ])
    expect(rows.map((row) => row.id)).toEqual(["provider:openai"])
  })
})

describe("a spender past the summary's top rows", () => {
  const top = group({ key: "user-ana", cost: 9 })
  const ana = organizationMember({
    organization_member_id: "om-ana",
    user_id: "user-ana",
    attribution_user_id: null,
    status: "active",
  })
  const wei = organizationMember({
    organization_member_id: "om-wei",
    user_id: "user-wei",
    attribution_user_id: null,
    status: "active",
  })

  it("is a person whose spend is unknown, not $0, while the summary folded some", () => {
    const rows = peopleRows([top, FOLD], [ana, wei], () => NO_LIMITS, false)
    const byId = new Map(rows.map((row) => [row.id, row]))
    expect(byId.get("person:om-wei")?.isFolded).toBe(true)
    expect(byId.get("person:om-ana")?.isFolded).toBeFalsy()
    expect(byId.get("person:om-ana")?.cost).toBe(9)
  })

  it("is a person who spent nothing when the summary folded none", () => {
    const rows = peopleRows([top], [ana, wei], () => NO_LIMITS, false)
    const weiRow = rows.find((row) => row.id === "person:om-wei")
    expect(weiRow?.isFolded).toBeFalsy()
    expect(weiRow?.cost).toBe(0)
  })

  it("is a key whose spend is unknown, not $0, while the summary folded some", () => {
    const keys = [
      apiKey({ id: "k-top", is_active: true, expires_at: null }),
      apiKey({ id: "k-quiet", is_active: true, expires_at: null }),
    ]
    const folded = keyRows(
      [group({ key: "k-top", cost: 9 }), FOLD],
      keys,
      () => undefined,
      () => NO_LIMITS,
      false,
    )
    const byId = new Map(folded.map((row) => [row.id, row]))
    expect(byId.get("key:k-quiet")?.isFolded).toBe(true)
    expect(byId.get("key:k-top")?.isFolded).toBeFalsy()

    const whole = keyRows(
      [group({ key: "k-top", cost: 9 })],
      keys,
      () => undefined,
      () => NO_LIMITS,
      false,
    )
    expect(whole.find((row) => row.id === "key:k-quiet")?.isFolded).toBeFalsy()
  })
})

describe("a key past its expiry", () => {
  const expired = apiKey({
    id: "k-old",
    key_name: "old-bot",
    is_active: true,
    expires_at: "2026-11-10T00:00:00Z",
  })
  const NOW = Date.parse("2026-11-17T12:00:00Z")

  it("keeps the month's spend it made, and is offered no limit to set", () => {
    const [row] = keyRows(
      [group({ key: "k-old", cost: 7 })],
      [expired],
      () => undefined,
      () => NO_LIMITS,
      true,
      { now: NOW },
    )
    expect(row?.cost).toBe(7)
    expect(row?.target).toBeNull()
  })

  it("is offered a limit while it can still be used", () => {
    const [row] = keyRows(
      [group({ key: "k-old", cost: 7 })],
      [expired],
      () => undefined,
      () => NO_LIMITS,
      true,
      { now: Date.parse("2026-11-05T00:00:00Z") },
    )
    expect(row?.target).not.toBeNull()
  })
})

describe("a key the caller does not list", () => {
  it("reads its own limit from the ceilings rather than no limit", () => {
    const ceilings = [
      organizationSpendCeiling({
        id: "c-teammate",
        scope_type: "api_token",
        scope_id: "k-teammate",
        max_budget: 40,
        current_spend: 10,
      }),
    ]
    const [row] = keyRows(
      [group({ key: "k-teammate", label: "ana-laptop" })],
      [],
      () => undefined,
      (id) => limitsOn(ceilings, "api_token", id),
      true,
    )
    expect(row?.name).toBe("ana-laptop")
    expect(row?.limits.binding?.max).toBe(40)
    // Not this page's key to cap: the row offers no limit to set.
    expect(row?.target).toBeNull()
  })
})

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

const summary = (cost: number) => ({
  totals: usageTotals({ cost, request_count: 1 }),
  series: [],
  by_user: [],
  by_api_key: [],
  by_model: [],
  by_provider: [
    { key: "anthropic", label: null, cost, tokens: 10, requests: 1 },
  ],
  errors_by_status_code: [],
})

const PLATFORM = workspace({ id: "w-platform", name: "Platform" })
const DATA = workspace({ id: "w-data", name: "Data" })

/**
 * An owner's view of the control plane. Each `fail` names a read that answers
 * 500; `pending` names one that never answers.
 */
function mockApi({
  fail = [],
  pending = [],
}: {
  fail?: string[]
  pending?: string[]
}) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), "http://test")
    const path = url.pathname.replace(
      new URL(API_ROOT, "http://test").pathname,
      "",
    )
    const read =
      path === "/organizations/me/usage/summary"
        ? url.searchParams.get("end_date")
          ? "previous"
          : url.searchParams.get("workspace_id") === DATA.id
            ? "data-workspace"
            : "usage"
        : path
    if (pending.includes(read)) return new Promise<Response>(() => {})
    if (fail.includes(read)) {
      return json({ detail: `${read} is unavailable` }, 500)
    }
    if (path === "/organizations/me") {
      return json(organizationContext({ deployment_operator: false }))
    }
    if (path === "/organizations/me/members") {
      return json({
        data: [
          organizationMember({ attribution_user_id: "u1", status: "active" }),
        ],
        count: 1,
      })
    }
    if (path === "/organizations/me/keys") return json([apiKey()])
    if (path === "/organizations/me/spend-ceilings") {
      return json({ data: [], count: 0 })
    }
    if (path === "/workspaces")
      return json({ data: [PLATFORM, DATA], count: 2 })
    if (read === "previous") return json(summary(4))
    return json(summary(5))
  })
}

const NOW = Date.parse("2026-11-16T12:00:00Z")
const WINDOW = windowOf("month", NOW)

async function renderSpend() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const result = {
    spend: undefined as unknown as Spend,
    byWorkspace: undefined as unknown as ReturnType<typeof useWorkspaceSpend>,
  }
  function Probe() {
    result.spend = useSpend(WINDOW, { resetEndedAt: NOW })
    result.byWorkspace = useWorkspaceSpend(WINDOW, true, NOW)
    return null
  }
  await renderWithRouter(
    <QueryClientProvider client={client}>
      <Probe />
    </QueryClientProvider>,
  )
  return result
}

describe("useSpend", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("reports a failed key list, and waits on the previous window", async () => {
    mockApi({ fail: ["/organizations/me/keys"], pending: ["previous"] })
    const result = await renderSpend()
    await waitFor(() => expect(result.spend.totals).toBeDefined())
    await waitFor(() => expect(result.spend.error).toBeTruthy())
    expect(String(result.spend.error)).toContain(
      "/organizations/me/keys is unavailable",
    )
    expect(result.spend.isLoading).toBe(true)
  })

  it("leaves limits unknown, with nothing to set, when the ceilings cannot be read", async () => {
    mockApi({ fail: ["/organizations/me/spend-ceilings"] })
    const result = await renderSpend()
    await waitFor(() => expect(result.spend.reads.limits).toBe("failed"))
    await waitFor(() => expect(result.spend.people.length).toBeGreaterThan(0))
    const [person] = result.spend.people
    expect(person?.limits.isUnknown).toBe(true)
    expect(person?.target).toBeNull()
    expect(result.spend.organization.target).toBeNull()
  })

  it("reads the providers in the summary it already asks for", async () => {
    mockApi({})
    const result = await renderSpend()
    await waitFor(() => expect(result.spend.providers).toHaveLength(1))
    expect(result.spend.providers[0]?.name).toBe("Anthropic")
    const windowReads = vi
      .mocked(globalThis.fetch)
      .mock.calls.map(([input]) => new URL(String(input), "http://test"))
      .filter(
        (url) =>
          url.pathname.endsWith("/organizations/me/usage/summary") &&
          !url.searchParams.has("end_date") &&
          // Not the per-workspace reads the probe's other hook makes.
          !url.searchParams.has("workspace_id"),
      )
    // One read for the window, carrying every breakdown the page cuts by.
    expect(windowReads).toHaveLength(1)
    expect(windowReads[0]?.searchParams.getAll("dimensions")).toEqual([
      "user",
      "api_key",
      "model",
      "provider",
    ])
  })

  it("marks figures unread, not zero, when usage cannot be read", async () => {
    mockApi({ fail: ["usage"] })
    const result = await renderSpend()
    await waitFor(() => expect(result.spend.reads.usage).toBe("failed"))
    await waitFor(() => expect(result.spend.people.length).toBeGreaterThan(0))
    expect(result.spend.people.every((row) => row.isUnread)).toBe(true)
  })
})

describe("useWorkspaceSpend", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("reports a workspace it could not read rather than $0.00", async () => {
    mockApi({ fail: ["data-workspace"] })
    const result = await renderSpend()
    await waitFor(() => expect(result.byWorkspace.error).toBeTruthy())
    await waitFor(() =>
      expect(
        result.byWorkspace.rows.find((row) => row.name === "Platform")?.cost,
      ).toBe(5),
    )
    const data = result.byWorkspace.rows.find((row) => row.name === "Data")
    expect(data?.isUnread).toBe(true)
    expect(String(result.byWorkspace.error)).toContain(
      "data-workspace is unavailable",
    )
  })
})
