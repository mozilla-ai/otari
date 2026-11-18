import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { waitFor } from "@testing-library/react"
import { act } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { type LimitTarget, useLimitWriter } from "@/features/spend/limits"
import { API_ROOT } from "@/shared/api/client"
import { organizationSpendCeiling } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

const CEILING = organizationSpendCeiling({
  id: "ceiling-old",
  budget_id: "budget-old",
  scope_type: "org_member",
  scope_id: "m1",
  max_budget: 50,
})

const TARGET: LimitTarget = {
  scope: "org_member",
  scopeId: "m1",
  name: "Wei Zhang",
  own: { ceiling: CEILING, max: 50, spent: 0, per: "a month" },
}

/**
 * The organization's budget and ceiling routes. `budgets` is the list read
 * (`"failed"` answers 500), each budget's `ceiling_count` as it starts; the
 * mock keeps the counts as ceilings move. `budgetsDelay` holds the read's
 * answer back that many milliseconds; `failCeilingUpdates` refuses that many
 * ceiling moves before one succeeds. A scope holds one ceiling at a time, as on
 * the server: a create on a scope that has one answers 409. Every write is
 * recorded as "METHOD path", and `scopes` is which ceiling each scope holds and
 * which budget it names.
 */
function mockApi({
  budgets,
  budgetsDelay = 0,
  failCeilingUpdates = 0,
}: {
  budgets: { budget_id: string; ceiling_count: number }[] | "failed"
  budgetsDelay?: number
  failCeilingUpdates?: number
}) {
  const writes: string[] = []
  // How many times the budgets read has been answered: what a writer waits on
  // before it decides whether a budget is this limit's alone.
  const served = { budgets: 0 }
  const scopes = new Map([["m1", { id: "ceiling-old", budget: "budget-old" }]])
  const counts = new Map(
    budgets === "failed"
      ? []
      : budgets.map((one) => [one.budget_id, one.ceiling_count]),
  )
  const move = (from: string | undefined, to: string | undefined) => {
    if (from !== undefined) counts.set(from, (counts.get(from) ?? 1) - 1)
    if (to !== undefined) counts.set(to, (counts.get(to) ?? 0) + 1)
  }
  let refusals = failCeilingUpdates
  let made = 0
  let ceilingsMade = 0
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const path = url.replace(API_ROOT, "").split("?")[0]
    const id = path.split("/").pop()
    if (method === "GET") {
      if (path === "/organizations/me/budgets") {
        await new Promise((resolve) => setTimeout(resolve, budgetsDelay))
        served.budgets += 1
        if (budgets === "failed") {
          return json({ detail: "Budgets are unavailable" }, 500)
        }
        const data = [...counts].map(([budget_id, ceiling_count]) => ({
          budget_id,
          ceiling_count,
        }))
        return json({ data, count: data.length })
      }
      return json([])
    }
    writes.push(`${method} ${path}`)
    const body = init?.body ? JSON.parse(String(init.body)) : {}
    if (method === "POST" && path === "/organizations/me/budgets") {
      made += 1
      const budget_id = `budget-new-${made}`
      counts.set(budget_id, 0)
      return json({ budget_id }, 201)
    }
    if (method === "DELETE" && path.startsWith("/organizations/me/budgets/")) {
      counts.delete(id ?? "")
    }
    if (method === "POST" && path === "/organizations/me/spend-ceilings") {
      if (scopes.has(body.scope_id)) {
        return json({ detail: "A spend ceiling already exists" }, 409)
      }
      ceilingsMade += 1
      const made = { id: `ceiling-new-${ceilingsMade}`, budget: body.budget_id }
      scopes.set(body.scope_id, made)
      move(undefined, body.budget_id)
      return json({ id: made.id, budget_id: body.budget_id }, 201)
    }
    if (
      method === "PATCH" &&
      path.startsWith("/organizations/me/spend-ceilings/")
    ) {
      if (refusals > 0) {
        refusals -= 1
        return json({ detail: "Spend ceilings are unavailable" }, 500)
      }
      for (const ceiling of scopes.values()) {
        if (ceiling.id !== id) continue
        move(ceiling.budget, body.budget_id)
        ceiling.budget = body.budget_id
      }
      return json({ id, budget_id: body.budget_id })
    }
    if (
      method === "DELETE" &&
      path.startsWith("/organizations/me/spend-ceilings/")
    ) {
      for (const [scope, ceiling] of scopes) {
        if (ceiling.id !== id) continue
        scopes.delete(scope)
        move(ceiling.budget, undefined)
      }
    }
    return json({ message: "ok" })
  })
  return { writes, scopes, served }
}

type Writer = ReturnType<typeof useLimitWriter>

/** The writer as a dialog holds it, mounted once for the whole test. */
async function renderWriter() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  const result = { current: undefined as unknown as Writer }
  function Probe() {
    result.current = useLimitWriter()
    return null
  }
  await renderWithRouter(
    <QueryClientProvider client={client}>
      <Probe />
    </QueryClientProvider>,
  )
  return { result, client }
}

describe("useLimitWriter", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("never edits a budget in place when the budgets cannot be read", async () => {
    const { writes, scopes } = mockApi({ budgets: "failed" })
    const { result } = await renderWriter()

    let isDone = false
    await act(async () => {
      isDone = await result.current.save(TARGET, 80, "calendar_month")
    })

    expect(isDone).toBe(true)
    expect(writes).not.toContain("PATCH /organizations/me/budgets/budget-old")
    expect(writes).toEqual([
      "POST /organizations/me/budgets",
      "PATCH /organizations/me/spend-ceilings/ceiling-old",
    ])
    expect(scopes.get("m1")).toEqual({
      id: "ceiling-old",
      budget: "budget-new-1",
    })
  })

  it("moves a limit on a shared budget onto one of its own, keeping its ceiling and so its spend", async () => {
    const { writes, scopes, served } = mockApi({
      budgets: [{ budget_id: "budget-old", ceiling_count: 3 }],
    })
    const { result } = await renderWriter()
    await waitFor(() => expect(served.budgets).toBeGreaterThan(0))

    let isDone = false
    await act(async () => {
      isDone = await result.current.save(TARGET, 60, "calendar_week")
    })

    expect(isDone).toBe(true)
    // Never a delete: the scope is capped throughout, and the ceiling that
    // carries what it has spent this period is the one that stays.
    expect(writes).toEqual([
      "POST /organizations/me/budgets",
      "PATCH /organizations/me/spend-ceilings/ceiling-old",
    ])
    expect(scopes.get("m1")).toEqual({
      id: "ceiling-old",
      budget: "budget-new-1",
    })
  })

  it("waits for the budgets read before removing, and deletes a budget only this limit used", async () => {
    const { writes, served } = mockApi({
      budgets: [{ budget_id: "budget-old", ceiling_count: 1 }],
      budgetsDelay: 50,
    })
    const { result } = await renderWriter()
    expect(served.budgets).toBe(0)

    let isDone = false
    await act(async () => {
      isDone = await result.current.remove(TARGET)
    })

    expect(isDone).toBe(true)
    expect(writes).toEqual([
      "DELETE /organizations/me/spend-ceilings/ceiling-old",
      "DELETE /organizations/me/budgets/budget-old",
    ])
  })

  it("edits a budget in place once the read says only this limit uses it", async () => {
    const { writes, served } = mockApi({
      budgets: [{ budget_id: "budget-old", ceiling_count: 1 }],
    })
    const { result } = await renderWriter()
    await waitFor(() => expect(served.budgets).toBeGreaterThan(0))

    await act(async () => {
      await result.current.save(TARGET, 80, "calendar_month")
    })

    expect(writes).toEqual(["PATCH /organizations/me/budgets/budget-old"])
  })

  it("changes the period by moving the ceiling, then deletes the budget only this limit used", async () => {
    const { writes, scopes, served } = mockApi({
      budgets: [{ budget_id: "budget-old", ceiling_count: 1 }],
    })
    const { result } = await renderWriter()
    await waitFor(() => expect(served.budgets).toBeGreaterThan(0))

    let isDone = false
    await act(async () => {
      isDone = await result.current.replace(TARGET, 80, "calendar_week")
    })

    expect(isDone).toBe(true)
    expect(writes).toEqual([
      "POST /organizations/me/budgets",
      "PATCH /organizations/me/spend-ceilings/ceiling-old",
      "DELETE /organizations/me/budgets/budget-old",
    ])
    expect(scopes.get("m1")).toEqual({
      id: "ceiling-old",
      budget: "budget-new-1",
    })
  })

  it("leaves the old limit standing when the ceiling cannot be moved, and drops the budget made for it", async () => {
    const { writes, scopes, served } = mockApi({
      budgets: [{ budget_id: "budget-old", ceiling_count: 1 }],
      failCeilingUpdates: 1,
    })
    const { result } = await renderWriter()
    await waitFor(() => expect(served.budgets).toBeGreaterThan(0))

    let isDone = true
    await act(async () => {
      isDone = await result.current.replace(TARGET, 80, "calendar_week")
    })

    expect(isDone).toBe(false)
    expect(result.current.error).toBeDefined()
    expect(writes).toEqual([
      "POST /organizations/me/budgets",
      "PATCH /organizations/me/spend-ceilings/ceiling-old",
      "DELETE /organizations/me/budgets/budget-new-1",
    ])
    expect(scopes.get("m1")).toEqual({
      id: "ceiling-old",
      budget: "budget-old",
    })

    writes.length = 0
    await act(async () => {
      isDone = await result.current.replace(TARGET, 80, "calendar_week")
    })
    expect(isDone).toBe(true)
    expect(writes).toEqual([
      "POST /organizations/me/budgets",
      "PATCH /organizations/me/spend-ceilings/ceiling-old",
      "DELETE /organizations/me/budgets/budget-old",
    ])
  })

  it("after a move, changes the limit's new budget and not the one it left", async () => {
    // Wei and one other person share budget-old; once Wei moves off it, the
    // list says one ceiling names it, and that ceiling is not Wei's.
    const { writes, served } = mockApi({
      budgets: [{ budget_id: "budget-old", ceiling_count: 2 }],
    })
    const { result } = await renderWriter()
    await waitFor(() => expect(served.budgets).toBeGreaterThan(0))
    await act(async () => {
      await result.current.save(TARGET, 60, "calendar_week")
    })
    const before = served.budgets
    await waitFor(() => expect(served.budgets).toBeGreaterThan(before))

    writes.length = 0
    await act(async () => {
      // The caller still holds the limit as it was before the move.
      await result.current.save(TARGET, 70, "calendar_week")
    })

    expect(writes).toEqual(["PATCH /organizations/me/budgets/budget-new-1"])
  })
})
