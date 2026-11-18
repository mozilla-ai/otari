import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { type LimitTarget, perWords } from "@/features/spend/limits"
import { SetLimitDialog } from "@/features/spend/SetLimitDialog"
import { API_ROOT } from "@/shared/api/client"
import { organizationSpendCeiling } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const TEN_DAYS = organizationSpendCeiling({
  scope_type: "org_member",
  scope_id: "u-wei",
  budget_id: "b-wei",
  max_budget: 30,
  reset_alignment: null,
  budget_duration_sec: 10 * 86_400,
})

/** Wei's own limit, on `ceiling`. */
const limitOn = (ceiling: typeof TEN_DAYS): LimitTarget => ({
  scope: "org_member",
  scopeId: "u-wei",
  name: "Wei Zhang",
  own: { ceiling, max: 30, spent: 4, per: perWords(ceiling) },
})

const target = limitOn(TEN_DAYS)

/**
 * Wei's budget, `b-wei`, named by `ceilingCount` ceilings: Wei's alone by
 * default, so its amount changes in place. A budget made here is `b-new`.
 */
function mockApi({ ceilingCount = 1 }: { ceilingCount?: number } = {}) {
  const writes: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const method = (init?.method ?? "GET").toUpperCase()
    const path = String(input).replace(API_ROOT, "").split("?")[0]
    if (method !== "GET") {
      writes.push(`${method} ${path}`)
      return Response.json({
        budget_id: path === "/organizations/me/budgets" ? "b-new" : "b-wei",
      })
    }
    if (path === "/organizations/me/budgets") {
      return Response.json({
        data: [{ budget_id: "b-wei", ceiling_count: ceilingCount }],
        count: 1,
      })
    }
    return Response.json({ data: [], count: 0 })
  })
  return writes
}

async function renderDialog(limitTarget: LimitTarget = target) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  await renderWithRouter(
    <QueryClientProvider client={client}>
      <SetLimitDialog
        target={limitTarget}
        title="Spend limit for Wei Zhang"
        description="Wei"
        onClose={() => {}}
      />
    </QueryClientProvider>,
  )
  return userEvent.setup()
}

describe("SetLimitDialog on a period the form cannot name", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("says the period it runs on rather than showing Month", async () => {
    mockApi()
    await renderDialog()
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByText(/It resets every 10 days now/),
    ).toBeVisible()
    for (const radio of within(dialog).getAllByRole("radio")) {
      expect(radio).not.toBeChecked()
    }
  })

  it("calls a rolling week 7 days, not a week", async () => {
    mockApi()
    await renderDialog(
      limitOn({ ...TEN_DAYS, budget_duration_sec: 7 * 86_400 }),
    )
    expect(
      within(await screen.findByRole("dialog")).getByText(
        /It resets every 7 days now/,
      ),
    ).toBeVisible()
  })

  it("changes only the amount when no period is picked", async () => {
    const writes = mockApi()
    const user = await renderDialog()
    const dialog = await screen.findByRole("dialog")
    const amount = within(dialog).getByRole("textbox", { name: /spend limit/i })
    await user.clear(amount)
    await user.type(amount, "45")
    await user.click(within(dialog).getByRole("button", { name: "Save limit" }))
    await waitFor(() =>
      expect(writes).toEqual(["PATCH /organizations/me/budgets/b-wei"]),
    )
  })
})

describe("SetLimitDialog on a budget other limits share", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  // Wei is on a $50 weekly budget three ceilings share, with $45 spent.
  const shared = organizationSpendCeiling({
    id: "c-wei",
    scope_type: "org_member",
    scope_id: "u-wei",
    budget_id: "b-wei",
    max_budget: 50,
    current_spend: 45,
    reset_alignment: "calendar_week",
    budget_duration_sec: null,
  })

  it("keeps what Wei has spent when only the amount changes", async () => {
    const writes = mockApi({ ceilingCount: 3 })
    const user = await renderDialog({
      scope: "org_member",
      scopeId: "u-wei",
      name: "Wei Zhang",
      own: { ceiling: shared, max: 50, spent: 45, per: perWords(shared) },
    })
    const dialog = await screen.findByRole("dialog")
    const amount = within(dialog).getByRole("textbox", { name: /spend limit/i })
    await user.clear(amount)
    await user.type(amount, "60")
    await user.click(within(dialog).getByRole("button", { name: "Save limit" }))
    // The ceiling holding the $45 is moved onto a budget of its own, never
    // deleted and made again from nothing.
    await waitFor(() =>
      expect(writes).toEqual([
        "POST /organizations/me/budgets",
        "PATCH /organizations/me/spend-ceilings/c-wei",
      ]),
    )
  })
})

describe("SetLimitDialog on a shorter period", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  // Wei is on a $100 monthly budget of their own, with $45 spent this month.
  const monthly = organizationSpendCeiling({
    id: "c-wei",
    scope_type: "org_member",
    scope_id: "u-wei",
    budget_id: "b-wei",
    max_budget: 100,
    current_spend: 45,
    reset_alignment: "calendar_month",
    budget_duration_sec: null,
  })

  /**
   * Wei's ceiling as the server keeps it: moving it onto another budget keeps
   * what it has recorded, and one made new has recorded nothing.
   */
  function mockCeiling() {
    const state: {
      ceiling?: { id: string; budget: string; spent: number }
    } = { ceiling: { id: "c-wei", budget: "b-wei", spent: 45 } }
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const method = (init?.method ?? "GET").toUpperCase()
      const path = String(input).replace(API_ROOT, "").split("?")[0]
      const body = init?.body ? JSON.parse(String(init.body)) : {}
      if (method === "GET" && path === "/organizations/me/budgets") {
        return Response.json({
          data: [{ budget_id: "b-wei", ceiling_count: 1 }],
          count: 1,
        })
      }
      if (method === "POST" && path === "/organizations/me/budgets") {
        return Response.json({ budget_id: "b-week" })
      }
      if (
        method === "PATCH" &&
        path === "/organizations/me/spend-ceilings/c-wei"
      ) {
        if (state.ceiling) state.ceiling.budget = body.budget_id
        return Response.json({ id: "c-wei", budget_id: body.budget_id })
      }
      if (
        method === "DELETE" &&
        path === "/organizations/me/spend-ceilings/c-wei"
      ) {
        state.ceiling = undefined
      }
      if (method === "POST" && path === "/organizations/me/spend-ceilings") {
        state.ceiling = { id: "c-new", budget: body.budget_id, spent: 0 }
        return Response.json({ id: "c-new", budget_id: body.budget_id })
      }
      return Response.json({ data: [], count: 0 })
    })
    return state
  }

  it("carries the month's spend to a weekly limit, and says so", async () => {
    const state = mockCeiling()
    const user = await renderDialog({
      scope: "org_member",
      scopeId: "u-wei",
      name: "Wei Zhang",
      own: { ceiling: monthly, max: 100, spent: 45, per: perWords(monthly) },
    })
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByText(
        "What they've spent this period still counts against the new limit.",
      ),
    ).toBeVisible()
    await user.click(within(dialog).getByRole("radio", { name: "Week" }))
    await user.click(within(dialog).getByRole("button", { name: "Save limit" }))

    await waitFor(() =>
      expect(state.ceiling).toEqual({
        id: "c-wei",
        budget: "b-week",
        spent: 45,
      }),
    )
  })
})
