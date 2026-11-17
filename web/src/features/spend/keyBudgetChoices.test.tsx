import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { useKeyBudgetChoices } from "@/features/spend/keyBudgetChoices"
import { API_ROOT } from "@/shared/api/client"
import { budget, workspace, workspaceBudgetDefault } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

const BUDGETS = [
  budget({ budget_id: "b-default", name: "Each Data member, weekly" }),
  budget({ budget_id: "b-team", name: "Team standard" }),
]

/** One workspace; its defaults read answers `defaults` (a status, or held). */
function mockApi(defaults: "ok" | "failed" | "held") {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url.includes("/workspaces/ws-data/member-budget-policies")) {
      if (defaults === "held") return new Promise<Response>(() => undefined)
      if (defaults === "failed") return json({ detail: "Unavailable" }, 500)
      const rows = [
        workspaceBudgetDefault({
          workspace_id: "ws-data",
          budget_id: "b-default",
        }),
      ]
      return json({ data: rows, count: rows.length })
    }
    if (url.startsWith(`${API_ROOT}/workspaces?`)) {
      return json({ data: [workspace({ id: "ws-data" })], count: 1 })
    }
    return json([])
  })
}

type Choices = ReturnType<typeof useKeyBudgetChoices>

async function renderChoices() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const result = { current: undefined as unknown as Choices }
  function Probe() {
    result.current = useKeyBudgetChoices(
      { budgets: BUDGETS, isLoading: false },
      true,
    )
    return null
  }
  await renderWithRouter(
    <QueryClientProvider client={client}>
      <Probe />
    </QueryClientProvider>,
  )
  return result
}

const ids = (choices: Choices) => choices.budgets.map((one) => one.budget_id)

describe("useKeyBudgetChoices", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("leaves out a budget that backs a workspace default, and keeps the rest", async () => {
    mockApi("ok")
    const result = await renderChoices()

    await waitFor(() => expect(result.current.isLoading).toBe(false))
    expect(ids(result.current)).toEqual(["b-team"])
  })

  it("stays loading until the defaults are read", async () => {
    mockApi("held")
    const result = await renderChoices()

    await waitFor(() =>
      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/member-budget-policies"),
        expect.anything(),
      ),
    )
    expect(result.current.isLoading).toBe(true)
  })

  it("offers no existing budget when the defaults cannot be read", async () => {
    mockApi("failed")
    const result = await renderChoices()

    await waitFor(() =>
      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/member-budget-policies"),
        expect.anything(),
      ),
    )
    await waitFor(() => expect(result.current.isLoading).toBe(false))
    expect(ids(result.current)).toEqual([])
  })
})
