import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationBudget, OrganizationSpendCeiling } from "@/client"
import {
  BudgetDetailPage,
  OrganizationBudgetDetail,
} from "@/features/budgets/BudgetDetailPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organization,
  organizationContext,
  organizationSpendCeiling as spendCeiling,
  workspace,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

const BUDGET_ID = "bbbbbbbb-1111-2222-3333-444444444444"

function budget(
  overrides: Partial<OrganizationBudget> = {},
): OrganizationBudget {
  return {
    budget_id: BUDGET_ID,
    organization_id: organization().id,
    name: "Engineering monthly",
    max_budget: 250,
    token_limit: null,
    request_limit: null,
    reset_cycle: "monthly",
    reset_every_n: null,
    reset_anchor_at: null,
    reset_weekdays: null,
    reset_month_day: 1,
    reset_month: null,
    ceiling_count: 2,
    applied_to: [],
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    ...overrides,
  }
}

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })

function mockApi({
  budgets = [budget()],
  ceilings = [
    spendCeiling({
      id: "c-org",
      budget_id: BUDGET_ID,
      current_spend: 12.5,
      reserved_spend: 2,
    }),
    spendCeiling({
      id: "c-ws",
      budget_id: BUDGET_ID,
      scope_type: "workspace",
      scope_id: workspace().id,
      current_spend: 240,
    }),
    // Another budget's ceiling: not this page's to show.
    spendCeiling({
      id: "c-other",
      budget_id: "other",
      scope_type: "workspace",
      scope_id: "ws-x",
    }),
  ],
}: {
  budgets?: OrganizationBudget[]
  ceilings?: OrganizationSpendCeiling[]
} = {}) {
  const requests: { url: string; method: string }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({ url, method })
    if (url.includes(`${API_ROOT}/organizations/me/spend-ceilings`)) {
      // As the server does: only the ceilings applying the budget asked about.
      const asked = new URL(url, "http://test").searchParams.get("budget_id")
      const held = ceilings.filter((ceiling) => ceiling.budget_id === asked)
      return jsonResponse({ data: held, count: held.length })
    }
    if (url.includes(`${API_ROOT}/organizations/me/budgets`)) {
      if (method === "DELETE") return jsonResponse({ message: "deleted" })
      // And each budget carries its entities, named where they have a name.
      const withEntities = budgets.map((row) => ({
        ...row,
        applied_to: ceilings
          .filter((ceiling) => ceiling.budget_id === row.budget_id)
          .map((ceiling) => ({
            scope_type: ceiling.scope_type,
            scope_id: ceiling.scope_id,
            provider_key_id: ceiling.provider_key_id,
            model: ceiling.model,
            current_spend: ceiling.current_spend,
            reserved_spend: ceiling.reserved_spend,
            current_tokens: ceiling.current_tokens,
            reserved_tokens: ceiling.reserved_tokens,
            current_requests: ceiling.current_requests,
            reserved_requests: ceiling.reserved_requests,
            name:
              ceiling.scope_type === "organization"
                ? organization().name
                : ceiling.scope_type === "workspace" &&
                    ceiling.scope_id === workspace().id
                  ? "Platform"
                  : null,
          })),
      }))
      return jsonResponse({ data: withEntities, count: budgets.length })
    }
    if (url.includes(`${API_ROOT}/organizations/me/members`))
      return jsonResponse({ data: [], count: 0 })
    if (url.includes(`${API_ROOT}/workspaces`)) {
      return jsonResponse({ data: [workspace({ name: "Platform" })], count: 1 })
    }
    if (url.endsWith(`${API_ROOT}/models`))
      return jsonResponse({ object: "list", data: [] })
    return jsonResponse([])
  })
  return requests
}

function renderDetail(budgetId = BUDGET_ID) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <OrganizationBudgetDetail
          organization={organizationContext({
            role: "admin",
            deployment_operator: false,
          })}
          budgetId={budgetId}
        />
      </QueryClientProvider>
    </DeploymentProvider>,
    {
      wrapper: withRouter({
        url: `/budgets/${budgetId}`,
        routes: [{ path: "/budgets", element: <span>the budget list</span> }],
      }),
    },
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("OrganizationBudgetDetail", () => {
  it("shows each entity's own spend against the limit, not a sum", async () => {
    mockApi()
    renderDetail()

    expect(
      await screen.findByRole("heading", { name: "Engineering monthly" }),
    ).toBeInTheDocument()
    const table = await screen.findByRole("grid", { name: "Spend by entity" })
    const rows = await within(table).findAllByRole("row")
    // A header and this budget's two entities; the other budget's is left out.
    expect(rows).toHaveLength(3)
    expect(within(table).getByText("Whole organization")).toBeInTheDocument()
    expect(within(table).getByText("Platform")).toBeInTheDocument()
    expect(
      within(table).getByText("$12.50 (+$2.00 held) of $250.00"),
    ).toBeInTheDocument()
    expect(within(table).getByText("$240.00 of $250.00")).toBeInTheDocument()
    expect(
      within(table).queryByRole("columnheader", { name: "Tokens" }),
    ).toBeNull()
  })

  it("adds a column for an axis only when the budget caps it", async () => {
    mockApi({ budgets: [budget({ token_limit: 1000 })] })
    renderDetail()

    const table = await screen.findByRole("grid", { name: "Spend by entity" })
    expect(
      within(table).getByRole("columnheader", { name: "Tokens" }),
    ).toBeInTheDocument()
    expect(
      within(table).queryByRole("columnheader", { name: "Requests" }),
    ).toBeNull()
  })

  it("says a budget applies to nothing rather than showing an empty table", async () => {
    mockApi({ budgets: [budget({ ceiling_count: 0 })], ceilings: [] })
    renderDetail()

    expect(await screen.findByText("Not applied yet")).toBeInTheDocument()
  })

  it("says so when the budget is not this organization's", async () => {
    mockApi()
    renderDetail("not-a-budget")

    expect(await screen.findByText("Budget not found")).toBeInTheDocument()
  })

  it("says what stops being capped before deleting, then returns to the list", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    renderDetail()

    await user.click(await screen.findByRole("button", { name: "Delete" }))
    const confirm = await screen.findByRole("alertdialog")
    expect(
      within(confirm).getByText(
        /Default Organization \(organization\), Platform stop being capped by it/,
      ),
    ).toBeInTheDocument()
    await user.click(
      within(confirm).getByRole("button", { name: "Delete budget" }),
    )

    expect(await screen.findByText("the budget list")).toBeInTheDocument()
    await waitFor(() =>
      expect(
        requests.some(
          (request) =>
            request.method === "DELETE" && request.url.includes(BUDGET_ID),
        ),
      ).toBe(true),
    )
  })
})

describe("BudgetDetailPage", () => {
  it("sends a deployment operator to the budget list, which has no detail route", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes(`${API_ROOT}/organizations/me`)) {
        return jsonResponse(organizationContext({ deployment_operator: true }))
      }
      return jsonResponse({ data: [], count: 0 })
    })
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    render(
      <DeploymentProvider value={bootstrap()}>
        <QueryClientProvider client={client}>
          <BudgetDetailPage budgetId={BUDGET_ID} />
        </QueryClientProvider>
      </DeploymentProvider>,
      {
        wrapper: withRouter({
          url: `/budgets/${BUDGET_ID}`,
          routes: [{ path: "/budgets", element: <span>the budget list</span> }],
        }),
      },
    )

    expect(await screen.findByText("the budget list")).toBeInTheDocument()
  })
})
