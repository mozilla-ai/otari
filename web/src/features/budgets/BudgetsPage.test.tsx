import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  Budget,
  BudgetResetLog,
  OrganizationContext,
  User,
} from "@/client"
import { BudgetsPage } from "@/features/budgets/BudgetsPage"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"

function testUser(user_id: string): User {
  return {
    user_id,
    alias: null,
    spend: 0,
    reserved: 0,
    current_tokens: 0,
    reserved_tokens: 0,
    current_requests: 0,
    reserved_requests: 0,
    budget_id: null,
    allowed_models: null,
    budget_started_at: null,
    next_budget_reset_at: null,
    blocked: false,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    metadata: {},
  }
}

function budget(overrides: Partial<Budget> = {}): Budget {
  return {
    budget_id: "11111111-2222-3333-4444-555555555555",
    organization_id: null,
    name: null,
    max_budget: 100,
    token_limit: null,
    request_limit: null,
    reset_alignment: null,
    budget_duration_sec: 86_400,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    user_count: 0,
    total_spend: 0,
    total_reserved: 0,
    ...overrides,
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(
  opts: {
    budgets?: Budget[]
    resetLogs?: BudgetResetLog[]
    users?: User[]
    failedUserUpdates?: string[]
    updateUser?: (userId: string) => Response | Promise<Response>
    // Who is asking, which is what decides which of the two pages this route
    // renders. An operator by default, so every case below describes the
    // deployment-wide page it always described.
    context?: OrganizationContext
  } = {},
) {
  let list = [...(opts.budgets ?? [])]
  const resetLogs = opts.resetLogs ?? []
  const users = opts.users ?? []
  const context = opts.context ?? organizationContext()

  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()

      if (url.includes("/v1/users")) {
        if (method === "PATCH") {
          const userId = decodeURIComponent(url.split("/").pop() ?? "")
          if (opts.failedUserUpdates?.includes(userId)) {
            return jsonResponse({ detail: "User update failed" }, 500)
          }
          if (opts.updateUser) {
            return opts.updateUser(userId)
          }
          return jsonResponse(testUser(userId))
        }
        return jsonResponse(users)
      }

      if (url.includes("/v1/budgets")) {
        if (url.includes("/reset-logs")) {
          return jsonResponse(resetLogs)
        }
        if (method === "POST") {
          const body = JSON.parse(String(init?.body)) as Partial<Budget>
          const row = budget({
            budget_id: "new-budget-id-0000-0000-000000000000",
            name: body.name ?? null,
            max_budget: body.max_budget ?? null,
            budget_duration_sec: body.budget_duration_sec ?? null,
          })
          list = [...list, row]
          return jsonResponse(row)
        }
        if (method === "PATCH") {
          const id = decodeURIComponent(url.split("/").pop() ?? "")
          const body = JSON.parse(String(init?.body)) as Partial<Budget>
          list = list.map((b) => (b.budget_id === id ? { ...b, ...body } : b))
          return jsonResponse(list.find((b) => b.budget_id === id))
        }
        if (method === "DELETE") {
          const id = decodeURIComponent(url.split("/").pop() ?? "")
          list = list.filter((b) => b.budget_id !== id)
          return new Response(null, { status: 204 })
        }
        return jsonResponse(list)
      }
      if (url.includes("/v1/organizations/me/budgets")) {
        return jsonResponse({ data: [], count: 0 })
      }
      if (url.includes("/v1/organizations/me/spend-ceilings")) {
        return jsonResponse({ data: [], count: 0 })
      }
      if (url.includes("/v1/organizations/me")) return jsonResponse(context)
      return jsonResponse([])
    })
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  // The assignment picker asks the organization roster what to call each person,
  // and that read is gated on the `organizations` surface, so the page needs the
  // deployment context the shell always gives it.
  return render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>{ui}</QueryClientProvider>
    </DeploymentProvider>,
  )
}

describe("BudgetsPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows onboarding when there are no budgets", async () => {
    mockApi({ budgets: [] })
    renderPage(<BudgetsPage />)

    expect(await screen.findByText("No budgets yet")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Create your first budget" }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Create budget" }),
    ).not.toBeInTheDocument()
    // Only the onboarding panel shows: the table (and its own "no rows" fallback,
    // whose "cap spending" text is unique to it) is suppressed so the two empty
    // states are not stacked.
    expect(screen.queryByText(/cap spending/)).not.toBeInTheDocument()
    expect(
      screen.queryByRole("grid", { name: "Budgets" }),
    ).not.toBeInTheDocument()
  })

  it("copies the full budget id, of which the table shows only a prefix", async () => {
    mockApi({
      budgets: [budget({ budget_id: "11111111-2222-3333-4444-555555555555" })],
    })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    const row = (await screen.findByText("11111111")).closest("tr")!
    await user.click(
      within(row).getByRole("button", { name: "Copy budget id" }),
    )

    expect(await navigator.clipboard.readText()).toBe(
      "11111111-2222-3333-4444-555555555555",
    )
  })

  it("lists a budget with its limit and humanized reset period", async () => {
    mockApi({
      budgets: [budget({ max_budget: 100, budget_duration_sec: 604_800 })],
    })
    renderPage(<BudgetsPage />)

    const row = (await screen.findByText("11111111")).closest("tr")!
    expect(within(row).getByText("$100.00")).toBeInTheDocument()
    expect(within(row).getByText("Weekly")).toBeInTheDocument()
  })

  it("renders an unlimited budget without a spend bar", async () => {
    mockApi({
      budgets: [
        budget({ max_budget: null, budget_duration_sec: null, user_count: 0 }),
      ],
    })
    renderPage(<BudgetsPage />)

    const row = (await screen.findByText("11111111")).closest("tr")!
    // One vocabulary across both budget pages, where this table used to say
    // "Unlimited" and the organization one "No limit" for the same state.
    expect(within(row).getByText("No limit")).toBeInTheDocument()
    expect(within(row).getByText("No reset")).toBeInTheDocument()
    expect(within(row).getByText("No users assigned")).toBeInTheDocument()
  })

  it("names a token cap instead of calling the budget unlimited", async () => {
    // A budget capping only tokens refuses requests, so the page it is
    // inspected from must not read as though nothing binds it.
    mockApi({
      budgets: [
        budget({
          max_budget: null,
          token_limit: 1_000_000,
          budget_duration_sec: null,
          user_count: 0,
        }),
      ],
    })
    renderPage(<BudgetsPage />)

    const row = (await screen.findByText("11111111")).closest("tr")!
    expect(within(row).getByText("1,000,000 tokens")).toBeInTheDocument()
    expect(within(row).queryByText("No limit")).not.toBeInTheDocument()
  })

  it("shows an aggregate spend bar across assigned users", async () => {
    mockApi({
      budgets: [budget({ max_budget: 100, user_count: 3, total_spend: 150 })],
    })
    renderPage(<BudgetsPage />)

    const row = (await screen.findByText("11111111")).closest("tr")!
    // 3 users × $100 = $300 allocated; $150 spent.
    expect(within(row).getByText("$150.00")).toBeInTheDocument()
    expect(within(row).getByText("of $300.00")).toBeInTheDocument()
    const bar = within(row).getByRole("progressbar")
    expect(bar).toHaveAttribute("aria-valuenow", "50")
  })

  it("creates a budget, posting the limit and chosen period", async () => {
    const fetchMock = mockApi({ budgets: [] })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    await screen.findByText("No budgets yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first budget" }),
    )
    await user.type(screen.getByLabelText("Name (optional)"), "team-free-tier")
    await user.type(screen.getByLabelText("Spending limit (USD)"), "250")
    await user.click(screen.getByRole("radio", { name: "Weekly" }))
    await user.click(screen.getByRole("button", { name: "Create budget" }))

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes("/v1/budgets") && (init?.method ?? "") === "POST",
    )
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({
      name: "team-free-tier",
      max_budget: 250,
      budget_duration_sec: 604_800,
    })

    // The created budget shows its name in the table.
    expect(await screen.findByText("team-free-tier")).toBeInTheDocument()
  })

  it("assigns the new budget to chosen users on create", async () => {
    const fetchMock = mockApi({
      budgets: [],
      users: [testUser("alice"), testUser("bob")],
    })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    await screen.findByText("No budgets yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first budget" }),
    )
    await user.type(screen.getByLabelText("Spending limit (USD)"), "100")
    // Pick a user from the assignment combobox, then submit.
    await user.type(screen.getByLabelText("Add a person"), "alice")
    await user.click(await screen.findByRole("option", { name: /alice/ }))
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Create budget" }))

    // The chosen user is PATCHed onto the newly created budget's id.
    const patch = await vi.waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([u, init]) =>
          String(u).includes("/v1/users/alice") &&
          (init?.method ?? "") === "PATCH",
      )
      if (!call) throw new Error("no PATCH yet")
      return call
    })
    expect(JSON.parse(String(patch[1]?.body))).toEqual({
      budget_id: "new-budget-id-0000-0000-000000000000",
    })
  })

  it("keeps failed initial assignments retryable without creating another budget", async () => {
    const fetchMock = mockApi({
      budgets: [],
      users: [testUser("alice")],
      failedUserUpdates: ["alice"],
    })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    await user.click(
      await screen.findByRole("button", { name: "Create your first budget" }),
    )
    await user.type(screen.getByLabelText("Add a person"), "alice")
    await user.click(await screen.findByRole("option", { name: /alice/ }))
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Create budget" }))

    expect(
      await screen.findByText(/these people were not updated: alice/),
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Retry assignments" }))
    await vi.waitFor(() => {
      const patches = fetchMock.mock.calls.filter(
        ([url, init]) =>
          String(url).includes("/v1/users/alice") &&
          (init?.method ?? "") === "PATCH",
      )
      expect(patches).toHaveLength(2)
    })

    const budgetPosts = fetchMock.mock.calls.filter(
      ([url, init]) =>
        String(url).includes("/v1/budgets") && (init?.method ?? "") === "POST",
    )
    expect(budgetPosts).toHaveLength(1)
  })

  it("prevents closing the form while initial user assignments are pending", async () => {
    let resolveUserUpdate: ((response: Response) => void) | undefined
    const userUpdate = new Promise<Response>((resolve) => {
      resolveUserUpdate = resolve
    })
    mockApi({
      budgets: [],
      users: [testUser("alice")],
      updateUser: () => userUpdate,
    })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    await user.click(
      await screen.findByRole("button", { name: "Create your first budget" }),
    )
    await user.type(screen.getByLabelText("Add a person"), "alice")
    await user.click(await screen.findByRole("option", { name: /alice/ }))
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Create budget" }))

    await vi.waitFor(() =>
      expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled(),
    )
    await act(async () => {
      resolveUserUpdate?.(jsonResponse(testUser("alice")))
      await userUpdate
    })
    await vi.waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Cancel" }),
      ).not.toBeInTheDocument(),
    )
  })

  it("does not submit a non-finite budget limit", async () => {
    const fetchMock = mockApi({ budgets: [] })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    await user.click(
      await screen.findByRole("button", { name: "Create your first budget" }),
    )
    await user.type(screen.getByLabelText("Spending limit (USD)"), "1e309")

    expect(screen.getByRole("button", { name: "Create budget" })).toBeDisabled()
    expect(
      fetchMock.mock.calls.some(
        ([url, init]) =>
          String(url).includes("/v1/budgets") &&
          (init?.method ?? "") === "POST",
      ),
    ).toBe(false)
  })

  it("blocks a custom period below one whole day instead of rounding it to zero", async () => {
    const fetchMock = mockApi({ budgets: [] })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    await user.click(
      await screen.findByRole("button", { name: "Create your first budget" }),
    )
    await user.click(screen.getByRole("radio", { name: "Custom" }))
    await user.click(screen.getByLabelText("Every N days"))
    await user.paste("0.1")

    expect(
      await screen.findByText("Enter a whole number of days."),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create budget" })).toBeDisabled()
    expect(
      fetchMock.mock.calls.some(
        ([u, init]) =>
          String(u).includes("/v1/budgets") && (init?.method ?? "") === "POST",
      ),
    ).toBe(false)
  })

  it("rejects a fractional custom period instead of rounding it up", async () => {
    const fetchMock = mockApi({ budgets: [] })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    await user.click(
      await screen.findByRole("button", { name: "Create your first budget" }),
    )
    await user.click(screen.getByRole("radio", { name: "Custom" }))
    await user.click(screen.getByLabelText("Every N days"))
    await user.paste("1.5")

    // 1.5 is flagged and blocks submit, never silently rounded to 2 days.
    expect(
      await screen.findByText("Enter a whole number of days."),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create budget" })).toBeDisabled()
    expect(
      fetchMock.mock.calls.some(
        ([u, init]) =>
          String(u).includes("/v1/budgets") && (init?.method ?? "") === "POST",
      ),
    ).toBe(false)
  })

  it("keeps a fractional edit visible with an error and blocks save (does not wipe the field)", async () => {
    // 14 days is a custom period (not a preset), so Edit opens with the field
    // seeded to "14"; making it fractional exercises the committed-value path the
    // paste-into-empty tests miss.
    mockApi({ budgets: [budget({ budget_duration_sec: 1_209_600 })] })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    const row = (await screen.findByText("11111111")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    const field = await screen.findByLabelText("Every N days")
    expect(field).toHaveValue("14")

    await user.type(field, ".5")

    // The invalid entry persists with an error instead of being wiped, and Save is
    // blocked so it cannot clear the committed period to "no reset".
    expect(field).toHaveValue("14.5")
    expect(
      screen.getByText("Enter a whole number of days."),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled()
  })

  it("creates an unlimited budget when the limit is left blank", async () => {
    const fetchMock = mockApi({ budgets: [] })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    await screen.findByText("No budgets yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first budget" }),
    )
    // Leave the limit blank; keep "No reset" (the default selection).
    await user.click(screen.getByRole("button", { name: "Create budget" }))

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes("/v1/budgets") && (init?.method ?? "") === "POST",
    )
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({
      name: null,
      max_budget: null,
      budget_duration_sec: null,
    })
  })

  it("opens the edit form seeded from the row's Edit action", async () => {
    mockApi({
      budgets: [budget({ max_budget: 42, budget_duration_sec: 86_400 })],
    })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    const row = (await screen.findByText("11111111")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    expect(
      await screen.findByRole("button", { name: "Save changes" }),
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Spending limit (USD)")).toHaveValue("42")
  })

  it("marks a budget an organization owns and withholds assignment on it", async () => {
    // `/v1/users` refuses to cap a gateway user at a tenant's budget, so offering
    // the multiselect would be offering a save that answers 404.
    mockApi({
      budgets: [
        budget({ organization_id: "99999999-8888-7777-6666-555555555555" }),
      ],
      users: [testUser("alice")],
    })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    const row = (await screen.findByText("11111111")).closest("tr")!
    expect(
      within(row).getByText("Owned by an organization"),
    ).toBeInTheDocument()

    await user.click(within(row).getByRole("button", { name: "Edit" }))
    expect(
      await screen.findByRole("button", { name: "Save changes" }),
    ).toBeInTheDocument()
    expect(screen.queryByText("Assign to people (optional)")).toBeNull()
    expect(screen.getByText(/belongs to an organization/)).toBeInTheDocument()
  })

  it("keeps the assignment control on the deployment's own budget", async () => {
    mockApi({ budgets: [budget()], users: [testUser("alice")] })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    const row = (await screen.findByText("11111111")).closest("tr")!
    expect(within(row).queryByText("Owned by an organization")).toBeNull()

    await user.click(within(row).getByRole("button", { name: "Edit" }))
    expect(
      await screen.findByText("Assign to people (optional)"),
    ).toBeInTheDocument()
  })

  it("reveals per-user reset history on demand", async () => {
    mockApi({
      budgets: [budget()],
      resetLogs: [
        {
          id: 1,
          user_id: "alice",
          budget_id: "11111111-2222-3333-4444-555555555555",
          previous_spend: 12.5,
          reset_at: "2026-02-01T00:00:00+00:00",
          next_reset_at: "2026-02-02T00:00:00+00:00",
        },
      ],
    })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    const row = (await screen.findByText("11111111")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "History" }))

    expect(await screen.findByText("alice")).toBeInTheDocument()
    expect(screen.getByText("$12.50")).toBeInTheDocument()
  })

  it("deletes a budget after an explicit confirm", async () => {
    const fetchMock = mockApi({ budgets: [budget()] })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    const row = (await screen.findByText("11111111")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))
    expect(within(row).getByText(/lose this limit/)).toBeInTheDocument()
    await user.click(
      within(row).getByRole("button", { name: "Delete permanently" }),
    )

    const del = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes("/v1/budgets/") && (init?.method ?? "") === "DELETE",
    )
    expect(del).toBeDefined()
    expect(screen.queryByText("11111111")).not.toBeInTheDocument()
  })

  it("bulk-deletes the selected budgets after a confirm", async () => {
    const fetchMock = mockApi({
      budgets: [
        budget({ budget_id: "b1", name: "Team monthly" }),
        budget({ budget_id: "b2", name: "Trial cap" }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<BudgetsPage />)

    const row = (await screen.findByText("Team monthly")).closest("tr")!
    await user.click(within(row).getByRole("checkbox"))

    const bar = (await screen.findByText("1 selected")).closest("div")!
    await user.click(within(bar).getByRole("button", { name: "Delete" }))

    const dialog = await screen.findByRole("alertdialog")
    await user.click(within(dialog).getByRole("button", { name: "Delete" }))

    await vi.waitFor(() => {
      const del = fetchMock.mock.calls.find(
        ([u, init]) =>
          String(u).includes("/v1/budgets/b1") &&
          (init?.method ?? "").toUpperCase() === "DELETE",
      )
      expect(del).toBeTruthy()
    })
  })

  it("gives an organization admin their own budgets, not the deployment's", async () => {
    // One route, two pages (otari-ai#1943). Every deployment-wide read this
    // file's other cases make answers 403 to a tenant, so an admin gets the
    // organization-scoped page instead of a screen of refusals.
    const requests = mockApi({
      budgets: [budget()],
      // An admin, not the owner the fixture defaults to: the matrix row is
      // about the admin, and it is the weaker of the two management roles.
      context: organizationContext({
        role: "admin",
        deployment_operator: false,
      }),
    })
    renderPage(<BudgetsPage />)

    expect(
      await screen.findByRole("grid", { name: "Organization budgets" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("grid", { name: "Organization spend ceilings" }),
    ).toBeInTheDocument()

    // Withheld at the request, not only in the markup.
    const read = requests.mock.calls.map(([url]) => String(url))
    expect(read.some((url) => /\/v1\/budgets/.test(url))).toBe(false)
    expect(read.some((url) => /\/v1\/scoped-budgets/.test(url))).toBe(false)
    expect(read.some((url) => /\/v1\/users/.test(url))).toBe(false)
  })

  it("keeps the deployment page for an operator", async () => {
    // The other side of the split, pinned so a future change cannot quietly
    // take the deployment's budgets away from the caller who owns them.
    mockApi({ budgets: [budget({ name: "Deployment wide" })] })
    renderPage(<BudgetsPage />)

    expect(await screen.findByText("Deployment wide")).toBeInTheDocument()
    expect(
      screen.queryByRole("grid", { name: "Organization budgets" }),
    ).toBeNull()
  })
})
