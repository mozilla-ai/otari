import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationBudget, OrganizationSpendCeiling } from "@/client"
import { OrganizationBudgetsPage } from "@/features/budgets/OrganizationBudgetsPage"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext, workspace } from "@/tests/fixtures"

interface RecordedRequest {
  url: string
  method: string
  body: unknown
}

function organizationBudget(
  overrides: Partial<OrganizationBudget> = {},
): OrganizationBudget {
  return {
    budget_id: "bbbbbbbb-1111-2222-3333-444444444444",
    organization_id: "11111111-1111-1111-1111-111111111111",
    name: "Engineering monthly",
    max_budget: 250,
    budget_duration_sec: null,
    reset_alignment: "calendar_month",
    ceiling_count: 0,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    ...overrides,
  }
}

function spendCeiling(
  overrides: Partial<OrganizationSpendCeiling> = {},
): OrganizationSpendCeiling {
  return {
    id: "cccccccc-1111-2222-3333-444444444444",
    scope_type: "organization",
    scope_id: "11111111-1111-1111-1111-111111111111",
    provider_key_id: null,
    budget_id: "bbbbbbbb-1111-2222-3333-444444444444",
    name: null,
    max_budget: 250,
    current_spend: 12.5,
    reserved_spend: 0,
    budget_duration_sec: null,
    reset_alignment: "calendar_month",
    period_start: "2026-08-01T00:00:00+00:00",
    period_end: "2026-09-01T00:00:00+00:00",
    manageable: true,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    ...overrides,
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

// Mocked at the `@/client` boundary (a real `fetch`), per the standards: the
// hooks and their invalidation are part of what is under test.
function mockApi({
  budgets = [organizationBudget()],
  ceilings = [] as OrganizationSpendCeiling[],
  writeStatus = 201,
}: {
  budgets?: OrganizationBudget[]
  ceilings?: OrganizationSpendCeiling[]
  writeStatus?: number
} = {}) {
  const requests: RecordedRequest[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
    if (url.includes("/v1/organizations/me/spend-ceilings")) {
      if (method === "GET") {
        return jsonResponse({ data: ceilings, count: ceilings.length })
      }
      if (method === "DELETE") return jsonResponse({ message: "deleted" })
      return jsonResponse(spendCeiling(), writeStatus)
    }
    if (url.includes("/v1/organizations/me/budgets")) {
      if (method === "GET") {
        return jsonResponse({ data: budgets, count: budgets.length })
      }
      if (method === "DELETE") return jsonResponse({ message: "deleted" })
      return jsonResponse(organizationBudget(), writeStatus)
    }
    if (url.includes("/v1/workspaces")) {
      return jsonResponse({
        data: [workspace({ name: "Engineering" })],
        count: 1,
      })
    }
    return jsonResponse([])
  })
  return requests
}

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap()}>
      <QueryClientProvider client={client}>
        <OrganizationBudgetsPage
          // The caller this page exists for: an admin who does not operate the
          // deployment. The fixture defaults to an owner who does, which is the
          // one caller that would reach the other page instead.
          organization={organizationContext({
            role: "admin",
            deployment_operator: false,
          })}
        />
      </QueryClientProvider>
    </DeploymentProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("OrganizationBudgetsPage", () => {
  it("reads only the organization's own surfaces, never the deployment's", async () => {
    // The point of the page. `/v1/budgets` and `/v1/scoped-budgets` answer 403
    // to a tenant, so touching either would paint a refusal on a page that is
    // the admin's to use.
    const requests = mockApi()
    renderPage()
    await screen.findByRole("grid", { name: "Organization budgets" })

    const read = requests.map((request) => request.url)
    expect(
      read.some((url) => url.includes("/v1/organizations/me/budgets")),
    ).toBe(true)
    expect(
      read.some((url) => url.includes("/v1/organizations/me/spend-ceilings")),
    ).toBe(true)
    for (const url of read) {
      expect(url).not.toMatch(/\/v1\/budgets/)
      expect(url).not.toMatch(/\/v1\/scoped-budgets/)
    }
  })

  it("lists a budget with its limit, period and how many ceilings hold it", async () => {
    mockApi({ budgets: [organizationBudget({ ceiling_count: 2 })] })
    renderPage()

    const table = await screen.findByRole("grid", {
      name: "Organization budgets",
    })
    // Awaited: `DataTable` renders the grid with a loading row, so the grid
    // exists a beat before its rows do.
    expect(
      await within(table).findByText("Engineering monthly"),
    ).toBeInTheDocument()
    expect(within(table).getByText(/250/)).toBeInTheDocument()
    expect(within(table).getByText(/1st at 00:00 UTC/)).toBeInTheDocument()
    expect(within(table).getByText("2 ceilings")).toBeInTheDocument()
  })

  it("shows no spend column on a budget, because that figure is not the tenant's", async () => {
    // The deployment page sums `users.spend`, which is deployment-wide and has
    // no tenancy column, so the same column here would be a cross-tenant read.
    mockApi()
    renderPage()

    const table = await screen.findByRole("grid", {
      name: "Organization budgets",
    })
    expect(
      within(table).queryByRole("columnheader", { name: /spent/i }),
    ).toBeNull()
  })

  it("creates a budget as a calendar period rather than a duration", async () => {
    // A duration is measured from the last reset, so "Monthly" as 30 days is a
    // 1.5 percent more generous product than the calendar month an admin means.
    const requests = mockApi({ budgets: [] })
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole("grid", { name: "Organization budgets" })

    await user.click(screen.getByRole("button", { name: "Add budget" }))
    await user.type(screen.getByLabelText("Name"), "Design")
    await user.type(screen.getByLabelText("Limit (USD)"), "75")
    const submit = screen.getAllByRole("button", { name: "Add budget" }).at(-1)
    await user.click(submit as HTMLElement)

    await waitFor(() =>
      expect(
        requests.some(
          (request) =>
            request.method === "POST" &&
            request.url.includes("/v1/organizations/me/budgets"),
        ),
      ).toBe(true),
    )
    const posted = requests.find(
      (request) =>
        request.method === "POST" &&
        request.url.includes("/v1/organizations/me/budgets"),
    )
    expect(posted?.body).toMatchObject({
      name: "Design",
      max_budget: 75,
      reset_alignment: "calendar_month",
      budget_duration_sec: null,
    })
  })

  it("refuses a limit that is not an amount rather than sending it", async () => {
    mockApi({ budgets: [] })
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole("grid", { name: "Organization budgets" })

    await user.click(screen.getByRole("button", { name: "Add budget" }))
    await user.type(screen.getByLabelText("Limit (USD)"), "-5")

    const submit = screen.getAllByRole("button", { name: "Add budget" }).at(-1)
    expect(submit).toBeDisabled()
  })

  it("says a blank limit means no limit rather than zero", async () => {
    mockApi({ budgets: [organizationBudget({ max_budget: null })] })
    renderPage()

    const table = await screen.findByRole("grid", {
      name: "Organization budgets",
    })
    expect(await within(table).findByText("No limit")).toBeInTheDocument()
  })

  it("warns that deleting a held budget will be refused, before trying", async () => {
    mockApi({ budgets: [organizationBudget({ ceiling_count: 3 })] })
    const user = userEvent.setup()
    renderPage()
    const table = await screen.findByRole("grid", {
      name: "Organization budgets",
    })

    await user.click(
      await within(table).findByRole("button", { name: "Delete" }),
    )

    expect(
      await screen.findByText(
        /held by 3 spend ceilings, so this will be refused/,
      ),
    ).toBeInTheDocument()
  })

  it("lists a ceiling with what it caps and what it has spent", async () => {
    mockApi({
      ceilings: [
        spendCeiling({
          scope_type: "workspace",
          scope_id: workspace().id,
          current_spend: 12.5,
          reserved_spend: 2,
        }),
      ],
    })
    renderPage()

    const table = await screen.findByRole("grid", {
      name: "Organization spend ceilings",
    })
    expect(
      await within(table).findByText("Engineering (workspace)"),
    ).toBeInTheDocument()
    expect(within(table).getByText("Every provider")).toBeInTheDocument()
    // Reserved counts towards the cap and is not spend yet, so both are shown:
    // the ceiling refuses on their sum.
    expect(within(table).getByText(/held/)).toBeInTheDocument()
  })

  it("marks a ceiling whose budget is set outside the organization", async () => {
    // What the otari-ai cutover writes. Listed rather than hidden, because it is
    // enforcing today and omitting it would let the page read as uncapped.
    mockApi({ ceilings: [spendCeiling({ manageable: false })] })
    renderPage()

    const table = await screen.findByRole("grid", {
      name: "Organization spend ceilings",
    })
    expect(
      await within(table).findByText("Set at the deployment level"),
    ).toBeInTheDocument()
  })

  it("never names the deployment operator to a tenant", async () => {
    // An operator is an internal role a tenant can neither see nor change, so
    // no copy on this page may explain a limit by naming one.
    mockApi({ ceilings: [spendCeiling({ manageable: false })] })
    const { container } = renderPage()
    // Awaited past the loading rows, so the assertion reads the real copy rather
    // than a table that has not rendered its marker yet.
    await screen.findByText("Set at the deployment level")

    expect(container.textContent).not.toMatch(/operator/i)
    expect(container.textContent).not.toMatch(/superuser/i)
  })

  it("creates a ceiling against the whole organization by default", async () => {
    const requests = mockApi()
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole("grid", { name: "Organization spend ceilings" })

    await user.click(screen.getByRole("button", { name: "Add ceiling" }))
    const submit = screen.getAllByRole("button", { name: "Add ceiling" }).at(-1)
    await user.click(submit as HTMLElement)

    await waitFor(() =>
      expect(
        requests.some(
          (request) =>
            request.method === "POST" &&
            request.url.includes("/v1/organizations/me/spend-ceilings"),
        ),
      ).toBe(true),
    )
    const posted = requests.find(
      (request) =>
        request.method === "POST" &&
        request.url.includes("/v1/organizations/me/spend-ceilings"),
    )
    // The scope an admin reaches this page to set, held to the organization's
    // own first budget.
    expect(posted?.body).toMatchObject({
      scope_type: "organization",
      budget_id: organizationBudget().budget_id,
    })
  })

  it("will not offer a ceiling with no budget to hold", async () => {
    mockApi({ budgets: [] })
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole("grid", { name: "Organization spend ceilings" })

    await user.click(screen.getByRole("button", { name: "Add ceiling" }))

    expect(await screen.findByText(/Add a budget first/)).toBeInTheDocument()
  })

  it("will not save a ceiling still holding a budget the organization does not own", async () => {
    // `FilterSelect` carries an unmatched value as its own option rather than
    // dropping it, so the deployment budget stays selected and Save looked
    // enabled while submitting an id the endpoint answers 404 for.
    mockApi({
      ceilings: [
        spendCeiling({
          manageable: false,
          // A budget id that is not among the organization's own, which is what
          // `manageable: false` means on the wire.
          budget_id: "dddddddd-9999-9999-9999-999999999999",
        }),
      ],
    })
    const user = userEvent.setup()
    renderPage()
    const table = await screen.findByRole("grid", {
      name: "Organization spend ceilings",
    })

    await user.click(await within(table).findByRole("button", { name: "Edit" }))

    expect(
      await screen.findByText(/Choose one of your own to take it over/),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save ceiling" })).toBeDisabled()
  })

  it("sends only the label and the budget when editing a ceiling", async () => {
    // The endpoint ignores the scope on a PATCH, because changing it would move
    // the ceiling to another identity while carrying its spend.
    const requests = mockApi({ ceilings: [spendCeiling()] })
    const user = userEvent.setup()
    renderPage()
    const table = await screen.findByRole("grid", {
      name: "Organization spend ceilings",
    })

    await user.click(await within(table).findByRole("button", { name: "Edit" }))
    await user.type(screen.getByLabelText("Name"), "Whole org")
    await user.click(screen.getByRole("button", { name: "Save ceiling" }))

    await waitFor(() =>
      expect(requests.some((request) => request.method === "PATCH")).toBe(true),
    )
    const patched = requests.find((request) => request.method === "PATCH")
    expect(Object.keys(patched?.body as object).sort()).toEqual([
      "budget_id",
      "name",
    ])
  })

  it("names a failed workspace roster instead of just offering no workspaces", async () => {
    // Without this the owner sees the consequence (no workspace to pick, rows
    // reading "A workspace") and never the cause.
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes("/v1/workspaces")) {
        return jsonResponse({ detail: "workspaces unavailable" }, 500)
      }
      if (url.includes("/v1/organizations/me/spend-ceilings")) {
        return jsonResponse({ data: [], count: 0 })
      }
      if (url.includes("/v1/organizations/me/budgets")) {
        return jsonResponse({ data: [organizationBudget()], count: 1 })
      }
      return jsonResponse([])
    })
    renderPage()

    expect((await screen.findAllByRole("alert")).length).toBeGreaterThan(0)
  })

  it("reports a failed read rather than an empty organization", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      jsonResponse({ detail: "nope" }, 500),
    )
    renderPage()

    // An empty table after a failed read says "nothing is capped", which is the
    // opposite of what a 500 means.
    expect((await screen.findAllByRole("alert")).length).toBeGreaterThan(0)
  })
})
