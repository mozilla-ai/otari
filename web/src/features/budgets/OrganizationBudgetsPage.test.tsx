import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  ApiKey,
  OrganizationBudget,
  OrganizationContext,
  OrganizationMember,
  OrganizationSpendCeiling,
} from "@/client"
import { OrganizationBudgetsPage } from "@/features/budgets/OrganizationBudgetsPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  apiKey,
  bootstrap,
  organization,
  organizationContext,
  organizationMember,
  organizationSpendCeiling as spendCeiling,
  workspace,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

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
    token_limit: null,
    request_limit: null,
    reset_cycle: "monthly",
    reset_every_n: null,
    reset_anchor_at: null,
    reset_weekdays: null,
    reset_month_day: 1,
    reset_month: null,
    ceiling_count: 0,
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
function catalogModel(id: string) {
  return {
    id,
    object: "model",
    created: 0,
    owned_by: id.split(":")[0],
    pricing_source: "none",
  }
}

function mockApi({
  budgets = [organizationBudget()],
  ceilings = [] as OrganizationSpendCeiling[],
  writeStatus = 201,
  budgetsGate,
  ceilingsGate,
  models = [] as string[],
  members = [] as OrganizationMember[],
  keys = [] as ApiKey[],
  workspacesStatus = 200,
}: {
  budgets?: OrganizationBudget[]
  ceilings?: OrganizationSpendCeiling[]
  writeStatus?: number
  /** What GET /v1/models serves, which is where the provider picker looks. */
  models?: string[]
  // Holds the budget list in flight, so a dialog can be opened before it
  // lands: that is when a default arriving after mount is observable.
  budgetsGate?: Promise<unknown>
  /** Holds the ceilings in flight, so an edit can be opened before they land. */
  ceilingsGate?: Promise<unknown>
  members?: OrganizationMember[]
  keys?: ApiKey[]
  workspacesStatus?: number
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
    if (url.includes(`${API_ROOT}/organizations/me/spend-ceilings`)) {
      if (method === "GET") {
        if (ceilingsGate) await ceilingsGate
        // Honors the window, like the endpoint: a test that ignored it could
        // not tell a paged read from a read of everything.
        const params = new URL(url, "http://localhost").searchParams
        const skip = Number(params.get("skip") ?? 0)
        const limit = Number(params.get("limit") ?? 100)
        return jsonResponse({
          data: ceilings.slice(skip, skip + limit),
          count: ceilings.length,
        })
      }
      if (method === "DELETE") return jsonResponse({ message: "deleted" })
      return jsonResponse(spendCeiling(), writeStatus)
    }
    if (url.includes(`${API_ROOT}/organizations/me/budgets`)) {
      if (method === "GET") {
        if (budgetsGate) await budgetsGate
        return jsonResponse({ data: budgets, count: budgets.length })
      }
      if (method === "DELETE") return jsonResponse({ message: "deleted" })
      return jsonResponse(organizationBudget(), writeStatus)
    }
    if (url.endsWith(`${API_ROOT}/models`)) {
      return jsonResponse({ object: "list", data: models.map(catalogModel) })
    }
    if (url.includes(`${API_ROOT}/organizations/me/members`)) {
      return jsonResponse({ data: members, count: members.length })
    }
    if (url.includes(`${API_ROOT}/organizations/me/keys`)) {
      return jsonResponse(keys)
    }
    if (url.includes(`${API_ROOT}/workspaces`)) {
      if (workspacesStatus !== 200) {
        return jsonResponse({ detail: "unavailable" }, workspacesStatus)
      }
      return jsonResponse({
        data: [workspace({ name: "Engineering" })],
        count: 1,
      })
    }
    return jsonResponse([])
  })
  return requests
}

// The caller this page exists for: an admin who does not operate the
// deployment. The fixture defaults to an owner who does, which is the one
// caller that would reach the other page instead.
const admin = (overrides: Partial<OrganizationContext> = {}) =>
  organizationContext({
    role: "admin",
    deployment_operator: false,
    ...overrides,
  })

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  // Switching organization invalidates every query rather than remounting this
  // page, so the page seeing a new context in place is what a switch looks like
  // from here. Held in state rather than passed by `rerender`, because the
  // router renders the page and does not re-render it for new wrapper props.
  let switchContext: (context: OrganizationContext) => void = () => {}
  function Harness() {
    const [context, setContext] = useState(admin())
    switchContext = setContext
    return (
      <DeploymentProvider value={bootstrap()}>
        <QueryClientProvider client={client}>
          <OrganizationBudgetsPage organization={context} />
        </QueryClientProvider>
      </DeploymentProvider>
    )
  }
  const result = render(<Harness />, {
    wrapper: withRouter({ url: "/budgets" }),
  })
  return {
    ...result,
    switchTo: (context: OrganizationContext) =>
      act(() => switchContext(context)),
  }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("OrganizationBudgetsPage", () => {
  it("reads only the organization's own surfaces, never the deployment's", async () => {
    // The point of the page. /api/v1/budgets and /api/v1/scoped-budgets answer 403
    // to a tenant, so touching either would paint a refusal on a page that is
    // the admin's to use.
    const requests = mockApi()
    renderPage()
    await screen.findByRole("grid", { name: "Budgets" })

    const read = requests.map((request) => request.url)
    expect(
      read.some((url) => url.includes(`${API_ROOT}/organizations/me/budgets`)),
    ).toBe(true)
    expect(
      read.some((url) =>
        url.includes(`${API_ROOT}/organizations/me/spend-ceilings`),
      ),
    ).toBe(true)
    for (const url of read) {
      expect(url).not.toMatch(/\/api\/v1\/budgets/)
      expect(url).not.toMatch(/\/api\/v1\/scoped-budgets/)
    }
  })

  it("lists a budget with its limit, cycle and what it applies to", async () => {
    mockApi({
      budgets: [organizationBudget({ ceiling_count: 3 })],
      ceilings: [
        spendCeiling({ id: "c-1", scope_type: "org_member", scope_id: "m-1" }),
        spendCeiling({ id: "c-2", scope_type: "org_member", scope_id: "m-2" }),
        spendCeiling({ id: "c-3", scope_type: "api_token", scope_id: "k-1" }),
      ],
    })
    renderPage()

    const table = await screen.findByRole("grid", {
      name: "Budgets",
    })
    // Awaited: `DataTable` renders the grid with a loading row, so the grid
    // exists a beat before its rows do.
    expect(
      await within(table).findByText("Engineering monthly"),
    ).toBeInTheDocument()
    expect(within(table).getByText(/250/)).toBeInTheDocument()
    expect(within(table).getByText(/1st at 00:00 UTC/)).toBeInTheDocument()
    expect(
      await within(table).findByText("2 organization members, 1 API key"),
    ).toBeInTheDocument()
  })

  it("says how many a budget applies to until the entities themselves land", async () => {
    // The count travels with the budget and the entities are a second read, so
    // the cell has an answer before that read lands. "Not applied yet" is the
    // one it must not be.
    mockApi({ budgets: [organizationBudget({ ceiling_count: 4 })] })
    renderPage()

    const table = await screen.findByRole("grid", { name: "Budgets" })
    expect(await within(table).findByText("4 entities")).toBeInTheDocument()
  })

  it("says a budget applies to nothing when nothing holds it", async () => {
    mockApi({ budgets: [organizationBudget({ ceiling_count: 0 })] })
    renderPage()

    const table = await screen.findByRole("grid", { name: "Budgets" })
    expect(
      await within(table).findByText("Not applied yet"),
    ).toBeInTheDocument()
  })

  it("shows no spend column on a budget, because that figure is not the tenant's", async () => {
    // The deployment page sums `users.spend`, which is deployment-wide and has
    // no tenancy column, so the same column here would be a cross-tenant read.
    mockApi()
    renderPage()

    const table = await screen.findByRole("grid", {
      name: "Budgets",
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
    await screen.findByText("No budgets created")

    await user.click(screen.getAllByRole("button", { name: "New Budget" })[0])
    await user.type(screen.getByLabelText("Name"), "Design")
    await user.type(screen.getByLabelText("Limit (USD)"), "75")
    const submit = screen
      .getAllByRole("button", { name: "Create budget" })
      .at(-1)
    await user.click(submit as HTMLElement)

    await waitFor(() =>
      expect(
        requests.some(
          (request) =>
            request.method === "POST" &&
            request.url.includes(`${API_ROOT}/organizations/me/budgets`),
        ),
      ).toBe(true),
    )
    const posted = requests.find(
      (request) =>
        request.method === "POST" &&
        request.url.includes(`${API_ROOT}/organizations/me/budgets`),
    )
    expect(posted?.body).toMatchObject({
      name: "Design",
      max_budget: 75,
      reset_cycle: "monthly",
      reset_month_day: 1,
    })
  })

  it("says what an unnamed budget will be called, before it is saved", async () => {
    // The name is optional, and a budget saved without one is handed out under
    // what it caps (#2130) rather than under the head of its id.
    mockApi({ budgets: [] })
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("No budgets created")

    await user.click(screen.getAllByRole("button", { name: "New Budget" })[0])
    await user.type(screen.getByLabelText("Limit (USD)"), "75")
    expect(
      screen.getByText(/Left blank, this budget is called "\$75.00 \/ month"/),
    ).toBeInTheDocument()
  })

  it("refuses a limit that is not an amount rather than sending it", async () => {
    mockApi({ budgets: [] })
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("No budgets created")

    await user.click(screen.getAllByRole("button", { name: "New Budget" })[0])
    await user.type(screen.getByLabelText("Limit (USD)"), "-5")

    const submit = screen
      .getAllByRole("button", { name: "Create budget" })
      .at(-1)
    expect(submit).toBeDisabled()
  })

  it("says a blank limit means no limit rather than zero", async () => {
    mockApi({ budgets: [organizationBudget({ max_budget: null })] })
    renderPage()

    const table = await screen.findByRole("grid", {
      name: "Budgets",
    })
    expect(await within(table).findByText("No limit")).toBeInTheDocument()
  })

  it("does not greet the next budget open with the last attempt's refusal", async () => {
    // The create and update mutations live inside the dialog, below the card's
    // key, so the remount that clears the draft clears the refusal too. Held in
    // the card they outlived it, and a failed edit of one row was what the next
    // row's dialog showed.
    mockApi({ writeStatus: 409 })
    const user = userEvent.setup()
    renderPage()

    await user.click(
      (await screen.findAllByRole("button", { name: "New Budget" }))[0],
    )
    await user.type(await screen.findByLabelText(/^Name/), "team-a")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create budget",
      }),
    )
    expect(await screen.findByRole("alert")).toBeInTheDocument()

    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    await user.click(screen.getAllByRole("button", { name: "New Budget" })[0])

    const reopened = await screen.findByRole("dialog")
    expect(within(reopened).queryByRole("alert")).toBeNull()
    expect(within(reopened).getByLabelText(/^Name/)).toHaveValue("")
  })

  it("says what stops being capped before deleting a budget that applies to something", async () => {
    const requests = mockApi({
      budgets: [organizationBudget({ ceiling_count: 1 })],
      ceilings: [
        spendCeiling({ scope_type: "workspace", scope_id: workspace().id }),
      ],
    })
    const user = userEvent.setup()
    renderPage()
    const table = await screen.findByRole("grid", { name: "Budgets" })
    await within(table).findByText("Engineering")

    await user.click(
      await within(table).findByRole("button", {
        name: "Delete Engineering monthly",
      }),
    )
    const confirm = await screen.findByRole("alertdialog")
    expect(
      within(confirm).getByText(/Engineering stops being capped by it/),
    ).toBeInTheDocument()
    await user.click(
      within(confirm).getByRole("button", { name: "Delete budget" }),
    )
    await waitFor(() =>
      expect(requests.some((request) => request.method === "DELETE")).toBe(
        true,
      ),
    )
  })

  it("links each budget to its own page", async () => {
    mockApi()
    renderPage()
    const table = await screen.findByRole("grid", { name: "Budgets" })
    expect(
      await within(table).findByRole("link", { name: "Engineering monthly" }),
    ).toHaveAttribute("href", "/budgets/bbbbbbbb-1111-2222-3333-444444444444")
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

  // ===========================================================================
  // Where a budget applies: the picker inside the budget form
  // ===========================================================================

  /** The budget form, once its picker has rendered. */
  async function openForm(name: "New Budget" | "Edit Engineering monthly") {
    const user = userEvent.setup()
    if (name === "New Budget") {
      await user.click(
        (await screen.findAllByRole("button", { name: "New Budget" }))[0],
      )
    } else {
      const table = await screen.findByRole("grid", { name: "Budgets" })
      await user.click(await within(table).findByRole("button", { name }))
    }
    const dialog = await screen.findByRole("dialog")
    await within(dialog).findByRole("group", { name: "Applied to" })
    return { user, dialog }
  }

  async function pick(
    user: ReturnType<typeof userEvent.setup>,
    dialog: HTMLElement,
    field: string,
    option: RegExp | string,
  ) {
    await user.click(within(dialog).getByRole("combobox", { name: field }))
    await user.click(await screen.findByRole("option", { name: option }))
    await user.keyboard("{Escape}")
  }

  const written = (requests: RecordedRequest[], method: string) =>
    requests.find(
      (request) =>
        request.method === method &&
        request.url.includes(`${API_ROOT}/organizations/me/budgets`),
    )?.body as { applied_to?: unknown[] } | undefined

  it("creates a budget and where it applies in one write", async () => {
    const requests = mockApi({ budgets: [] })
    renderPage()
    const { user, dialog } = await openForm("New Budget")

    await user.click(
      within(dialog).getByRole("checkbox", { name: /whole organization/ }),
    )
    await pick(user, dialog, "Workspaces", "Engineering")
    await user.click(
      within(dialog).getByRole("button", { name: "Create budget" }),
    )

    await waitFor(() => expect(written(requests, "POST")).toBeDefined())
    expect(written(requests, "POST")?.applied_to).toEqual([
      {
        scope_type: "organization",
        scope_id: organization().id,
        provider_key_id: null,
        model: null,
      },
      {
        scope_type: "workspace",
        scope_id: workspace().id,
        provider_key_id: null,
        model: null,
      },
    ])
    // No second write: the ceilings are the budget's, not a follow-up.
    expect(
      requests.some(
        (request) =>
          request.url.includes("spend-ceilings") && request.method !== "GET",
      ),
    ).toBe(false)
  })

  it("applies a budget to a provider or a model picked from the ones served here", async () => {
    // Read off the catalog because /v1/providers is operator-only and this page
    // is the one an admin who is not an operator lands on.
    const requests = mockApi({ budgets: [], models: ["openai-eu:gpt-4o"] })
    renderPage()
    const { user, dialog } = await openForm("New Budget")

    await pick(user, dialog, "Providers", "openai-eu")
    await pick(user, dialog, "Models", "openai-eu:gpt-4o")
    await user.click(
      within(dialog).getByRole("button", { name: "Create budget" }),
    )

    await waitFor(() =>
      expect(written(requests, "POST")?.applied_to).toEqual([
        {
          scope_type: "organization",
          scope_id: organization().id,
          provider_key_id: "openai-eu",
          model: null,
        },
        {
          scope_type: "organization",
          scope_id: organization().id,
          provider_key_id: "openai-eu",
          model: "gpt-4o",
        },
      ]),
    )
  })

  it("offers members by their membership and keys by their id", async () => {
    const requests = mockApi({
      budgets: [],
      members: [
        organizationMember({
          full_name: "Pat Okafor",
          workspaces: [
            {
              role: "member",
              workspace_id: workspace().id,
              workspace_member_id: "wm-1",
              workspace_name: "Engineering",
            },
          ],
        }),
      ],
      keys: [apiKey({ key_name: "ci-bot", workspace_id: workspace().id })],
    })
    renderPage()
    const { user, dialog } = await openForm("New Budget")

    await pick(user, dialog, "Organization members", /Pat Okafor/)
    await pick(user, dialog, "Workspace members", /Pat Okafor/)
    await pick(user, dialog, "API keys", /ci-bot/)
    await user.click(
      within(dialog).getByRole("button", { name: "Create budget" }),
    )

    await waitFor(() => expect(written(requests, "POST")).toBeDefined())
    expect(
      (
        (written(requests, "POST")?.applied_to ?? []) as {
          scope_type: string
          scope_id: string
        }[]
      ).map((entity) => [entity.scope_type, entity.scope_id]),
    ).toEqual([
      ["org_member", organizationMember().organization_member_id],
      ["workspace_member", "wm-1"],
      ["api_token", "key-1"],
    ])
  })

  it("edits the whole set, keeping an entity the picker does not offer", async () => {
    // A workspace narrowed to a provider is a real ceiling no group expresses.
    // Sending the set without it would remove it, so it is listed and kept.
    const narrowed = spendCeiling({
      id: "c-narrowed",
      scope_type: "workspace",
      scope_id: workspace().id,
      provider_key_id: "openai",
    })
    const plain = spendCeiling({
      id: "c-plain",
      scope_type: "workspace",
      scope_id: workspace().id,
    })
    const requests = mockApi({
      budgets: [organizationBudget({ ceiling_count: 2 })],
      ceilings: [narrowed, plain],
    })
    renderPage()
    const { user, dialog } = await openForm("Edit Engineering monthly")

    const others = within(dialog).getByRole("list", { name: "Also applied to" })
    expect(
      within(others).getByText("Engineering, on openai"),
    ).toBeInTheDocument()
    await user.click(
      within(dialog).getByRole("button", { name: "Remove Engineering" }),
    )
    await user.click(
      within(dialog).getByRole("button", { name: "Save budget" }),
    )

    await waitFor(() =>
      expect(written(requests, "PATCH")?.applied_to).toEqual([
        {
          scope_type: "workspace",
          scope_id: workspace().id,
          provider_key_id: "openai",
          model: null,
        },
      ]),
    )
  })

  it("will not save an edit before it has read where the budget applies", async () => {
    // The edit sends the whole set, so a save before the set lands would remove
    // every entity the form had not read yet.
    let release: () => void = () => {}
    const requests = mockApi({
      budgets: [organizationBudget({ ceiling_count: 1 })],
      ceilingsGate: new Promise<void>((resolve) => {
        release = resolve
      }),
    })
    const user = userEvent.setup()
    renderPage()
    const table = await screen.findByRole("grid", { name: "Budgets" })
    await user.click(
      await within(table).findByRole("button", {
        name: "Edit Engineering monthly",
      }),
    )
    const dialog = await screen.findByRole("dialog")

    expect(
      within(dialog).getByText("Reading where this budget applies…"),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: "Save budget" }),
    ).toBeDisabled()

    release()
    await within(dialog).findByRole("group", { name: "Applied to" })
    expect(
      within(dialog).getByRole("button", { name: "Save budget" }),
    ).toBeEnabled()
    expect(requests.some((request) => request.method === "PATCH")).toBe(false)
  })

  it("lists an entity another budget carries, but will not pick it", async () => {
    mockApi({
      budgets: [
        organizationBudget({
          budget_id: "b-research",
          name: "Research",
          ceiling_count: 1,
        }),
      ],
      ceilings: [
        spendCeiling({
          budget_id: "b-research",
          scope_type: "workspace",
          scope_id: workspace().id,
        }),
      ],
    })
    renderPage()
    const { user, dialog } = await openForm("New Budget")

    await user.click(
      within(dialog).getByRole("combobox", { name: "Workspaces" }),
    )
    const option = await screen.findByRole("option", { name: /Engineering/ })
    expect(option).toHaveAttribute("aria-disabled", "true")
    expect(option).toHaveAccessibleName("Engineering (On Research)")
  })

  it("never names the deployment operator to a tenant", async () => {
    // A ceiling on a budget this organization does not own is one the
    // deployment set. "Deployment level" is the fact; who operates it is not
    // the tenant's to see.
    mockApi({
      budgets: [],
      ceilings: [
        spendCeiling({ budget_id: "not-ours", scope_type: "organization" }),
      ],
    })
    renderPage()
    const { dialog } = await openForm("New Budget")

    expect(
      within(dialog).getByRole("checkbox", { name: /whole organization/ }),
    ).toBeDisabled()
    expect(
      within(dialog).getByText("On a budget set at the deployment level"),
    ).toBeInTheDocument()
    expect(within(dialog).queryByText(/operator/i)).toBeNull()
  })

  it("drops an open edit when the organization changes under it", async () => {
    // `editing` holds a row, and a switch leaves this page mounted. Without the
    // reset the form stays open on a budget from the organization just left, and
    // saving PATCHes an id the new organization does not own.
    const requests = mockApi()
    const { switchTo } = renderPage()
    await openForm("Edit Engineering monthly")

    switchTo(
      admin({
        organization: organization({
          id: "77777777-7777-7777-7777-777777777777",
          name: "Second Organization",
        }),
      }),
    )

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    expect(requests.some((request) => request.method === "PATCH")).toBe(false)
  })

  it("names a list that failed instead of just offering nothing", async () => {
    mockApi({ budgets: [], workspacesStatus: 500 })
    renderPage()
    const { dialog } = await openForm("New Budget")

    expect(
      await within(dialog).findByText(/Could not load workspaces/),
    ).toBeInTheDocument()
  })

  it("seeds each open fresh, whichever budget was opened last", async () => {
    mockApi({
      budgets: [organizationBudget({ ceiling_count: 1 })],
      ceilings: [
        spendCeiling({ scope_type: "workspace", scope_id: workspace().id }),
      ],
    })
    renderPage()
    const { user, dialog } = await openForm("Edit Engineering monthly")
    expect(
      within(dialog).getByRole("list", { name: "Workspaces, selected" }),
    ).toBeInTheDocument()

    await user.keyboard("{Escape}")
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    const next = await openForm("New Budget")
    expect(
      within(next.dialog).queryByRole("list", { name: "Workspaces, selected" }),
    ).toBeNull()
  })
})
