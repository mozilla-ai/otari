import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { WorkspaceBudgetDefault } from "@/client"
import { WorkspaceDefaultLimit } from "@/features/spend/WorkspaceDefaultLimit"
import { useAllSpendCeilings } from "@/shared/api/budgets"
import { API_ROOT } from "@/shared/api/client"
import { useOrganizationMembers } from "@/shared/api/organizations"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { organizationContext, workspaceBudgetDefault } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const WORKSPACE = "w-platform"

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

/**
 * An owner in Platform. `defaults` is the workspace's per-person defaults:
 * a list, `"failed"` for a 500, or `"pending"` for a read that never answers.
 * `failDefaultCreate` refuses the write that points the default at a budget;
 * `budgetInUse` refuses a budget delete as the server does while ceilings
 * still hold it. A listed default follows its budget's amount, as the server
 * reads it through the budget, and a default made here is listed after. Every
 * write is in `writes`; `requests` is every request, reads too, in order.
 */
function mockApi({
  defaults,
  failDefaultCreate = false,
  budgetInUse = false,
}: {
  defaults: WorkspaceBudgetDefault[] | "failed" | "pending"
  failDefaultCreate?: boolean
  budgetInUse?: boolean
}) {
  const writes: string[] = []
  const requests: string[] = []
  let listed = Array.isArray(defaults) ? [...defaults] : defaults
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    const path = url.replace(API_ROOT, "").split("?")[0]
    requests.push(`${method} ${path}`)
    if (method !== "GET") {
      writes.push(`${method} ${path}`)
      const body = init?.body ? JSON.parse(String(init.body)) : {}
      if (path === "/organizations/me/budgets") {
        return json({ budget_id: "budget-new" }, 201)
      }
      if (
        method === "PATCH" &&
        path.startsWith("/organizations/me/budgets/") &&
        Array.isArray(listed)
      ) {
        const budgetId = path.split("/").pop()
        listed = listed.map((one) =>
          one.budget_id === budgetId
            ? { ...one, max_budget: body.max_budget }
            : one,
        )
      }
      if (
        method === "DELETE" &&
        path.startsWith("/organizations/me/budgets/") &&
        budgetInUse
      ) {
        return json({ detail: "The budget still holds 3 ceilings" }, 409)
      }
      if (path.endsWith("/member-budget-policies")) {
        if (failDefaultCreate) {
          return json({ detail: "Defaults are unavailable" }, 500)
        }
        if (method === "POST" && Array.isArray(listed)) {
          listed = [
            ...listed,
            workspaceBudgetDefault({
              workspace_id: WORKSPACE,
              budget_id: body.budget_id,
              max_budget: 25,
              reset_alignment: "calendar_month",
              budget_duration_sec: null,
            }),
          ]
        }
      }
      return json({ message: "ok" })
    }
    if (path === "/organizations/me") {
      return json(
        organizationContext({
          deployment_operator: false,
          workspace_memberships: [
            { workspace_id: WORKSPACE, name: "Platform", role: "owner" },
          ],
        }),
      )
    }
    if (path.endsWith("/member-budget-policies")) {
      if (listed === "pending") return new Promise<Response>(() => {})
      if (listed === "failed") {
        return json({ detail: "Defaults are unavailable" }, 500)
      }
      return json({ data: listed, count: listed.length })
    }
    if (
      path === "/organizations/me/spend-ceilings" ||
      path === "/organizations/me/members"
    ) {
      return json({ data: [], count: 0 })
    }
    return json([])
  })
  return { writes, requests }
}

/** The people rows' read beside the line, as Home's People cut holds it. */
function CeilingsReader() {
  useAllSpendCeilings()
  return null
}

/** The people list beside the line, which carries each person's limit. */
function MembersReader() {
  useOrganizationMembers()
  return null
}

async function renderLimit() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  await renderWithRouter(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <WorkspaceDefaultLimit />
        <CeilingsReader />
        <MembersReader />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
  )
  return userEvent.setup()
}

const weekly = workspaceBudgetDefault({
  workspace_id: WORKSPACE,
  max_budget: 40,
  reset_alignment: "calendar_week",
  budget_duration_sec: null,
})

describe("WorkspaceDefaultLimit", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("reads a $0 default as a limit of $0, not as none", async () => {
    mockApi({
      defaults: [
        workspaceBudgetDefault({
          workspace_id: WORKSPACE,
          max_budget: 0,
          reset_alignment: "calendar_month",
          budget_duration_sec: null,
        }),
      ],
    })
    await renderLimit()
    expect(
      await screen.findByText("may spend up to $0.00 a month each."),
    ).toBeVisible()
  })

  it("reports a defaults list it could not read, and offers nothing to set", async () => {
    mockApi({ defaults: "failed" })
    await renderLimit()
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Defaults are unavailable",
    )
    expect(
      screen.queryByRole("button", { name: "Set spend limit for everyone" }),
    ).toBeNull()
  })

  it("offers nothing to set while the list is still being read", async () => {
    mockApi({ defaults: "pending" })
    await renderLimit()
    expect(await screen.findByText("…")).toBeVisible()
    expect(
      screen.queryByRole("button", { name: "Set spend limit for everyone" }),
    ).toBeNull()
  })

  it("deletes the budget it made when the default cannot point at it", async () => {
    const { writes } = mockApi({ defaults: [], failDefaultCreate: true })
    const user = await renderLimit()
    await user.click(
      await screen.findByRole("button", {
        name: "Set spend limit for everyone",
      }),
    )
    const dialog = await screen.findByRole("dialog")
    await user.type(
      within(dialog).getByLabelText("Monthly spend limit, in dollars"),
      "25",
    )
    await user.click(within(dialog).getByRole("button", { name: "Save limit" }))

    await waitFor(() =>
      expect(writes).toEqual([
        "POST /organizations/me/budgets",
        `POST /workspaces/${WORKSPACE}/member-budget-policies`,
        "DELETE /organizations/me/budgets/budget-new",
      ]),
    )
    expect(await within(dialog).findByRole("alert")).toHaveTextContent(
      "Defaults are unavailable",
    )
  })

  it("keeps a budget the people it reached still use, and reports no error", async () => {
    const { writes } = mockApi({ defaults: [weekly], budgetInUse: true })
    const user = await renderLimit()
    await screen.findByText("may spend up to $40.00 a week each.")
    await user.click(screen.getByRole("button", { name: "Row actions" }))
    await user.click(
      await screen.findByRole("menuitem", {
        name: /^Remove the spend limit for everyone in Platform/,
      }),
    )
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Remove",
      }),
    )
    await waitFor(() =>
      expect(writes).toEqual([
        `DELETE /workspaces/${WORKSPACE}/member-budget-policies/${weekly.id}`,
        `DELETE /organizations/me/budgets/${weekly.budget_id}`,
      ]),
    )
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("opens an existing default on its own period, not on a month", async () => {
    mockApi({ defaults: [weekly] })
    const user = await renderLimit()
    expect(
      await screen.findByText("may spend up to $40.00 a week each."),
    ).toBeVisible()
    // A row's actions fold into one menu at the test's viewport width.
    await user.click(screen.getByRole("button", { name: "Row actions" }))
    await user.click(
      await screen.findByRole("menuitem", { name: "Change spend limit…" }),
    )
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).queryByLabelText("Monthly spend limit, in dollars"),
    ).toBeNull()
    expect(
      within(dialog).getByLabelText("Spend limit, in dollars"),
    ).toHaveValue("40")
    expect(
      within(dialog).getByText(
        "A request that would take a person past the limit is refused until their week resets, and the refusal shows in Sessions.",
      ),
    ).toBeVisible()
  })

  it("shows the new amount on the line and in the dialog once it changes", async () => {
    mockApi({ defaults: [weekly] })
    const user = await renderLimit()
    await screen.findByText("may spend up to $40.00 a week each.")
    await user.click(screen.getByRole("button", { name: "Row actions" }))
    await user.click(
      await screen.findByRole("menuitem", { name: "Change spend limit…" }),
    )
    const dialog = await screen.findByRole("dialog")
    const amount = within(dialog).getByLabelText("Spend limit, in dollars")
    await user.clear(amount)
    await user.type(amount, "55")
    await user.click(within(dialog).getByRole("button", { name: "Save limit" }))

    expect(
      await screen.findByText("may spend up to $55.00 a week each."),
    ).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Row actions" }))
    await user.click(
      await screen.findByRole("menuitem", { name: "Change spend limit…" }),
    )
    expect(
      within(await screen.findByRole("dialog")).getByLabelText(
        "Spend limit, in dollars",
      ),
    ).toHaveValue("55")
  })

  it("reads the limits a new default gives everyone once the default exists", async () => {
    const { requests } = mockApi({ defaults: [] })
    const user = await renderLimit()
    await user.click(
      await screen.findByRole("button", {
        name: "Set spend limit for everyone",
      }),
    )
    const dialog = await screen.findByRole("dialog")
    await user.type(
      within(dialog).getByLabelText("Monthly spend limit, in dollars"),
      "25",
    )
    await user.click(within(dialog).getByRole("button", { name: "Save limit" }))

    expect(
      await screen.findByText("may spend up to $25.00 a month each."),
    ).toBeVisible()
    // The server caps everyone already in the workspace when the default is
    // made, so the people rows' ceilings are read again after that write, not
    // only after the budget's.
    const made = requests.indexOf(
      `POST /workspaces/${WORKSPACE}/member-budget-policies`,
    )
    expect(made).toBeGreaterThan(-1)
    await waitFor(() =>
      expect(
        requests.lastIndexOf("GET /organizations/me/spend-ceilings"),
      ).toBeGreaterThan(made),
    )
  })

  it("reads the people list again once a default's amount changes", async () => {
    const { requests } = mockApi({ defaults: [weekly] })
    const user = await renderLimit()
    await screen.findByText("may spend up to $40.00 a week each.")
    await user.click(screen.getByRole("button", { name: "Row actions" }))
    await user.click(
      await screen.findByRole("menuitem", { name: "Change spend limit…" }),
    )
    const dialog = await screen.findByRole("dialog")
    const amount = within(dialog).getByLabelText("Spend limit, in dollars")
    await user.clear(amount)
    await user.type(amount, "100")
    await user.click(within(dialog).getByRole("button", { name: "Save limit" }))

    expect(
      await screen.findByText("may spend up to $100.00 a week each."),
    ).toBeVisible()
    // Each person's limit in the list is read through the default's budget.
    const changed = requests.indexOf(
      `PATCH /organizations/me/budgets/${weekly.budget_id}`,
    )
    expect(changed).toBeGreaterThan(-1)
    await waitFor(() =>
      expect(
        requests.lastIndexOf("GET /organizations/me/members"),
      ).toBeGreaterThan(changed),
    )
  })
})
