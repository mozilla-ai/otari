import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { WorkspaceBudgetDefaultsPage } from "@/features/workspaces/WorkspaceBudgetDefaultsPage"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { organizationContext, workspaceBudgetDefault } from "@/tests/fixtures"

const ALPHA = "11111111-1111-1111-1111-111111111111"

function mockApi({
  memberships = [{ workspace_id: ALPHA, name: "Alpha", role: "admin" }],
  defaults = [workspaceBudgetDefault()],
}: {
  memberships?: { workspace_id: string; name: string; role: string }[]
  defaults?: ReturnType<typeof workspaceBudgetDefault>[]
} = {}) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url.includes("/member-budget-policies")) {
      return Response.json({ data: defaults, count: defaults.length })
    }
    if (url.includes("/v1/provider-credentials")) {
      return Response.json([])
    }
    return Response.json(
      organizationContext({ workspace_memberships: memberships }),
    )
  })
}

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <WorkspaceBudgetDefaultsPage />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
  )
}

describe("WorkspaceBudgetDefaultsPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("shows the budget defaults of the selected workspace, named by the organization", async () => {
    mockApi()
    renderPage()

    expect(
      await screen.findByText("Budget defaults for Alpha"),
    ).toBeInTheDocument()
    expect(await screen.findByText("Default member budget")).toBeInTheDocument()
  })

  it("says so rather than showing an empty list when there is no workspace", async () => {
    mockApi({ memberships: [] })
    renderPage()

    expect(await screen.findByText("No workspace selected")).toBeInTheDocument()
  })

  it("offers management controls to a workspace owner even without an organization role", async () => {
    // "member" at the organization level, but "owner" of the selected
    // workspace itself: the server's OR rule
    // (`require_workspace_management_access`), which the page has to match.
    mockApi({
      memberships: [{ workspace_id: ALPHA, name: "Alpha", role: "owner" }],
    })
    renderPage()

    await screen.findByText("Budget defaults for Alpha")
    expect(
      screen.getByRole("button", { name: "Add default" }),
    ).toBeInTheDocument()
  })

  it("preserves a reset period that isn't a whole number of days when the field is left untouched", async () => {
    // 3,600 seconds (one hour) cannot be shown in a whole-days field; editing
    // an unrelated part of the form and saving must not read that display gap
    // as "clear the reset period".
    const fetchMock = mockApi({
      defaults: [
        workspaceBudgetDefault({
          id: "default-1",
          name: "Hourly-reset default",
          budget_duration_sec: 3_600,
        }),
      ],
    })
    const user = userEvent.setup()
    renderPage()

    const row = (await screen.findByText("Hourly-reset default")).closest("li")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    const daysField = await screen.findByLabelText(
      "Reset every N days (optional)",
    )
    expect(daysField).toHaveValue("")
    expect(screen.getByText(/currently set to 3600s/i)).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Save changes" }))

    const patch = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes("/default-1") && (init?.method ?? "") === "PATCH",
    )
    expect(JSON.parse(String(patch?.[1]?.body))).toMatchObject({
      budget_duration_sec: 3_600,
    })
  })

  it("clears the reset period when the days field is explicitly emptied", async () => {
    const fetchMock = mockApi({
      defaults: [
        workspaceBudgetDefault({
          id: "default-1",
          name: "Daily-reset default",
          budget_duration_sec: 86_400,
        }),
      ],
    })
    const user = userEvent.setup()
    renderPage()

    const row = (await screen.findByText("Daily-reset default")).closest("li")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    const daysField = await screen.findByLabelText(
      "Reset every N days (optional)",
    )
    expect(daysField).toHaveValue("1")
    await user.clear(daysField)
    await user.click(screen.getByRole("button", { name: "Save changes" }))

    const patch = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes("/default-1") && (init?.method ?? "") === "PATCH",
    )
    expect(JSON.parse(String(patch?.[1]?.body))).toMatchObject({
      budget_duration_sec: null,
    })
  })
})
