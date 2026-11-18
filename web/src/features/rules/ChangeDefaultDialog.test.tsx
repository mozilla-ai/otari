import { Outlet } from "@tanstack/react-router"
import { screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ChangeDefaultDialog } from "@/features/rules/ChangeDefaultDialog"
import type { TeamDefault } from "@/features/rules/rules"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

const WEEKLY: TeamDefault = {
  workspaceId: "ws-data",
  team: "Data",
  policyId: "p",
  budgetId: "d",
  max: 25,
  period: { reset_alignment: "calendar_week", budget_duration_sec: null },
}

async function renderDialog(teamDefault: TeamDefault) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
    Response.json({ data: [], count: 0 }),
  )
  await renderWithRouter(
    <ChangeDefaultDialog teamDefault={teamDefault} onClose={() => {}} />,
    {
      url: "/rules",
      shell: (
        <Provider>
          <DeploymentProvider value={bootstrap()}>
            <Outlet />
          </DeploymentProvider>
        </Provider>
      ),
    },
  )
}

describe("Change the limit for everyone in a workspace", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("says the period the workspace's limit already runs on", async () => {
    await renderDialog(WEEKLY)
    const field = await screen.findByRole("textbox", {
      name: "Spend limit, in dollars",
    })
    expect(field).toHaveValue("25")
    expect(
      screen.getByText(
        "A request that would take a person past the limit is refused until their week resets, and the refusal shows in Sessions.",
      ),
    ).toBeVisible()
  })

  it("opens a $0 limit on $0, not empty", async () => {
    await renderDialog({ ...WEEKLY, max: 0 })
    expect(
      await screen.findByRole("textbox", { name: "Spend limit, in dollars" }),
    ).toHaveValue("0")
  })

  it("calls a monthly limit monthly", async () => {
    await renderDialog({
      ...WEEKLY,
      period: { reset_alignment: "calendar_month", budget_duration_sec: null },
    })
    expect(
      await screen.findByRole("textbox", {
        name: "Monthly spend limit, in dollars",
      }),
    ).toBeVisible()
  })
})
