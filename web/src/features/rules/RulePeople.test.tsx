import { Outlet } from "@tanstack/react-router"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { RulePeople } from "@/features/rules/RulePeople"
import type { Rule, RulePerson } from "@/features/rules/rules"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

const PEOPLE: RulePerson[] = Array.from({ length: 12 }, (_, index) => ({
  name: `Person ${String(index + 1).padStart(2, "0")}`,
  userId: `user-${index}`,
  memberId: `om-${index}`,
  team: "Platform",
  spent: 12 - index,
  isYou: false,
}))

const RULE: Rule = {
  id: "spend-each-b",
  name: "Platform per person",
  type: "spend",
  scope: { kind: "each", teams: ["Platform"], count: 12, includesYou: false },
  mode: "Enforce",
  subject: {},
  spend: { max: 40, per: "a week", isEach: true },
  people: PEOPLE,
}

describe("People under a rule", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("moves focus to the first person Show all adds", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json({ data: [], count: 0 }),
    )
    const user = userEvent.setup()
    await renderWithRouter(<RulePeople rule={RULE} isManager />, {
      url: "/rules/spend-each-b",
      shell: (
        <Provider>
          <DeploymentProvider value={bootstrap()}>
            <Outlet />
          </DeploymentProvider>
        </Provider>
      ),
    })
    expect(screen.queryByText("Person 11")).toBeNull()
    await user.click(await screen.findByRole("button", { name: "Show all 12" }))
    // Most spent first, so the eleventh is the first one added.
    expect(await screen.findByText("Person 11")).toBeVisible()
    await waitFor(() =>
      expect(document.activeElement).toHaveAttribute("data-key", "om-10"),
    )
  })

  it("reads a person in two workspaces against each one's own ceiling", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json({ data: [], count: 0 }),
    )
    const ethan: RulePerson = {
      name: "Ethan Brooks",
      userId: "user-ethan",
      memberId: "om-ethan",
      team: "Data, Platform",
      spent: 12,
      peak: { spent: 6, max: 10 },
      isYou: false,
    }
    await renderWithRouter(
      <RulePeople
        rule={{
          ...RULE,
          spend: { max: 10, per: "a week", isEach: true },
          people: [ethan],
        }}
        isManager
      />,
      {
        url: "/rules/spend-each-b",
        shell: (
          <Provider>
            <DeploymentProvider value={bootstrap()}>
              <Outlet />
            </DeploymentProvider>
          </Provider>
        ),
      },
    )
    expect(await screen.findByText("60%")).toBeVisible()
    // Spent is the placement the bar is drawn from, not the $12 across both.
    expect(screen.getByText("$6.00")).toBeVisible()
    expect(screen.queryByText("$12.00")).toBeNull()
    expect(screen.queryByText("120%")).toBeNull()
  })
})
