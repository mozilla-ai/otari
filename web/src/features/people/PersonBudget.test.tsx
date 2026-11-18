import { Outlet } from "@tanstack/react-router"
import { screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { PersonBudget } from "@/features/people/PersonBudget"
import { ALL_READ, type PersonRow } from "@/features/people/people"
import { organizationMember, organizationSpendCeiling } from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

const WEI = organizationMember({
  organization_member_id: "om-wei",
  user_id: "login-wei",
  attribution_user_id: "user-wei",
  full_name: "Wei Zhang",
  role: "member",
})

const PERSON: PersonRow = {
  id: "om-wei",
  member: WEI,
  userId: "user-wei",
  name: "Wei Zhang",
  email: "",
  teams: ["Platform"],
  role: "member",
  isInvited: false,
  spend: 0,
  tokens: 0,
  spendRow: undefined,
  sessions: 0,
  refusals: 0,
  apiKeys: undefined,
  isOutsideView: false,
  unread: ALL_READ,
}

/** Wei's own limit, at `max_budget` with nothing spent yet. */
async function renderBudget(maxBudget: number) {
  const ceiling = organizationSpendCeiling({
    scope_type: "org_member",
    scope_id: "om-wei",
    name: "Wei, monthly",
    max_budget: maxBudget,
    current_spend: 0,
  })
  await renderWithRouter(
    <PersonBudget person={PERSON} ceilings={[ceiling]} />,
    {
      url: "/people/om-wei",
      shell: (
        <Provider>
          <Outlet />
        </Provider>
      ),
    },
  )
}

describe("PersonBudget", () => {
  it("says a $0 limit lets nothing be spent, not that the person is on pace", async () => {
    await renderBudget(0)
    await screen.findByRole("link", { name: "Wei, monthly" })
    expect(screen.queryByText(/On pace to stay under/)).toBeNull()
    expect(
      screen.getByText(
        "The limit is $0: nothing may be spent, so every request is refused.",
      ),
    ).toBeVisible()
  })

  it("says a limit with nothing spent yet is on pace", async () => {
    await renderBudget(40)
    expect(
      await screen.findByText("On pace to stay under the limit this period."),
    ).toBeVisible()
  })
})
