import { Outlet } from "@tanstack/react-router"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { OrganizationMembersPage } from "@/features/organization/OrganizationMembersPage"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  organizationMember,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

const OWNER = organizationMember({
  organization_member_id: "owner-membership",
  user_id: "aaaaaaaa-0000-0000-0000-000000000000",
  full_name: "Operator",
  role: "owner",
})
const ANALYST = organizationMember({
  organization_member_id: "analyst-membership",
  user_id: "bbbbbbbb-0000-0000-0000-000000000000",
  attribution_user_id: "bbbbbbbb-0000-0000-0000-000000000000",
  full_name: "Analyst",
  role: "member",
})

/** An owner's roster, with the organization's limits refused or never answered. */
async function renderRoster(ceilings: "refused" | "pending") {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url === `${API_ROOT}/organizations/me`) {
      return Response.json(
        organizationContext({ role: "owner", deployment_operator: false }),
      )
    }
    if (url.includes("/organizations/me/spend-ceilings")) {
      if (ceilings === "pending") return new Promise<Response>(() => {})
      return Response.json({ detail: "Forbidden" }, { status: 403 })
    }
    if (url.startsWith(`${API_ROOT}/organizations/me/members?`)) {
      return Response.json({ data: [OWNER, ANALYST], count: 2 })
    }
    return Response.json({ data: [], count: 0 })
  })
  await renderWithRouter(<OrganizationMembersPage />, {
    shell: (
      <Provider>
        <DeploymentProvider value={bootstrap()}>
          <SelectedWorkspaceProvider>
            <Outlet />
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </Provider>
    ),
  })
}

describe("a member's spend limit, from their row", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  for (const [state, hint] of [
    ["refused", "Could not read"],
    ["pending", "Loading…"],
  ] as const) {
    it(`is not offered while the limits are ${state}`, async () => {
      const user = userEvent.setup()
      await renderRoster(state)
      const trigger = await screen.findByRole("button", {
        name: /^Actions for Analyst/,
      })
      await waitFor(() =>
        expect(trigger).toHaveAccessibleName(
          `Actions for Analyst. Set spend limit… ${hint}.`,
        ),
      )
      await user.click(trigger)
      const item = await screen.findByRole("menuitem", {
        name: /^Set spend limit…/,
      })
      expect(item).toHaveAttribute("aria-disabled", "true")
      expect(within(item).getByText(hint)).toBeInTheDocument()
    })
  }
})
