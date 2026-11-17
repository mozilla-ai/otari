import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { OverviewIndex } from "@/features/overview/OverviewPage"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  organizationMember,
  SURFACES,
  workspaceActivation,
} from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

afterEach(() => {
  vi.restoreAllMocks()
})

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })

/** The caller is "me-id"; the gateway serves what Sessions' filter bar reads. */
function mockGateway(context: Parameters<typeof organizationContext>[0]) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), "http://localhost")
    const path = url.pathname.replace(/^.*\/api\/v1/, "")
    if (path === "/organizations/me") {
      return json(
        organizationContext({
          workspace_memberships: [
            {
              workspace_id: "ws-1",
              name: "Production",
              role: "owner",
            },
          ],
          ...context,
        }),
      )
    }
    if (path === "/organizations/me/members") {
      const member = organizationMember({ attribution_user_id: "me-id" })
      return json({ data: [member], total: 1 })
    }
    if (path.includes("/activation")) return json(workspaceActivation())
    if (path === "/models") {
      // Something to route to, so the setup guide's gate is open.
      return json({
        object: "list",
        data: [{ id: "m", object: "model", created: 0, owned_by: "openai" }],
      })
    }
    if (path === "/providers") {
      return json({ providers: [{ provider: "openai" }] })
    }
    if (path.endsWith("/hosted-providers")) return json({ data: [], count: 0 })
    return json([])
  })
}

function renderSessions(url: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider
        value={bootstrap({ surfaces: [...SURFACES, "traces"] })}
      >
        <SelectedWorkspaceProvider>
          <OverviewIndex />
        </SelectedWorkspaceProvider>
      </DeploymentProvider>
    </QueryClientProvider>,
    { wrapper: withRouter({ url }) },
  )
}

const REMOVE_USER_FILTER = { name: "Remove User filter", hidden: true }

describe("Sessions scope", () => {
  it("opens a member on everyone's requests, narrowed by the User filter", async () => {
    mockGateway({ deployment_operator: false, role: "member" })
    renderSessions("/requests")

    await screen.findByRole("navigation", {
      name: "Sessions views",
      hidden: true,
    })
    expect(screen.queryByText("Mine")).toBeNull()
    expect(screen.queryByRole("button", REMOVE_USER_FILTER)).toBeNull()
  })

  it("reads a legacy Mine link as the caller, with a chip to drop it", async () => {
    mockGateway({ deployment_operator: false, role: "admin" })
    renderSessions("/requests?scope=mine")

    const remove = await screen.findByRole("button", REMOVE_USER_FILTER)
    const user = userEvent.setup()
    await user.click(remove)
    await waitFor(() =>
      expect(screen.queryByRole("button", REMOVE_USER_FILTER)).toBeNull(),
    )
  })
})
