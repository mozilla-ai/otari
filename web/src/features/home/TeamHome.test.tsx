import { Outlet } from "@tanstack/react-router"
import { screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { TeamHome } from "@/features/home/TeamHome"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  organizationMember,
  SURFACES,
  usageTotals,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

const ADA = organizationMember({
  organization_member_id: "m-ada",
  user_id: "u-ada",
  attribution_user_id: "u-ada",
  full_name: "Ada Lovelace",
})
const WEI = organizationMember({
  organization_member_id: "m-wei",
  user_id: "u-wei",
  attribution_user_id: "u-wei",
  full_name: "Wei Zhang",
})

describe("Home, Where it went", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("shows no columns until the roster says which cut it opens on", async () => {
    let answerRoster: ((response: Response) => void) | undefined
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const path = String(input).slice(API_ROOT.length).split("?")[0]
      if (path === "/organizations/me") {
        return Response.json(
          organizationContext({ role: "owner", deployment_operator: false }),
        )
      }
      if (path === "/organizations/me/members") {
        return new Promise<Response>((resolve) => {
          answerRoster = resolve
        })
      }
      // Every other read is still on its way.
      return new Promise<Response>(() => {})
    })
    await renderWithRouter(<TeamHome />, {
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

    const section = await screen.findByRole("region", {
      name: "Where it went",
    })
    await waitFor(() => expect(answerRoster).toBeDefined())
    expect(within(section).getByText("Loading…")).toBeVisible()
    expect(within(section).queryAllByRole("columnheader")).toHaveLength(0)

    answerRoster?.(Response.json({ data: [ADA, WEI], count: 2 }))
    const table = await within(section).findByRole("grid", {
      name: "Spend by people",
    })
    expect(
      within(table).getByRole("columnheader", { name: "Person" }),
    ).toBeVisible()
  })
})

describe("Home, People active", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  const placed = (id: string, workspaceId: string, name: string) =>
    organizationMember({
      organization_member_id: `m-${id}`,
      user_id: `u-${id}`,
      attribution_user_id: `u-${id}`,
      full_name: name,
      workspaces: [
        {
          workspace_id: workspaceId,
          workspace_member_id: `wm-${id}`,
          workspace_name: workspaceId,
          role: "member",
        },
      ],
    })

  /** Every read answered; returns the trace count requests it saw. */
  function mockApi() {
    const counts: URL[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL(String(input), "http://test")
      const path = url.pathname.replace(
        new URL(API_ROOT, "http://test").pathname,
        "",
      )
      if (path === "/organizations/me") {
        return Response.json(
          organizationContext({
            role: "owner",
            deployment_operator: false,
            // The sidebar has a workspace selected, which the page's own
            // "All workspaces" must not narrow to.
            workspace_memberships: [
              { workspace_id: "ws-a", name: "Platform", role: "owner" },
            ],
          }),
        )
      }
      if (path === "/organizations/me/members") {
        return Response.json({
          data: [
            placed("ada", "ws-a", "Ada Lovelace"),
            placed("lin", "ws-a", "Lin Chen"),
            placed("wei", "ws-b", "Wei Zhang"),
          ],
          count: 3,
        })
      }
      if (path === "/organizations/me/usage/summary") {
        return Response.json({
          totals: usageTotals({ cost: 0, request_count: 0 }),
          series: [],
          by_user: [],
          by_api_key: [],
          by_model: [],
          errors_by_status_code: [],
        })
      }
      if (path.endsWith("/traces/count")) {
        counts.push(url)
        return Response.json({ count: 3 })
      }
      if (path === "/organizations/me/spend-ceilings") {
        return Response.json({ data: [], count: 0 })
      }
      if (path === "/workspaces") return Response.json({ data: [], count: 0 })
      return Response.json([])
    })
    return counts
  }

  async function renderHome(url: string, { hasTraces = true } = {}) {
    await renderWithRouter(<TeamHome />, {
      url,
      shell: (
        <Provider>
          <DeploymentProvider
            value={bootstrap(
              hasTraces ? { surfaces: [...SURFACES, "traces"] } : {},
            )}
          >
            <SelectedWorkspaceProvider>
              <Outlet />
            </SelectedWorkspaceProvider>
          </DeploymentProvider>
        </Provider>
      ),
    })
  }

  it("counts the people of the workspace the page shows, not the organization", async () => {
    mockApi()
    await renderHome("/?workspace=ws-a")
    // Two of the three members are in ws-a, and nobody spent anything.
    expect(await screen.findByText("0 of 2")).toBeVisible()
  })

  it("links the sessions where the deployment records them", async () => {
    mockApi()
    await renderHome("/?workspace=ws-a")
    expect(await screen.findByText("0 of 2")).toBeVisible()
    expect(
      await screen.findByRole("link", { name: "See the sessions" }),
    ).toBeVisible()
    expect(screen.getByText("A row opens its sessions.")).toBeVisible()
  })

  it("offers no sessions where the deployment records none", async () => {
    const counts = mockApi()
    await renderHome("/?workspace=ws-a", { hasTraces: false })
    expect(await screen.findByText("0 of 2")).toBeVisible()
    expect(screen.getByText("in this workspace")).toBeVisible()
    expect(screen.queryByRole("link", { name: "See the sessions" })).toBeNull()
    expect(screen.queryByText("A row opens its sessions.")).toBeNull()
    expect(counts).toHaveLength(0)
  })

  it("counts every workspace's sessions when the page shows them all", async () => {
    const counts = mockApi()
    await renderHome("/")
    await waitFor(() => expect(counts.length).toBeGreaterThan(0))
    for (const url of counts) {
      expect(url.searchParams.get("workspace_id")).toBeNull()
    }
  })
})
