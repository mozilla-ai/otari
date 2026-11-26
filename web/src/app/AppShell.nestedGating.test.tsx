import { screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AppShell } from "@/app/AppShell"
import { Provider } from "@/app/provider"
import type { DeploymentBootstrap } from "@/client"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  BASE_CAPABILITIES,
  EntitlementProvider,
} from "@/shared/hooks/useEntitlements"
import { bootstrap, organizationContext, SURFACES } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

/**
 * A group whose child is served by a surface of its own, which no base row
 * declares any more. Contributed through the row seam, so the rail and the
 * route gate read it exactly as they would a base row. The paths are real
 * routes the registry does not claim; what is under test is the gate, not
 * where the rows point.
 */
vi.mock("@/app/nav/overlayNavItems", async () => {
  const { FiActivity, FiBarChart2 } = await import("react-icons/fi")
  return {
    OVERLAY_NAV_ITEMS: [
      {
        sectionId: "index",
        items: [
          {
            to: "/aliases",
            label: "Reports",
            surface: "routing",
            icon: FiBarChart2,
            children: [
              { to: "/aliases", label: "Spend", icon: FiBarChart2 },
              {
                to: "/activity",
                label: "Activity",
                icon: FiActivity,
                // Grouped with Reports, served by the usage surface.
                surface: "usage",
              },
            ],
          },
        ],
      },
    ],
  }
})

function stubDesktop() {
  vi.stubGlobal(
    "matchMedia",
    vi.fn().mockReturnValue({
      matches: false,
      media: "",
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => true,
    }),
  )
}

function renderShell(deployment: DeploymentBootstrap, url = "/") {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
    String(input) === `${API_ROOT}/organizations/me`
      ? Response.json(organizationContext())
      : Response.json({ data: [], count: 0 }),
  )
  return renderWithRouter(<div>PAGE CONTENT</div>, {
    url,
    shell: (
      <Provider>
        <DeploymentProvider value={deployment}>
          <EntitlementProvider
            value={{ capabilities: BASE_CAPABILITIES, isLoading: false }}
          >
            <SelectedWorkspaceProvider>
              <AppShell />
            </SelectedWorkspaceProvider>
          </EntitlementProvider>
        </DeploymentProvider>
      </Provider>
    ),
  })
}

const WITHOUT_USAGE = SURFACES.filter((surface) => surface !== "usage")

describe("a nested destination with a surface of its own", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    window.localStorage.clear()
  })

  it("is a row of its group when the deployment serves its surface", async () => {
    stubDesktop()
    await renderShell(bootstrap())
    const sidebar = screen.getByRole("navigation", { name: "Sidebar" })

    expect(
      await within(sidebar).findByRole("button", { name: "Reports" }),
    ).toBeInTheDocument()
    expect(within(sidebar).getByText("Activity")).toBeInTheDocument()
  })

  it("is dropped when the deployment lacks its surface, and the group with it", async () => {
    stubDesktop()
    // Gated on its own surface, not its group's, which this deployment serves.
    await renderShell(bootstrap({ surfaces: WITHOUT_USAGE }))
    const sidebar = screen.getByRole("navigation", { name: "Sidebar" })

    // With one child left the row stops being a group: a disclosure that opens
    // onto a single row asks for a click to tell you nothing. It wears the
    // group's name and goes straight to its page.
    const reports = await within(sidebar).findByRole("link", {
      name: "Reports",
    })
    expect(reports).toHaveAttribute("href", "/aliases")
    expect(
      within(sidebar).queryByRole("button", { name: "Reports" }),
    ).toBeNull()
    expect(within(sidebar).queryByText("Activity")).toBeNull()
  })

  it("answers its route with the panel when its surface is gone", async () => {
    stubDesktop()
    // Dropping the row is not enough, because a bookmark still reaches the
    // route; the entry `navItemForPath` answers for it carries its surface.
    await renderShell(bootstrap({ surfaces: WITHOUT_USAGE }), "/activity")

    expect(
      await screen.findByText("Activity is not available here"),
    ).toBeInTheDocument()
    expect(screen.queryByText("PAGE CONTENT")).toBeNull()
  })
})
