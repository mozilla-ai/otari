import { Outlet } from "@tanstack/react-router"
import { screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"
import { McpServersPage } from "./McpServersPage"

describe("McpServersPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("titles the page MCP servers, with no Tools title or tab row", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/organizations/me")) {
        return Response.json(organizationContext({ role: "owner" }))
      }
      return Response.json({ data: [], count: 0 })
    })
    await renderWithRouter(<McpServersPage />, {
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

    expect(
      await screen.findByRole("heading", { name: "MCP servers", level: 1 }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("heading", { name: "Tools" })).toBeNull()
    expect(screen.queryByRole("navigation", { name: "Tools views" })).toBeNull()
  })
})
