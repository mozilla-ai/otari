import { screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AppShell } from "@/app/AppShell"
import { Provider } from "@/app/provider"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

/**
 * The seam as a build that replaces the module sees it. The base renders
 * nothing, so `AppShell.test.tsx` passes whether or not the shell mounts it;
 * deleting `<ShellBanner />` turns this file red.
 *
 * Mocked by the same `@/…` specifier the shell imports, because that is the
 * one a superset build's alias replaces.
 */
vi.mock("@/app/overlayShellBanner", () => ({
  ShellBanner: () => <p>BANNER</p>,
}))

async function renderShell() {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
    Response.json(organizationContext()),
  )
  return renderWithRouter(<p>PAGE</p>, {
    shell: (
      <Provider>
        <DeploymentProvider value={bootstrap()}>
          <SelectedWorkspaceProvider>
            <AppShell />
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </Provider>
    ),
  })
}

describe("a build that replaces the shell-banner module", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("renders its band once, above the header and outside the page", async () => {
    await renderShell()

    expect(await screen.findByText("PAGE")).toBeInTheDocument()
    const banner = screen.getByText("BANNER")
    expect(screen.getAllByText("BANNER")).toHaveLength(1)
    expect(banner.closest("main")).toBeNull()
    expect(banner.closest("header")).toBeNull()
    // In flow before the row the header leads, so the mobile drawer, which is
    // positioned against that row, moves down with the header.
    const header = screen.getByRole("banner")
    expect(
      banner.compareDocumentPosition(header) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
  })
})
