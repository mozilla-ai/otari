import { screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AppShell } from "@/app/AppShell"
import { Provider } from "@/app/provider"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

/**
 * The seam as a build that replaces the module sees it. The base renders
 * nothing, so nothing else in the suite can tell whether the header mounts it.
 *
 * Mocked by the same `@/…` specifier the shell imports, because that is the
 * one a superset build's alias replaces.
 */
vi.mock("@/app/nav/overlayMobileWalletSlot", () => ({
  MobileWalletNavSlot: () => <span>$12.34</span>,
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

describe("a build that replaces the mobile wallet-slot module", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("renders its chip in the header, in a wrapper hidden from md up", async () => {
    await renderShell()

    const header = screen.getByRole("banner")
    const chip = await within(header).findByText("$12.34")
    // jsdom has no layout, so the breakpoint is asserted as the class that
    // carries it. The desktop cluster holds the other wallet slot.
    expect(chip.parentElement?.className).toContain("md:hidden")
  })
})
