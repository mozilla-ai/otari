import { screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { TopBarActions } from "@/app/nav/TopBarActions"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

/**
 * The seam as a build that replaces the module sees it, which is the one thing
 * `TopBarActions.test.tsx` cannot show: the slot renders nothing there, so every
 * assertion in it holds whether or not the top bar mounts the slot at all.
 * Deleting `<WalletNavSlot />` from the cluster leaves that suite green and turns
 * this file red, which is the point of it.
 *
 * Mocked by the same `@/…` specifier the top bar imports, because that is the
 * one a superset build's alias replaces.
 */
vi.mock("@/app/nav/overlayWalletSlot", () => ({
  WalletNavSlot: () => <span>$12.34</span>,
}))

describe("a build that replaces the wallet-slot module", () => {
  it("renders its chip in the top bar", async () => {
    await renderWithRouter(
      <AppProviders>
        <DeploymentProvider value={bootstrap()}>
          <TopBarActions />
        </DeploymentProvider>
      </AppProviders>,
    )

    expect(await screen.findByText("$12.34")).toBeInTheDocument()
  })

  it("puts it at the end of the cluster, after the base links", async () => {
    await renderWithRouter(
      <AppProviders>
        <DeploymentProvider value={bootstrap()}>
          <TopBarActions />
        </DeploymentProvider>
      </AppProviders>,
    )

    const cluster = (await screen.findByRole("link", { name: "Documentation" }))
      .parentElement

    expect([...(cluster?.children ?? [])].map((el) => el.textContent)).toEqual([
      "Documentation",
      "$12.34",
    ])
  })
})
