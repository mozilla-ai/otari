import { screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { TopBarActions } from "@/app/nav/TopBarActions"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
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
  afterEach(() => {
    vi.restoreAllMocks()
  })

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
    // A member, so Playground's membership gate has an answer to settle on.
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json(organizationContext()),
    )
    await renderWithRouter(
      <AppProviders>
        <DeploymentProvider value={bootstrap()}>
          <TopBarActions />
        </DeploymentProvider>
      </AppProviders>,
    )

    await screen.findByRole("link", { name: "Playground" })
    const cluster = (await screen.findByRole("link", { name: "Documentation" }))
      .parentElement

    expect([...(cluster?.children ?? [])].map((el) => el.textContent)).toEqual([
      "Playground",
      "Documentation",
      "Community",
      "$12.34",
    ])
  })
})
