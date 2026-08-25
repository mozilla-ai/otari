import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { WalletNavSlot } from "@/app/nav/overlayWalletSlot"
import { TopBarActions } from "@/app/nav/TopBarActions"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

// The cluster holds a router Link, so it needs a real router; `renderWithRouter`
// mounts it at "/" and resolves the first location before the assertions run.
function renderActions(management_url: string | null = null) {
  return renderWithRouter(
    <AppProviders>
      <DeploymentProvider value={bootstrap({ management_url })}>
        <TopBarActions />
      </DeploymentProvider>
    </AppProviders>,
  )
}

const cluster = async () =>
  (await screen.findByRole("link", { name: "Documentation" })).parentElement

describe("TopBarActions", () => {
  it("contributes nothing where the balance goes", async () => {
    await renderActions()

    // The design draws a balance at the end of this cluster and this build has
    // none, so the seam is mounted and empty. Asserting the cluster's whole
    // membership rather than the absence of one chip is what would catch a
    // placeholder growing here later.
    expect((await cluster())?.children).toHaveLength(1)
  })

  it("renders the slot module's own empty default", () => {
    // The other half of the assertion above: the cluster is short because the
    // slot renders nothing, not because it was never mounted.
    const { container } = render(<WalletNavSlot />)

    expect(container).toBeEmptyDOMElement()
  })

  it("still omits the hosted playground on a gateway that has one to point at", async () => {
    await renderActions("https://otari.ai/")

    // `management_url` is only half of that gate: the base build grants no
    // capabilities, so the entitlement half withholds the link regardless.
    expect(
      screen.queryByRole("link", { name: "Playground" }),
    ).not.toBeInTheDocument()
  })
})
