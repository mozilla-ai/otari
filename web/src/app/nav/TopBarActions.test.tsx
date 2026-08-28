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
function renderActions(
  overrides: Partial<{
    management_url: string | null
    docs_url: string | null
  }> = {},
) {
  return renderWithRouter(
    <AppProviders>
      <DeploymentProvider value={bootstrap(overrides)}>
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

  it("adds nothing to the cluster on a gateway attached to otari.ai", async () => {
    await renderActions({ management_url: "https://otari.ai/" })

    // The hosted Playground was the one link this cluster derived from
    // `management_url`, and otari-ai#1909 retired it along with its backend.
    // Asserting the whole membership rather than the absence of that one link
    // is what would catch another hosted link reappearing here.
    expect((await cluster())?.children).toHaveLength(1)
  })

  it("points Documentation at the bundled guide when no docs site is configured", async () => {
    await renderActions()

    // The hash route, because the guide ships with this gateway: a router Link
    // renders the router's own href rather than an absolute URL.
    const link = await screen.findByRole("link", { name: "Documentation" })
    expect(link).toHaveAttribute("href", "/docs")
    expect(link).not.toHaveAttribute("target")
  })

  it("retargets Documentation at the deployment's own docs site", async () => {
    await renderActions({ docs_url: "https://docs.otari.ai/en/" })

    const link = await screen.findByRole("link", { name: "Documentation" })
    expect(link).toHaveAttribute("href", "https://docs.otari.ai/en/")
    // A new tab, like every other link out of the dashboard: this one leaves the
    // app, where the bundled guide is a page inside it.
    expect(link).toHaveAttribute("target", "_blank")
    expect(link).toHaveAttribute("rel", "noopener noreferrer")
  })
})
