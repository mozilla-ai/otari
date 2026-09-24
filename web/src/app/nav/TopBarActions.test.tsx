import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { WalletNavSlot } from "@/app/nav/overlayWalletSlot"
import { TopBarActions } from "@/app/nav/TopBarActions"
import type { DeploymentBootstrap } from "@/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

// The cluster holds a router Link, so it needs a real router; `renderWithRouter`
// mounts it at "/" and resolves the first location before the assertions run.
function renderActions(
  overrides: Partial<DeploymentBootstrap> = {},
  onOpenFeedback?: () => void,
) {
  return renderWithRouter(
    <AppProviders>
      <DeploymentProvider value={bootstrap(overrides)}>
        <TopBarActions onOpenFeedback={onOpenFeedback} />
      </DeploymentProvider>
    </AppProviders>,
  )
}

const cluster = async () =>
  (await screen.findByRole("link", { name: "Documentation" })).parentElement

describe("TopBarActions", () => {
  it("places the local Playground immediately before Documentation", async () => {
    await renderActions()
    const link = screen.getByRole("link", { name: "Playground" })
    expect(link).toHaveAttribute("href", "/playground")
    expect(link.nextElementSibling).toBe(
      screen.getByRole("link", { name: "Documentation" }),
    )
  })

  it("omits Playground when the deployment does not serve it", async () => {
    await renderActions({ surfaces: [] })
    expect(
      screen.queryByRole("link", { name: "Playground" }),
    ).not.toBeInTheDocument()
    expect(await cluster()).toHaveTextContent("Documentation")
  })

  it("contributes nothing where the balance goes", async () => {
    await renderActions()

    // The design draws a balance at the end of this cluster and this build has
    // none, so the seam is mounted and empty. Asserting the cluster's whole
    // membership rather than the absence of one chip is what would catch a
    // placeholder growing here later.
    expect((await cluster())?.children).toHaveLength(2)
  })

  it("renders the slot module's own empty default", () => {
    // The other half of the assertion above: the cluster is short because the
    // slot renders nothing, not because it was never mounted.
    const { container } = render(<WalletNavSlot />)

    expect(container).toBeEmptyDOMElement()
  })

  it("adds nothing to the cluster on a gateway attached to otari.ai", async () => {
    await renderActions({ management_url: "https://otari.ai/" })

    // A management URL does not retarget the local Playground.
    expect((await cluster())?.children).toHaveLength(2)
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

  it("places Feedback immediately after Documentation and opens it", async () => {
    const onOpenFeedback = vi.fn()
    await renderActions({ feedback_enabled: true }, onOpenFeedback)
    const button = screen.getByRole("button", { name: "Feedback" })
    expect(
      screen.getByRole("link", { name: "Documentation" }).nextElementSibling,
    ).toBe(button)
    // The same class as the links beside it, so it reads as one of them.
    expect(button.className).toContain(
      screen.getByRole("link", { name: "Documentation" }).className,
    )
    await userEvent.setup().click(button)
    expect(onOpenFeedback).toHaveBeenCalledOnce()
  })

  it("offers no Feedback when the deployment has it off", async () => {
    await renderActions({ feedback_enabled: false }, vi.fn())
    expect(await cluster()).toHaveTextContent("Documentation")
    expect(
      screen.queryByRole("button", { name: "Feedback" }),
    ).not.toBeInTheDocument()
  })
})
