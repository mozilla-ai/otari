import { screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"

import { AccountMenu } from "@/app/nav/AccountMenu"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

// The menu holds a router Link, so it needs a real router; `renderWithRouter`
// mounts it at "/" and resolves the first location before the assertions run.
async function openMenu(overrides: Partial<{ docs_url: string | null }> = {}) {
  await renderWithRouter(
    <AppProviders>
      <DeploymentProvider value={bootstrap(overrides)}>
        <AccountMenu collapsed={false} />
      </DeploymentProvider>
    </AppProviders>,
  )
  await userEvent.setup().click(screen.getByRole("button", { name: "Account" }))
}

describe("AccountMenu", () => {
  it("opens the account page, rather than naming a destination it cannot reach", async () => {
    await openMenu()

    const link = await screen.findByRole("link", { name: "Account settings" })
    expect(link).toHaveAttribute("href", "/account")
  })

  it("keeps the bundled guide reachable on a phone when no docs site is configured", async () => {
    await openMenu()

    const link = await screen.findByRole("link", { name: "Documentation" })
    expect(link).toHaveAttribute("href", "/docs")
    expect(link).not.toHaveAttribute("target")
  })

  it("retargets the phone's Documentation row at the deployment's own docs site", async () => {
    await openMenu({ docs_url: "https://docs.otari.ai/en/" })

    const link = await screen.findByRole("link", { name: "Documentation" })
    expect(link).toHaveAttribute("href", "https://docs.otari.ai/en/")
    expect(link).toHaveAttribute("target", "_blank")
    // Still the row the top bar hands off to below `md`, not a second entry
    // point: retargeting it must not make it visible where the cluster already
    // draws one.
    expect(link).toHaveClass("md:hidden")
  })
})
