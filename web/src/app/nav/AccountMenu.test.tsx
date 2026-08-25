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
async function openMenu() {
  await renderWithRouter(
    <AppProviders>
      <DeploymentProvider value={bootstrap()}>
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
})
