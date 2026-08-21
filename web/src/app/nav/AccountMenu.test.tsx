import { screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AccountMenu } from "@/app/nav/AccountMenu"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

// The menu holds a router Link, so it needs a real router; `renderWithRouter`
// mounts it at "/" and resolves the first location before the assertions run.
async function openMenu(
  signInMethods: ("master_key" | "password")[] = ["master_key"],
) {
  await renderWithRouter(
    <AppProviders>
      <DeploymentProvider value={bootstrap({ sign_in_methods: signInMethods })}>
        <AccountMenu collapsed={false} />
      </DeploymentProvider>
    </AppProviders>,
  )
  await userEvent.setup().click(screen.getByRole("button", { name: "Account" }))
}

describe("AccountMenu", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("opens the account page, rather than naming a destination it cannot reach", async () => {
    await openMenu()

    const link = await screen.findByRole("link", { name: "Account settings" })
    expect(link).toHaveAttribute("href", "/account")
  })

  it("names the master key while the deployment is still unclaimed", async () => {
    await openMenu(["master_key"])

    expect(await screen.findByText("Master-key session")).toBeInTheDocument()
  })

  it("stops naming the master key once an operator has claimed the deployment", async () => {
    await openMenu(["password"])

    expect(await screen.findByText("Password sign-in")).toBeInTheDocument()
    expect(screen.queryByText("Master-key session")).not.toBeInTheDocument()
  })
})
