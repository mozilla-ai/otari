import { screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AccountMenu } from "@/app/nav/AccountMenu"
import { PasswordCard } from "@/features/account/PasswordCard"
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

  it("reports the deployment as unclaimed rather than naming a credential", async () => {
    // `sign_in_methods` says whether the operator has claimed, not how this
    // caller signed in, and on an unclaimed deployment both a master-key
    // session and a signed-up member's password session are possible
    // (otari#702). Naming either would name the other one wrongly.
    await openMenu(["master_key"])

    expect(await screen.findByText("Unclaimed deployment")).toBeInTheDocument()
    expect(screen.queryByText("Master-key session")).not.toBeInTheDocument()
  })

  it("names the password sign-in once an operator has claimed the deployment", async () => {
    await openMenu(["password"])

    expect(await screen.findByText("Password sign-in")).toBeInTheDocument()
    expect(screen.queryByText("Unclaimed deployment")).not.toBeInTheDocument()
  })
})

describe("AccountMenu after a claim", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("stops the whole tab offering the master key once the claim lands", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          email: "operator@example.com",
          master_key_sign_in_retired: true,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )
    const user = userEvent.setup()
    // One provider over both, as the shell mounts them: the card is the page
    // and the menu is the chrome around it. The menu reads the same bootstrap
    // the card claims against, and a claim that only flipped the card's own
    // state would leave this menu naming a session kind that has ended, and a
    // later sign-out on a login screen offering a credential the gateway now
    // refuses.
    await renderWithRouter(
      <AppProviders>
        <DeploymentProvider
          value={bootstrap({ sign_in_methods: ["master_key"] })}
        >
          <AccountMenu collapsed={false} />
          <PasswordCard />
        </DeploymentProvider>
      </AppProviders>,
    )

    await user.click(screen.getByRole("button", { name: "Account" }))
    expect(await screen.findByText("Unclaimed deployment")).toBeInTheDocument()
    await user.keyboard("{Escape}")

    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("New password"), "a-real-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "a-real-password",
    )
    await user.click(screen.getByRole("button", { name: "Set password" }))
    expect(await screen.findByLabelText("Current password")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Account" }))
    expect(await screen.findByText("Password sign-in")).toBeInTheDocument()
    expect(screen.queryByText("Unclaimed deployment")).not.toBeInTheDocument()
  })
})
