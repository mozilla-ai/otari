import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { SessionType } from "@/client"
import { AccountPage } from "@/features/account/AccountPage"
import * as apiClient from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

function renderPage(sessionType: SessionType) {
  return render(
    <AppProviders>
      <DeploymentProvider value={bootstrap({ session_type: sessionType })}>
        <AccountPage />
      </DeploymentProvider>
    </AppProviders>,
  )
}

describe("AccountPage", () => {
  beforeEach(() => {
    // The name card reads the membership context, which is the only request
    // this page makes.
    vi.spyOn(apiClient, "apiFetch").mockResolvedValue(
      organizationContext() as never,
    )
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("offers the name and password forms to the operator this gateway signs in", async () => {
    renderPage("local_operator")

    expect(
      screen.getByRole("heading", { name: "Account settings" }),
    ).toBeInTheDocument()
    // The heading rather than a field: the password form is a dialog now, and
    // which of its three readings applies is the signed-in identity's, not this
    // page's.
    expect(
      screen.getByRole("heading", { name: "Dashboard password" }),
    ).toBeInTheDocument()
    expect(
      await screen.findByRole("heading", { name: "Your name" }),
    ).toBeInTheDocument()
  })

  it("says why there is nothing to change when another control plane owns the session", () => {
    renderPage("hosted_user")

    expect(
      screen.queryByRole("heading", { name: "Dashboard password" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: "Your name" }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByText(
        /managed by the control plane that issued your session/i,
      ),
    ).toBeInTheDocument()
  })
})
