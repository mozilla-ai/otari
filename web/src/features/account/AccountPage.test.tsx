import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { SessionType } from "@/client"
import { AccountPage } from "@/features/account/AccountPage"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
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
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("offers the password form to the operator this gateway signs in", () => {
    renderPage("local_operator")

    expect(
      screen.getByRole("heading", { name: "Account settings" }),
    ).toBeInTheDocument()
    expect(screen.getByLabelText("New password")).toBeInTheDocument()
  })

  it("says why there is nothing to change when another control plane owns the session", () => {
    renderPage("hosted_user")

    expect(screen.queryByLabelText("New password")).not.toBeInTheDocument()
    expect(
      screen.getByText(
        /managed by the control plane that issued your session/i,
      ),
    ).toBeInTheDocument()
  })
})
