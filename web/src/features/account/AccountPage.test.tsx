import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { AccountPage } from "@/features/account/AccountPage"
import * as apiClient from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

function renderPage(passkeysEnabled = true) {
  return render(
    <AppProviders>
      <DeploymentProvider
        value={bootstrap({ passkeys_enabled: passkeysEnabled })}
      >
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
    renderPage()

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

  it("shows the passkeys page unless the deployment turned passkeys off", async () => {
    const { unmount } = renderPage()
    expect(
      await screen.findByRole("heading", { name: "Passkeys" }),
    ).toBeInTheDocument()
    unmount()
    vi.mocked(apiClient.apiFetch).mockClear()

    renderPage(false)
    expect(
      await screen.findByRole("heading", { name: "Your name" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("heading", { name: "Passkeys" })).toBeNull()
    // The page that would have asked routes that are not mounted never ran.
    expect(
      vi
        .mocked(apiClient.apiFetch)
        .mock.calls.some(([path]) => String(path).includes("/auth/webauthn")),
    ).toBe(false)
  })
})
