import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { CheckEmailPage } from "@/features/auth/CheckEmailPage"
import { rememberSignInAddress } from "@/features/auth/signInAddress"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { ThemeProvider } from "@/shared/hooks/useTheme"
import { bootstrap } from "@/tests/fixtures"

function renderPage(hash: string) {
  return render(
    <DeploymentProvider value={bootstrap()}>
      <ThemeProvider>
        <CheckEmailPage hash={hash} />
      </ThemeProvider>
    </DeploymentProvider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  window.location.hash = ""
  window.sessionStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
  window.location.hash = ""
})

describe("CheckEmailPage", () => {
  it("says what was sent without ever saying whether the address exists", () => {
    renderPage("#/check-email?type=signup")

    expect(screen.getByRole("main")).toContainElement(
      screen.getByRole("heading", { name: "Check your email" }),
    )

    expect(
      screen.getByText(/If you can sign up with that address/),
    ).toBeInTheDocument()
  })

  it("names the address the page before mailed, in the same conditional", () => {
    rememberSignInAddress("ada@example.com")
    renderPage("#/check-email?type=signup")

    expect(
      screen.getByText(/If you can sign up with ada@example.com/),
    ).toBeInTheDocument()
  })

  it("uses the resend wording when that is what sent it", () => {
    renderPage("#/check-email?type=resend")

    expect(
      screen.getByText(/If that address has an account waiting to be verified/),
    ).toBeInTheDocument()
  })
})
