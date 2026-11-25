import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { MailReadiness } from "@/client"
import { CheckEmailPage } from "@/features/auth/CheckEmailPage"
import { rememberSignInAddress } from "@/features/auth/signInAddress"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { ThemeProvider } from "@/shared/hooks/useTheme"
import { bootstrap } from "@/tests/fixtures"

function renderPage(route: string, mail: MailReadiness = "delivers") {
  return render(
    <DeploymentProvider value={bootstrap({ mail })}>
      <ThemeProvider>
        <CheckEmailPage route={route} />
      </ThemeProvider>
    </DeploymentProvider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  window.history.replaceState(null, "", "/")
  window.sessionStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
  window.history.replaceState(null, "", "/")
})

describe("CheckEmailPage", () => {
  it("says what was sent without ever saying whether the address exists", () => {
    renderPage("/check-email?type=signup")

    expect(screen.getByRole("main")).toContainElement(
      screen.getByRole("heading", { name: "Check your email" }),
    )

    expect(
      screen.getByText(/If you can sign up with that address/),
    ).toBeInTheDocument()
  })

  it("names the address the page before mailed, in the same conditional", () => {
    rememberSignInAddress("ada@example.com")
    renderPage("/check-email?type=signup")

    expect(
      screen.getByText(/If you can sign up with ada@example.com/),
    ).toBeInTheDocument()
  })

  it("uses the resend wording when that is what sent it", () => {
    renderPage("/check-email?type=resend")

    expect(
      screen.getByText(/If that address has an account waiting to be verified/),
    ).toBeInTheDocument()
  })

  it("promises no inbox on a deployment whose email reaches nobody", () => {
    rememberSignInAddress("ada@example.com")
    renderPage("/check-email?type=signup", "sends_only")

    expect(
      screen.queryByRole("heading", { name: "Check your email" }),
    ).not.toBeInTheDocument()
    expect(screen.queryByText(/we have sent/)).not.toBeInTheDocument()
    expect(
      screen.getByText(/If you can sign up with ada@example.com/),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/cannot email you a link that works, so ask whoever/),
    ).toBeInTheDocument()
  })

  it("keeps the resend wording conditional where email reaches nobody", () => {
    renderPage("/check-email?type=resend", "sends_only")

    expect(
      screen.getByText(/If that address has an account waiting to be verified/),
    ).toBeInTheDocument()
    expect(screen.queryByText(/we have sent/)).not.toBeInTheDocument()
  })

  it.each([
    ["delivers", "signup", "If you can sign up with ada@example.com"],
    ["delivers", "resend", "If ada@example.com has an account waiting"],
    ["sends_only", "signup", "If you can sign up with ada@example.com"],
    ["sends_only", "resend", "If ada@example.com has an account waiting"],
  ] as const)(
    "opens in the conditional on every mail state (%s, %s)",
    (mail, type, clause) => {
      // The enumeration-safety invariant: no wording may start by stating that
      // there was an identity to act on.
      rememberSignInAddress("ada@example.com")
      renderPage(`/check-email?type=${type}`, mail)

      expect(
        screen.getByText(new RegExp(`^${clause.replace(".", "\\.")}`)),
      ).toBeInTheDocument()
    },
  )
})
