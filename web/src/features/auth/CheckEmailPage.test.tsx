import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { CheckEmailPage } from "@/features/auth/CheckEmailPage"
import { ThemeProvider } from "@/shared/hooks/useTheme"

function renderPage(hash: string) {
  return render(
    <ThemeProvider>
      <CheckEmailPage hash={hash} />
    </ThemeProvider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  window.location.hash = ""
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
      screen.getByText(/If that address is on this gateway's roster/),
    ).toBeInTheDocument()
  })

  it("uses the resend wording when that is what sent it", () => {
    renderPage("#/check-email?type=resend")

    expect(
      screen.getByText(/If that address is registered and still unverified/),
    ).toBeInTheDocument()
  })
})
