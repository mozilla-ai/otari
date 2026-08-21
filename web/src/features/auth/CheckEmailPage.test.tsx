import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { CheckEmailPage } from "@/features/auth/CheckEmailPage"

function renderPage(hash: string) {
  return render(<CheckEmailPage hash={hash} />)
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
