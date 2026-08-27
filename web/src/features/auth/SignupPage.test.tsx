import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { SignupPage } from "@/features/auth/SignupPage"
import { ApiError, apiFetch } from "@/shared/api/client"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { recordEvent, resetTelemetrySpy } from "@/tests/telemetry"

// The network boundary, not the hooks: the real hooks, their query keys, and
// the mutation state the page branches on all stay live.
vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

// The telemetry seam, replaced the way a superset build's alias replaces it: the
// base module records nothing, so the funnel is only observable through a
// stand-in.
vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: vi.fn(() => telemetrySpy) }
})

// `PublicAuthPage` passes the whole hash down, so the plain page is the one
// reached from the sign-in screen and a `?email=` one is the accept page's
// handoff (otari#835).
function renderPage(hash = "#/signup") {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SignupPage hash={hash} />
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  resetTelemetrySpy()
  window.location.hash = ""
})

afterEach(() => {
  vi.restoreAllMocks()
  window.location.hash = ""
})

describe("SignupPage", () => {
  it("claims the identity and lands on the check-email page", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    await vi.waitFor(() => {
      expect(window.location.hash).toBe("#/check-email?type=signup")
    })
    const [path, init] = vi.mocked(apiFetch).mock.calls[0] ?? []
    expect(path).toBe("/v1/auth/signup")
    expect(JSON.parse(String(init?.body))).toEqual({
      email: "ada@example.com",
      password: "correct-horse",
      full_name: null,
    })
  })

  it("keeps the button disabled until the two passwords agree", async () => {
    const user = userEvent.setup()
    renderPage()

    const submit = screen.getByRole("button", { name: "Claim account" })
    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "typo")

    expect(
      await screen.findByText("The two passwords do not match."),
    ).toBeInTheDocument()
    expect(submit).toBeDisabled()
    expect(apiFetch).not.toHaveBeenCalled()
  })

  it("refuses a password the gateway would refuse, without asking it", async () => {
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "short")
    await user.type(screen.getByLabelText("Confirm password"), "short")

    expect(
      await screen.findByText("At least 8 characters."),
    ).toBeInTheDocument()
    expect(apiFetch).not.toHaveBeenCalled()
  })

  it("shows the gateway's own refusal and stays on the form", async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      new ApiError(503, "Outgoing mail is not configured on this deployment."),
    )
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    expect(
      await screen.findByText(
        "Outgoing mail is not configured on this deployment.",
      ),
    ).toBeInTheDocument()
    // No navigation: the page only leaves for #/check-email on a success.
    expect(window.location.hash).toBe("")
  })
})

describe("the telemetry the signup page records", () => {
  it("records the attempt and then the claim it produced", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.SIGNUP_STARTED, {
      authentication_method: "password",
    })
    await vi.waitFor(() => {
      expect(recordEvent).toHaveBeenCalledWith(
        TELEMETRY_EVENTS.SIGNUP_SUCCESS,
        // Always verification-bound: this endpoint is enumeration-safe, so the
        // page reads nothing back and neither does this.
        { authentication_method: "password", requires_verification: true },
      )
    })
  })

  it("records a refused claim under its status, not its message", async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      new ApiError(503, "Outgoing mail is not configured on this deployment."),
    )
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    await vi.waitFor(() => {
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.SIGNUP_FAILED, {
        authentication_method: "password",
        status: 503,
      })
    })
  })

  it("records nothing for a form its own button will not submit", async () => {
    // This page validates by disabling the submit rather than by refusing one,
    // so there is no moment at which a validation failure could be recorded and
    // none is manufactured. `FORM_VALIDATION_FAILED` comes from the sign-in
    // screen, which keeps its button live on purpose.
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "typo")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    expect(recordEvent).not.toHaveBeenCalled()
  })

  it("prefills the invited address, read-only, and claims that one", async () => {
    // The address is the invitation's, not the visitor's to choose: another one
    // has nothing to claim, and signup answers the same enumeration-safe
    // sentence either way, so an editable field would fail silently.
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage("#/signup?email=ada%40example.com")

    const emailField = screen.getByLabelText("Email")
    expect(emailField).toHaveValue("ada@example.com")
    expect(emailField).toHaveAttribute("readonly")

    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    await vi.waitFor(() => {
      expect(window.location.hash).toBe("#/check-email?type=signup")
    })
    const [, init] = vi.mocked(apiFetch).mock.calls[0] ?? []
    expect(JSON.parse(String(init?.body)).email).toBe("ada@example.com")
  })

  it("offers the plain form to anyone who needs another address", () => {
    // The way out of the read-only field, so a prefill that is wrong for this
    // visitor is not a dead end of its own.
    renderPage("#/signup?email=ada%40example.com")

    expect(
      screen.getByRole("link", { name: "Claim a different address instead" }),
    ).toHaveAttribute("href", "#/signup")
  })

  it("asks for the address when the link carries none", () => {
    renderPage()

    const emailField = screen.getByLabelText("Email")
    expect(emailField).toHaveValue("")
    expect(emailField).not.toHaveAttribute("readonly")
    expect(
      screen.queryByRole("link", { name: "Claim a different address instead" }),
    ).toBeNull()
  })
})
