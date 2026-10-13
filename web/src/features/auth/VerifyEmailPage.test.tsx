import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { VerifyEmailPage } from "@/features/auth/VerifyEmailPage"
import { ApiError, apiFetch } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { ThemeProvider } from "@/shared/hooks/useTheme"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { bootstrap } from "@/tests/fixtures"
import { recordEvent, resetTelemetrySpy } from "@/tests/telemetry"

vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

// The telemetry seam, replaced the way a superset build's alias replaces it: the
// base module records nothing, so the outcome this page reports is only
// observable through a stand-in.
vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: vi.fn(() => telemetrySpy) }
})

function renderPage(hash: string) {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap()}>
      <ThemeProvider>
        <QueryClientProvider client={client}>
          <VerifyEmailPage hash={hash} />
        </QueryClientProvider>
      </ThemeProvider>
    </DeploymentProvider>,
  )
}

async function verifyWith(password: string) {
  const user = userEvent.setup()
  await user.type(screen.getByLabelText("Password"), password)
  await user.click(screen.getByRole("button", { name: "Verify email" }))
}

beforeEach(() => {
  vi.clearAllMocks()
  resetTelemetrySpy()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("VerifyEmailPage", () => {
  it("asks for the signup password rather than verifying on arrival", () => {
    // Someone who did not sign up can receive this link; a click alone must
    // not activate whatever password a stranger chose for the address.
    renderPage("#/verify-email?token=abc123")

    expect(apiFetch).not.toHaveBeenCalled()
    expect(screen.getByLabelText("Password")).toBeInTheDocument()
    expect(
      screen.getByRole("link", {
        name: "Didn't sign up? Set a password instead",
      }),
    ).toHaveAttribute("href", "#/recover-password")
  })

  it("sends the token with the password and names the confirmed address", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ email: "ada@example.com" } as never)
    renderPage("#/verify-email?token=abc123")

    await verifyWith("correct-horse")

    expect(
      await screen.findByRole("heading", { name: "Email verified" }),
    ).toBeInTheDocument()
    expect(screen.getByText(/ada@example.com is confirmed/)).toBeInTheDocument()
    const [path, init] = vi.mocked(apiFetch).mock.calls[0] ?? []
    expect(path).toBe("/auth/verify-email")
    expect(JSON.parse(String(init?.body))).toEqual({
      token: "abc123",
      password: "correct-horse",
    })
  })

  it("keeps the form on a refusal, so a mistyped password can be tried again", async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      new ApiError(
        400,
        "This verification link is invalid, expired, or already used, or the password is not the one chosen when signing up",
      ),
    )
    renderPage("#/verify-email?token=abc123")

    await verifyWith("a-typo")

    expect(
      await screen.findByText(/the password is not the one chosen/),
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Password")).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Send a new verification link" }),
    ).toHaveAttribute("href", "#/resend-verification")
  })

  it.each(["#/verify-email", "#/verify-email?token="])(
    "asks the gateway nothing when %s carries no token",
    (hash) => {
      renderPage(hash)

      expect(
        screen.getByText(/missing its verification token/),
      ).toBeInTheDocument()
      expect(apiFetch).not.toHaveBeenCalled()
    },
  )
})

describe("the telemetry the verification page records", () => {
  it("records a confirmed address", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ email: "ada@example.com" } as never)
    renderPage("#/verify-email?token=abc123")

    await verifyWith("correct-horse")

    await screen.findByRole("heading", { name: "Email verified" })
    expect(recordEvent).toHaveBeenCalledTimes(1)
    expect(recordEvent).toHaveBeenCalledWith(
      TELEMETRY_EVENTS.EMAIL_VERIFICATION_SUCCESS,
    )
  })

  it("records a refusal under its status, not its message", async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      new ApiError(400, "This verification link has expired."),
    )
    renderPage("#/verify-email?token=spent")

    await verifyWith("correct-horse")

    await screen.findByText("This verification link has expired.")
    expect(recordEvent).toHaveBeenCalledWith(
      TELEMETRY_EVENTS.EMAIL_VERIFICATION_FAILED,
      { error_code: "verification_rejected", status: 400 },
    )
  })

  it("records nothing for a link that carries no token", async () => {
    renderPage("#/verify-email")

    await screen.findByRole("heading", { name: "Verify your email" })
    expect(recordEvent).not.toHaveBeenCalled()
  })
})
