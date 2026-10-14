import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { readSignInAddress } from "@/features/auth/signInAddress"
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
  await user.type(await screen.findByLabelText("Password"), password)
  await user.click(screen.getByRole("button", { name: "Verify email" }))
}

const VALIDATE_PATH = "/auth/verify-email/validate"

// The gateway as this page meets it: the arrival check answers `linkStatus`,
// and every other call is `redeem`, which each test sets up.
const redeem = vi.fn()
let linkStatus = "valid"

beforeEach(() => {
  vi.clearAllMocks()
  resetTelemetrySpy()
  window.sessionStorage.clear()
  redeem.mockReset()
  linkStatus = "valid"
  vi.mocked(apiFetch).mockImplementation(async (path, init) =>
    path === VALIDATE_PATH ? { status: linkStatus } : redeem(path, init),
  )
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("VerifyEmailPage", () => {
  it("asks for the signup password rather than verifying on arrival", async () => {
    // Someone who did not sign up can receive this link; a click alone must
    // not activate whatever password a stranger chose for the address.
    renderPage("#/verify-email?token=abc123")

    expect(await screen.findByLabelText("Password")).toBeInTheDocument()
    expect(vi.mocked(apiFetch).mock.calls[0]?.[0]).toBe(VALIDATE_PATH)
    expect(redeem).not.toHaveBeenCalled()
    expect(
      screen.getByRole("link", {
        name: "Forgot that password, or didn't sign up? Set a new one",
      }),
    ).toHaveAttribute("href", "#/recover-password")
  })

  // Only the first signup's password verifies, so someone who signed up twice
  // has to be told which one to type.
  it("tells someone who signed up twice which password verifies", async () => {
    renderPage("#/verify-email?token=abc123")

    expect(
      await screen.findByText(
        /If you signed up more than once, use the password you chose first/,
      ),
    ).toBeInTheDocument()
  })

  it.each([
    ["invalid", "This link no longer works"],
    ["expired", "This link has expired"],
  ])(
    "says a %s link cannot be used before asking for a password",
    async (status, title) => {
      linkStatus = status
      renderPage("#/verify-email?token=spent")

      expect(
        await screen.findByRole("heading", { name: title }),
      ).toBeInTheDocument()
      expect(screen.queryByLabelText("Password")).toBeNull()
      expect(
        screen.getByRole("link", { name: "Send a new verification link" }),
      ).toHaveAttribute("href", "#/resend-verification")
    },
  )

  it("hands the confirmed address to the sign-in screen", async () => {
    redeem.mockResolvedValue({ email: "ada@example.com" })
    renderPage("#/verify-email?token=abc123")

    await verifyWith("correct-horse")

    await screen.findByRole("heading", { name: "Email verified" })
    expect(readSignInAddress()).toBe("ada@example.com")
  })

  it("sends the token with the password and names the confirmed address", async () => {
    redeem.mockResolvedValue({ email: "ada@example.com" } as never)
    renderPage("#/verify-email?token=abc123")

    await verifyWith("correct-horse")

    expect(
      await screen.findByRole("heading", { name: "Email verified" }),
    ).toBeInTheDocument()
    expect(screen.getByText(/ada@example.com is confirmed/)).toBeInTheDocument()
    const [path, init] = redeem.mock.calls[0] ?? []
    expect(path).toBe("/auth/verify-email")
    expect(JSON.parse(String(init?.body))).toEqual({
      token: "abc123",
      password: "correct-horse",
    })
  })

  it("keeps the form on a refusal, so a mistyped password can be tried again", async () => {
    redeem.mockRejectedValue(
      new ApiError(
        400,
        'That is not the password chosen when signing up. This link still works, so try again, or choose a new password with "Forgot your password?".',
      ),
    )
    renderPage("#/verify-email?token=abc123")

    await verifyWith("a-typo")

    expect(
      await screen.findByText(/This link still works, so try again/),
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
    redeem.mockResolvedValue({ email: "ada@example.com" } as never)
    renderPage("#/verify-email?token=abc123")

    await verifyWith("correct-horse")

    await screen.findByRole("heading", { name: "Email verified" })
    expect(recordEvent).toHaveBeenCalledTimes(1)
    expect(recordEvent).toHaveBeenCalledWith(
      TELEMETRY_EVENTS.EMAIL_VERIFICATION_SUCCESS,
    )
  })

  it("records a refusal under its status, not its message", async () => {
    redeem.mockRejectedValue(
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
