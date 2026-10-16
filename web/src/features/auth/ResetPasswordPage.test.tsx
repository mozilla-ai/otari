import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ResetPasswordPage } from "@/features/auth/ResetPasswordPage"
import { ApiError, apiFetch } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { ThemeProvider } from "@/shared/hooks/useTheme"
import { bootstrap } from "@/tests/fixtures"

vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

function renderPage(hash: string, termsUrl: string | null = null) {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap({ terms_url: termsUrl })}>
      <ThemeProvider>
        <QueryClientProvider client={client}>
          <ResetPasswordPage hash={hash} />
        </QueryClientProvider>
      </ThemeProvider>
    </DeploymentProvider>,
  )
}

const VALIDATE_PATH = "/auth/password/reset/validate"

// The gateway as this page meets it: the arrival check answers `linkStatus`,
// and every other call is `redeem`, which each test sets up.
const redeem = vi.fn()
let linkStatus = "valid"

beforeEach(() => {
  vi.clearAllMocks()
  redeem.mockReset()
  linkStatus = "valid"
  vi.mocked(apiFetch).mockImplementation(async (path, init) =>
    path === VALIDATE_PATH ? { status: linkStatus } : redeem(path, init),
  )
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ResetPasswordPage", () => {
  it("only checks the link on arrival, then waits for the new password", async () => {
    renderPage("#/reset-password?token=abc123")

    expect(await screen.findByLabelText("New password")).toBeInTheDocument()
    const [path, init] = vi.mocked(apiFetch).mock.calls[0] ?? []
    expect(path).toBe(VALIDATE_PATH)
    expect(JSON.parse(String(init?.body))).toEqual({ token: "abc123" })
    expect(redeem).not.toHaveBeenCalled()
  })

  // A spent link used to show the whole form and refuse only after a new
  // password had been typed twice.
  it.each([
    ["invalid", "This link no longer works"],
    ["expired", "This link has expired"],
  ])(
    "says a %s link cannot be used before asking for a password",
    async (status, title) => {
      linkStatus = status
      renderPage("#/reset-password?token=spent")

      expect(
        await screen.findByRole("heading", { name: title }),
      ).toBeInTheDocument()
      expect(screen.queryByLabelText("New password")).toBeNull()
      expect(
        screen.getByRole("link", { name: "Request a new reset link" }),
      ).toHaveAttribute("href", "#/recover-password")
      expect(redeem).not.toHaveBeenCalled()
    },
  )

  it("keeps the form when the gateway cannot check the link", async () => {
    vi.mocked(apiFetch).mockImplementation(async (path, init) => {
      if (path === VALIDATE_PATH) throw new ApiError(404, "Not Found")
      return redeem(path, init)
    })
    renderPage("#/reset-password?token=abc123")

    expect(await screen.findByLabelText("New password")).toBeInTheDocument()
  })

  it("sends the token with the new password and says the other sessions ended", async () => {
    redeem.mockResolvedValue(undefined as never)
    const user = userEvent.setup()
    renderPage("#/reset-password?token=abc123")

    await user.type(
      await screen.findByLabelText("New password"),
      "correct-horse",
    )
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "correct-horse",
    )
    await user.click(screen.getByRole("button", { name: "Set password" }))

    expect(
      await screen.findByRole("heading", { name: "Password updated" }),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Any other session this account held has ended/),
    ).toBeInTheDocument()
    const [path, init] = redeem.mock.calls[0] ?? []
    expect(path).toBe("/auth/password/reset/confirm")
    expect(JSON.parse(String(init?.body))).toEqual({
      token: "abc123",
      new_password: "correct-horse",
    })
  })

  it("holds the submit until the confirmation matches", async () => {
    const user = userEvent.setup()
    renderPage("#/reset-password?token=abc123")

    await user.type(
      await screen.findByLabelText("New password"),
      "correct-horse",
    )
    await user.type(screen.getByLabelText("Confirm new password"), "typo")

    expect(
      await screen.findByText("The two passwords do not match."),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Set password" })).toBeDisabled()
  })

  it("shows the gateway's refusal for a spent token, and offers another", async () => {
    redeem.mockRejectedValue(
      new ApiError(
        400,
        "This password reset link is invalid, expired, or already used",
      ),
    )
    const user = userEvent.setup()
    renderPage("#/reset-password?token=stale")

    await user.type(
      await screen.findByLabelText("New password"),
      "correct-horse",
    )
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "correct-horse",
    )
    await user.click(screen.getByRole("button", { name: "Set password" }))

    expect(
      await screen.findByText(
        "This password reset link is invalid, expired, or already used",
      ),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Request a new reset link" }),
    ).toHaveAttribute("href", "#/recover-password")

    // The refusal describes a call that is no longer the one being made.
    await user.type(screen.getByLabelText("New password", { exact: true }), "x")

    expect(
      screen.queryByText(
        "This password reset link is invalid, expired, or already used",
      ),
    ).toBeNull()
  })

  it("sends the terms only when ticked, since only a first password needs them", async () => {
    redeem.mockResolvedValue(undefined as never)
    const user = userEvent.setup()
    renderPage("#/reset-password?token=abc123", "https://example.com/terms")

    await user.type(
      await screen.findByLabelText("New password"),
      "correct-horse",
    )
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "correct-horse",
    )
    // An ordinary reset of an account that accepted them long ago goes through as it is.
    expect(screen.getByRole("button", { name: "Set password" })).toBeEnabled()
    await user.click(screen.getByRole("button", { name: "Set password" }))

    await screen.findByRole("heading", { name: "Password updated" })
    const [, init] = redeem.mock.calls[0] ?? []
    expect(JSON.parse(String(init?.body))).toEqual({
      token: "abc123",
      new_password: "correct-horse",
    })
  })

  it("shows the gateway's terms refusal for a first password, and goes through once they are ticked", async () => {
    redeem
      .mockRejectedValueOnce(
        new ApiError(400, "Accept the terms of service to continue"),
      )
      .mockResolvedValueOnce(undefined as never)
    const user = userEvent.setup()
    renderPage("#/reset-password?token=abc123", "https://example.com/terms")

    await user.type(
      await screen.findByLabelText("New password"),
      "correct-horse",
    )
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "correct-horse",
    )
    await user.click(screen.getByRole("button", { name: "Set password" }))

    expect(
      await screen.findByText("Accept the terms of service to continue"),
    ).toBeInTheDocument()
    await user.click(screen.getByLabelText("I accept the terms of service"))
    expect(
      screen.queryByText("Accept the terms of service to continue"),
    ).toBeNull()
    await user.click(screen.getByRole("button", { name: "Set password" }))

    await screen.findByRole("heading", { name: "Password updated" })
    const [, retry] = redeem.mock.calls[1] ?? []
    expect(JSON.parse(String(retry?.body))).toEqual({
      token: "abc123",
      new_password: "correct-horse",
      terms_accepted: true,
    })
  })

  it("says there is nothing to set when the link carries no token", () => {
    renderPage("#/reset-password")

    expect(screen.getByText(/missing its reset token/)).toBeInTheDocument()
    expect(screen.queryByLabelText("New password")).toBeNull()
    expect(apiFetch).not.toHaveBeenCalled()
  })
})
