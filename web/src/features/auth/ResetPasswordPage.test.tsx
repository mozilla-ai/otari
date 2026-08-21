import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ResetPasswordPage } from "@/features/auth/ResetPasswordPage"
import { ApiError, apiFetch } from "@/shared/api/client"

vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

function renderPage(hash: string) {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <ResetPasswordPage hash={hash} />
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ResetPasswordPage", () => {
  it("waits for the new password rather than acting on arrival", () => {
    renderPage("#/reset-password?token=abc123")

    expect(apiFetch).not.toHaveBeenCalled()
    expect(screen.getByLabelText("New password")).toBeInTheDocument()
  })

  it("sends the token with the new password and says the other sessions ended", async () => {
    vi.mocked(apiFetch).mockResolvedValue(undefined as never)
    const user = userEvent.setup()
    renderPage("#/reset-password?token=abc123")

    await user.type(screen.getByLabelText("New password"), "correct-horse")
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
    const [path, init] = vi.mocked(apiFetch).mock.calls[0] ?? []
    expect(path).toBe("/v1/auth/password/reset/confirm")
    expect(JSON.parse(String(init?.body))).toEqual({
      token: "abc123",
      new_password: "correct-horse",
    })
  })

  it("holds the submit until the confirmation matches", async () => {
    const user = userEvent.setup()
    renderPage("#/reset-password?token=abc123")

    await user.type(screen.getByLabelText("New password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm new password"), "typo")

    expect(
      await screen.findByText("The two passwords do not match."),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Set password" })).toBeDisabled()
  })

  it("shows the gateway's refusal for a spent token, and offers another", async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      new ApiError(
        400,
        "This password reset link is invalid, expired, or already used",
      ),
    )
    const user = userEvent.setup()
    renderPage("#/reset-password?token=stale")

    await user.type(screen.getByLabelText("New password"), "correct-horse")
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

  it("says there is nothing to set when the link carries no token", () => {
    renderPage("#/reset-password")

    expect(screen.getByText(/missing its reset token/)).toBeInTheDocument()
    expect(screen.queryByLabelText("New password")).toBeNull()
  })
})
