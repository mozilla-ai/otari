import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { RecoverPasswordPage } from "@/features/auth/RecoverPasswordPage"
import { ApiError, apiFetch } from "@/shared/api/client"

// The network boundary, not the hooks: the real hooks, their query keys, and
// the mutation state the page branches on all stay live.
vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <RecoverPasswordPage />
    </QueryClientProvider>,
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

describe("RecoverPasswordPage", () => {
  it("confirms in the conditional rather than reporting on the address", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.click(screen.getByRole("button", { name: "Send reset link" }))

    expect(
      await screen.findByText(/If that address has a password on this gateway/),
    ).toBeInTheDocument()
    expect(vi.mocked(apiFetch).mock.calls[0]?.[0]).toBe(
      "/v1/auth/password/reset",
    )
  })

  it("clears a stale refusal as soon as the address is retyped", async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      new ApiError(429, "Too many attempts. Try again in a minute."),
    )
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.click(screen.getByRole("button", { name: "Send reset link" }))
    expect(
      await screen.findByText("Too many attempts. Try again in a minute."),
    ).toBeInTheDocument()

    await user.type(screen.getByLabelText("Email"), "x")

    expect(
      screen.queryByText("Too many attempts. Try again in a minute."),
    ).toBeNull()
  })
})
