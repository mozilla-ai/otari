import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ResendVerificationPage } from "@/features/auth/ResendVerificationPage"
import { apiFetch } from "@/shared/api/client"

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
      <ResendVerificationPage />
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

describe("ResendVerificationPage", () => {
  it("sends a fresh link and lands on the check-email page", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.click(screen.getByRole("button", { name: "Send link" }))

    await vi.waitFor(() => {
      expect(window.location.hash).toBe("#/check-email?type=resend")
    })
    expect(vi.mocked(apiFetch).mock.calls[0]?.[0]).toBe(
      "/v1/auth/resend-verification",
    )
  })
})
