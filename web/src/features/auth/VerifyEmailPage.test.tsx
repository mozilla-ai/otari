import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { StrictMode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { VerifyEmailPage } from "@/features/auth/VerifyEmailPage"
import { ApiError, apiFetch } from "@/shared/api/client"

vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

function renderPage(hash: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return {
    client,
    ...render(
      <QueryClientProvider client={client}>
        <VerifyEmailPage hash={hash} />
      </QueryClientProvider>,
    ),
  }
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("VerifyEmailPage", () => {
  it("verifies on arrival and names the address that is now confirmed", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ email: "ada@example.com" } as never)

    renderPage("#/verify-email?token=abc123")

    expect(
      await screen.findByRole("heading", { name: "Email verified" }),
    ).toBeInTheDocument()
    expect(screen.getByText(/ada@example.com is confirmed/)).toBeInTheDocument()
    const [path, init] = vi.mocked(apiFetch).mock.calls[0] ?? []
    expect(path).toBe("/v1/auth/verify-email")
    // In the body, not the URL: the token is a bearer credential and a URL is
    // what an access log retains.
    expect(JSON.parse(String(init?.body))).toEqual({ token: "abc123" })
  })

  it("spends a single-use token once, even under StrictMode", async () => {
    // `main.tsx` wraps the app in StrictMode, which in development mounts,
    // unmounts and mounts again. Without the page's own guard that is two
    // POSTs of one single-use token: the first consumes it, the second 400s,
    // and the observer ends on the failure, so a verification that worked
    // renders "Verification failed".
    vi.mocked(apiFetch).mockResolvedValue({ email: "ada@example.com" } as never)
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })

    render(
      <StrictMode>
        <QueryClientProvider client={client}>
          <VerifyEmailPage hash="#/verify-email?token=abc123" />
        </QueryClientProvider>
      </StrictMode>,
    )

    expect(
      await screen.findByRole("heading", { name: "Email verified" }),
    ).toBeInTheDocument()
    expect(apiFetch).toHaveBeenCalledTimes(1)
  })

  it("re-renders without re-verifying", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ email: "ada@example.com" } as never)

    // The same client across both renders. Handing the second one a fresh
    // QueryClient would throw away the cache that is the thing under test.
    const { client, rerender } = renderPage("#/verify-email?token=abc123")
    await screen.findByRole("heading", { name: "Email verified" })
    rerender(
      <QueryClientProvider client={client}>
        <VerifyEmailPage hash="#/verify-email?token=abc123" />
      </QueryClientProvider>,
    )

    expect(apiFetch).toHaveBeenCalledTimes(1)
  })

  it("offers a fresh link when the token is expired or already used", async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      new ApiError(
        400,
        "This verification link is invalid, expired, or already used",
      ),
    )

    renderPage("#/verify-email?token=stale")

    expect(
      await screen.findByText(
        "This verification link is invalid, expired, or already used",
      ),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Send a new verification link" }),
    ).toHaveAttribute("href", "#/resend-verification")
  })

  it.each(["#/verify-email", "#/verify-email?token="])(
    "asks the gateway nothing when %s carries no token",
    (hash) => {
      // The truncated form is the one that used to strand: an empty string is
      // not null, so the missing-token branch was skipped, and it is not a
      // non-empty token either, so the query stayed disabled and the page sat
      // on "Confirming your address…" forever having asked nothing.
      renderPage(hash)

      expect(
        screen.getByText(/missing its verification token/),
      ).toBeInTheDocument()
      expect(apiFetch).not.toHaveBeenCalled()
    },
  )
})
