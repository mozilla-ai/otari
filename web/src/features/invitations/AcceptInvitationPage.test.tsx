import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AcceptInvitationPage } from "@/features/invitations/AcceptInvitationPage"
import { ApiError, apiFetch } from "@/shared/api/client"

// Mocks the network boundary (apiFetch), not the hooks, per
// .github/instructions/frontend-standards.instructions.md: the hooks
// (useValidateInvitation, useAcceptInvitation) and TanStack Query stay real,
// so a loading/error state comes from the real hook logic, not a stub of it.
vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

function mockApi(opts: {
  preview?: unknown
  previewError?: string
  acceptError?: string
}) {
  vi.mocked(apiFetch).mockImplementation(async (path) => {
    const url = String(path)
    if (url === "/v1/invitations/validate") {
      if (opts.previewError) {
        throw new ApiError(400, opts.previewError)
      }
      return (opts.preview ?? {
        email: "ada@example.com",
        organization_name: "Acme",
        role: "admin",
        expires_at: "2026-01-08T00:00:00+00:00",
      }) as never
    }
    if (url === "/v1/invitations/accept") {
      if (opts.acceptError) {
        throw new ApiError(400, opts.acceptError)
      }
      return { organization_name: "Acme", role: "admin" } as never
    }
    throw new Error(`Unexpected apiFetch call: ${url}`)
  })
}

function renderPage(hash: string) {
  window.location.hash = hash
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <AcceptInvitationPage />
    </QueryClientProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
  window.location.hash = ""
})

describe("AcceptInvitationPage", () => {
  it("previews the invitation and accepts it on confirmation", async () => {
    mockApi({})
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123")

    expect(await screen.findByText("Acme")).toBeInTheDocument()
    expect(screen.getByText("ada@example.com")).toBeInTheDocument()
    expect(screen.getByText("admin")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Accept invitation" }))

    expect(
      await screen.findByText(/You're now a member of/),
    ).toBeInTheDocument()
  })

  it("says there is nothing to accept when the link has no token", () => {
    mockApi({})
    renderPage("#/accept-invitation")

    expect(screen.getByText(/missing its invitation token/)).toBeInTheDocument()
  })

  it("shows the server's reason when the token is invalid or expired", async () => {
    mockApi({ previewError: "This invitation has expired" })
    renderPage("#/accept-invitation?token=expired")

    expect(
      await screen.findByText("This invitation has expired"),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Accept invitation" }),
    ).toBeNull()
  })

  it("shows the server's reason when accepting fails", async () => {
    mockApi({
      acceptError:
        "This invitation has already been used or is no longer valid",
    })
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123")

    await user.click(
      await screen.findByRole("button", { name: "Accept invitation" }),
    )

    expect(
      await screen.findByText(
        "This invitation has already been used or is no longer valid",
      ),
    ).toBeInTheDocument()
  })
})
