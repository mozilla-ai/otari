import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AcceptInvitationPage } from "@/features/invitations/AcceptInvitationPage"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(opts: {
  preview?: unknown
  previewStatus?: number
  acceptStatus?: number
}) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    if (url.includes("/v1/invitations/validate/")) {
      if (opts.previewStatus && opts.previewStatus >= 400) {
        return jsonResponse(
          { detail: "This invitation has expired" },
          opts.previewStatus,
        )
      }
      return jsonResponse(
        opts.preview ?? {
          email: "ada@example.com",
          organization_name: "Acme",
          role: "admin",
          expires_at: "2026-01-08T00:00:00+00:00",
        },
      )
    }
    if (url.includes("/v1/invitations/accept") && method === "POST") {
      if (opts.acceptStatus && opts.acceptStatus >= 400) {
        return jsonResponse(
          {
            detail:
              "This invitation has already been used or is no longer valid",
          },
          opts.acceptStatus,
        )
      }
      return jsonResponse({ organization_name: "Acme", role: "admin" })
    }
    throw new Error(`Unexpected fetch: ${method} ${url}`)
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
    mockApi({ previewStatus: 400 })
    renderPage("#/accept-invitation?token=expired")

    expect(
      await screen.findByText("This invitation has expired"),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Accept invitation" }),
    ).toBeNull()
  })

  it("shows the server's reason when accepting fails", async () => {
    mockApi({ acceptStatus: 400 })
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
