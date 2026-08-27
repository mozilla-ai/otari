import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AuthProvider } from "@/features/auth/AuthContext"
import { AcceptInvitationPage } from "@/features/invitations/AcceptInvitationPage"
import { ApiError, apiFetch } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"

// Mocks the network boundary (apiFetch), not the hooks, per
// .github/instructions/frontend-standards.instructions.md: the hooks
// (useValidateInvitation, useAcceptInvitation) and TanStack Query stay real,
// so a loading/error state comes from the real hook logic, not a stub of it.
vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

// The marker AuthProvider seeds `isAuthenticated` from. Set rather than mocked
// the hook, so the signed-in case runs the same context the app runs.
const SESSION_MARKER = "otari.dashboard.hasSession"

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

// Mail-ready by default, which is the deployment that emailed the link this
// page is answering; the fixture's own default is a gateway with no transport,
// and the test about that case names it.
function renderPage(hash: string, { mailReady = true } = {}) {
  window.location.hash = hash
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return {
    client,
    ...render(
      <QueryClientProvider client={client}>
        <AuthProvider>
          <DeploymentProvider value={bootstrap({ mail_ready: mailReady })}>
            <AcceptInvitationPage />
          </DeploymentProvider>
        </AuthProvider>
      </QueryClientProvider>,
    ),
  }
}

afterEach(() => {
  vi.restoreAllMocks()
  window.location.hash = ""
  window.localStorage.clear()
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

  it("hands off to signup with the invited address prefilled", async () => {
    // The whole of otari#835: an invited identity is password-less on the
    // roster until it is claimed, so the claim form is the next step and the
    // address it is bound to is one the invitee should not have to retype.
    mockApi({})
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123")

    await user.click(
      await screen.findByRole("button", { name: "Accept invitation" }),
    )

    expect(
      await screen.findByText(/set your password to sign in/i),
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Set your password" }))

    expect(window.location.hash).toBe("#/signup?email=ada%40example.com")
  })

  it("also offers sign-in, for an address that already has a password", async () => {
    // Offered rather than decided: whether the invited identity is already
    // claimed is the enumeration answer POST /v1/auth/signup withholds, so the
    // page may not ask, and both endings have to be reachable from here.
    mockApi({})
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123")

    await user.click(
      await screen.findByRole("button", { name: "Accept invitation" }),
    )

    expect(
      await screen.findByRole("link", {
        name: "Already have a password? Sign in",
      }),
    ).toHaveAttribute("href", "#/")
  })

  it("sends an already signed-in visitor to the dashboard instead", async () => {
    window.localStorage.setItem(SESSION_MARKER, "1")
    mockApi({})
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123")

    await user.click(
      await screen.findByRole("button", { name: "Accept invitation" }),
    )

    expect(await screen.findByText(/already signed in/i)).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Set your password" }),
    ).toBeNull()

    await user.click(
      screen.getByRole("button", { name: "Go to the dashboard" }),
    )
    expect(window.location.hash).toBe("#/")
  })

  it("says who to ask when the deployment cannot send mail", async () => {
    // An invitation reaches such a gateway by hand: the invite response carries
    // the accept link when it could not be emailed. Signup cannot run there
    // (create_user_for_signup refuses before writing), so offering it would be
    // a button whose only outcome is a 503.
    mockApi({})
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123", { mailReady: false })

    await user.click(
      await screen.findByRole("button", { name: "Accept invitation" }),
    )

    expect(
      await screen.findByText(/not configured to send mail/i),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Set your password" }),
    ).toBeNull()
    expect(
      screen.getByRole("button", { name: "Go to sign in" }),
    ).toBeInTheDocument()
  })

  it("keeps the next step when a refetch finds the token spent", async () => {
    // The preview refuses a token that has been accepted, and a reconnect is
    // enough to ask again. Answering with "already used" over a membership this
    // visitor just gained would take the handoff away from them.
    let accepted = false
    vi.mocked(apiFetch).mockImplementation(async (path) => {
      const url = String(path)
      if (url === "/v1/invitations/validate") {
        if (accepted) {
          throw new ApiError(
            400,
            "This invitation has already been used or is no longer valid",
          )
        }
        return {
          email: "ada@example.com",
          organization_name: "Acme",
          role: "admin",
          expires_at: "2026-01-08T00:00:00+00:00",
        } as never
      }
      accepted = true
      return { organization_name: "Acme", role: "admin" } as never
    })
    const user = userEvent.setup()
    const { client } = renderPage("#/accept-invitation?token=abc123")

    await user.click(
      await screen.findByRole("button", { name: "Accept invitation" }),
    )
    expect(
      await screen.findByRole("button", { name: "Set your password" }),
    ).toBeInTheDocument()

    await act(async () => {
      await client.refetchQueries({ queryKey: ["invitation-preview"] })
    })

    expect(
      screen.getByRole("button", { name: "Set your password" }),
    ).toBeInTheDocument()
    expect(screen.queryByText(/already been used/)).toBeNull()
  })

  it("says there is nothing to accept when the link has no token", () => {
    mockApi({})
    renderPage("#/accept-invitation")

    expect(screen.getByText(/missing its invitation token/)).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Set your password" }),
    ).toBeNull()
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
    // No handoff on this path, which is the failure mode a prefilled claim form
    // would be: an identity that cannot be claimed reached through a page that
    // looks like it should work is a worse dead end than the refusal itself.
    expect(
      screen.queryByRole("button", { name: "Set your password" }),
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
    expect(
      screen.queryByRole("button", { name: "Set your password" }),
    ).toBeNull()
  })
})
