import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { DeploymentType } from "@/client"
import { AuthProvider } from "@/features/auth/AuthContext"
import { AcceptInvitationPage } from "@/features/invitations/AcceptInvitationPage"
import { ApiError, apiFetch } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { ThemeProvider } from "@/shared/hooks/useTheme"
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

// An address that can already sign in by default, so the plain accept button
// is what most tests press; the ones about claiming a seat say so.
function mockApi(opts: {
  needsPassword?: boolean
  previewError?: string
  acceptError?: string
}) {
  const accepted: unknown[] = []
  vi.mocked(apiFetch).mockImplementation(async (path, init) => {
    const url = String(path)
    if (url === "/invitations/validate") {
      if (opts.previewError) {
        throw new ApiError(400, opts.previewError)
      }
      return {
        email: "ada@example.com",
        organization_name: "Acme",
        role: "admin",
        expires_at: "2026-01-08T00:00:00+00:00",
        needs_password: opts.needsPassword ?? false,
      } as never
    }
    if (url === "/invitations/accept") {
      if (opts.acceptError) {
        throw new ApiError(400, opts.acceptError)
      }
      const body = JSON.parse(String(init?.body)) as { password?: string }
      accepted.push(body)
      return {
        organization_name: "Acme",
        role: "admin",
        password_set: body.password !== undefined,
      } as never
    }
    throw new Error(`Unexpected apiFetch call: ${url}`)
  })
  return accepted
}

// Mail-ready by default, which is the deployment that emailed the link this
// page is answering; the fixture's own default is a gateway with no transport,
// and the test about that case names it.
function renderPage(
  hash: string,
  {
    mailReady = true,
    oauthProviders = [] as string[],
    deploymentType = "standalone" as DeploymentType,
    termsUrl = null as string | null,
  } = {},
) {
  window.location.hash = hash
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return {
    client,
    ...render(
      <ThemeProvider>
        <QueryClientProvider client={client}>
          <AuthProvider>
            <DeploymentProvider
              value={bootstrap({
                deployment_type: deploymentType,
                mail_ready: mailReady,
                oauth_providers: oauthProviders,
                terms_url: termsUrl,
              })}
            >
              <AcceptInvitationPage />
            </DeploymentProvider>
          </AuthProvider>
        </QueryClientProvider>
      </ThemeProvider>,
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
    const accepted = mockApi({})
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123")

    expect(await screen.findByText("Acme")).toBeInTheDocument()
    expect(screen.getByText("ada@example.com")).toBeInTheDocument()
    expect(screen.getByText("admin")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Accept invitation" }))

    expect(
      await screen.findByText(/You're now a member of/),
    ).toBeInTheDocument()
    expect(accepted).toEqual([{ token: "abc123" }])
  })

  it("sends an address that can already sign in to the sign-in screen", async () => {
    mockApi({})
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123")

    await user.click(
      await screen.findByRole("button", { name: "Accept invitation" }),
    )

    expect(
      await screen.findByText("Sign in to get started."),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText("Password")).toBeNull()
    await user.click(screen.getByRole("button", { name: "Go to sign in" }))
    expect(window.location.hash).toBe("#/")
  })

  it("sets a password for an address that has never signed in, in the same step", async () => {
    const accepted = mockApi({ needsPassword: true })
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123")

    await screen.findByText(/Set a password to sign in as ada@example.com/)
    expect(
      screen.queryByRole("button", { name: "Accept invitation" }),
    ).toBeNull()
    await user.type(screen.getByLabelText("Full name (optional)"), "Ada")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "correct-horse")
    await user.click(
      screen.getByRole("button", { name: "Accept and set password" }),
    )

    expect(
      await screen.findByText(
        /Your password is set. Sign in as ada@example.com/,
      ),
    ).toBeInTheDocument()
    expect(accepted).toEqual([
      { token: "abc123", password: "correct-horse", full_name: "Ada" },
    ])
  })

  it("needs no mail to let an invitee in", async () => {
    // The shared-link case: the deployment sends nothing, the operator handed
    // the link over, and setting the password here is the whole way in.
    mockApi({ needsPassword: true })
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123", { mailReady: false })

    await user.type(await screen.findByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "correct-horse")
    await user.click(
      screen.getByRole("button", { name: "Accept and set password" }),
    )

    expect(await screen.findByText(/Your password is set/)).toBeInTheDocument()
    expect(screen.queryByText(/sends no mail/i)).toBeNull()
  })

  it("holds the submit until the two passwords match", async () => {
    const accepted = mockApi({ needsPassword: true })
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123")

    await user.type(await screen.findByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "correct-hors")

    expect(
      screen.getByRole("button", { name: "Accept and set password" }),
    ).toBeDisabled()
    expect(accepted).toEqual([])
  })

  it("requires the terms where the deployment publishes them", async () => {
    const accepted = mockApi({ needsPassword: true })
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123", {
      termsUrl: "https://example.com/terms",
    })

    await user.type(await screen.findByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "correct-horse")
    const submit = screen.getByRole("button", {
      name: "Accept and set password",
    })
    expect(submit).toBeDisabled()

    await user.click(screen.getByLabelText("I accept the terms of service"))
    await user.click(submit)

    await screen.findByText(/Your password is set/)
    expect(accepted).toEqual([
      {
        token: "abc123",
        password: "correct-horse",
        full_name: null,
        terms_accepted: true,
      },
    ])
  })

  it("lets an invitee accept and sign in with a provider instead of a password", async () => {
    // A provider-verified address resolves a rostered identity that holds no
    // password at all, so where there is a provider the form is not the only
    // way in.
    const accepted = mockApi({ needsPassword: true })
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123", {
      mailReady: false,
      oauthProviders: ["google"],
    })

    await user.click(
      await screen.findByRole("button", {
        name: "Accept and sign in with a provider instead",
      }),
    )

    expect(
      await screen.findByText(/Sign in with one of the providers/),
    ).toBeInTheDocument()
    expect(accepted).toEqual([{ token: "abc123" }])
  })

  it("offers no provider route where the deployment has none", async () => {
    mockApi({ needsPassword: true })
    renderPage("#/accept-invitation?token=abc123")

    await screen.findByLabelText("Password")
    expect(
      screen.queryByRole("button", {
        name: "Accept and sign in with a provider instead",
      }),
    ).toBeNull()
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
    await user.click(
      screen.getByRole("button", { name: "Go to the dashboard" }),
    )
    expect(window.location.hash).toBe("#/")
  })

  it("keeps the next step when a refetch finds the token spent", async () => {
    // The preview refuses a token that has been accepted, and a reconnect is
    // enough to ask again. Answering with "already used" over a membership this
    // visitor just gained would take the handoff away from them.
    let accepted = false
    vi.mocked(apiFetch).mockImplementation(async (path) => {
      const url = String(path)
      if (url === "/invitations/validate") {
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
          needs_password: false,
        } as never
      }
      accepted = true
      return {
        organization_name: "Acme",
        role: "admin",
        password_set: false,
      } as never
    })
    const user = userEvent.setup()
    const { client } = renderPage("#/accept-invitation?token=abc123")

    await user.click(
      await screen.findByRole("button", { name: "Accept invitation" }),
    )
    expect(
      await screen.findByRole("button", { name: "Go to sign in" }),
    ).toBeInTheDocument()

    await act(async () => {
      await client.refetchQueries({ queryKey: ["invitation-preview"] })
    })

    expect(
      screen.getByRole("button", { name: "Go to sign in" }),
    ).toBeInTheDocument()
    expect(screen.queryByText(/already been used/)).toBeNull()
  })

  it("says there is nothing to accept when the link has no token", () => {
    mockApi({})
    renderPage("#/accept-invitation")

    expect(screen.getByText(/missing its invitation token/)).toBeInTheDocument()
    expect(screen.queryByLabelText("Password")).toBeNull()
    // A refusal is not a handoff, but it still owes a way out: Back onto a
    // spent token lands here, and the welcome guide is not one.
    expect(
      screen.getByRole("link", { name: "Back to sign in" }),
    ).toHaveAttribute("href", "#/")
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
    // No claim form on this path: an invitation that cannot be accepted
    // reached through a page that looks like it should work is a worse dead
    // end than the refusal itself.
    expect(screen.queryByLabelText("Password")).toBeNull()
    expect(
      screen.getByRole("link", { name: "Back to sign in" }),
    ).toHaveAttribute("href", "#/")
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
    expect(screen.queryByLabelText("Password")).toBeNull()
  })

  it("offers the welcome guide where the deployment serves it", async () => {
    mockApi({})
    renderPage("#/accept-invitation?token=abc123")

    expect(
      await screen.findByRole("link", { name: /welcome guide/ }),
    ).toHaveAttribute("href", "/welcome")
  })

  // otari.ai serves the bundle from its own edge and `/welcome` from nowhere,
  // so the footer would be a link out of the app to a 404. See
  // `welcomeGuideHref`. Invitations are a hosted flow, which is what makes this
  // the page the broken link was most likely to be clicked from.
  it("offers no welcome guide on a hosted deployment, which serves none", async () => {
    mockApi({})
    renderPage("#/accept-invitation?token=abc123", {
      deploymentType: "hosted",
    })

    expect(await screen.findByText("Acme")).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: /welcome guide/ })).toBeNull()
  })
})
