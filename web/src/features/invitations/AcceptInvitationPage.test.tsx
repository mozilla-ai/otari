import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { DeploymentType } from "@/client"
import { AuthProvider } from "@/features/auth/AuthContext"
import { readSignInAddress } from "@/features/auth/signInAddress"
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
  /** Whether the address can already sign in; by default, unless it needs a password. */
  canSignIn?: boolean
  /** The gateway's `ways_in` for an address that cannot sign in yet. */
  waysIn?: string[]
  verificationPending?: boolean
  previewError?: string
  acceptError?: string
  /** Whose session this browser holds; omitted, the profile read 404s. */
  sessionEmail?: string
  /** A status the profile read fails with instead. */
  profileStatus?: number
}) {
  const accepted: unknown[] = []
  const canSignIn = opts.canSignIn ?? !opts.needsPassword
  const waysIn = opts.waysIn ?? []
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
        can_sign_in: canSignIn,
        ways_in: opts.needsPassword || canSignIn ? [] : waysIn,
      } as never
    }
    if (url === "/auth/profile") {
      if (opts.profileStatus !== undefined) {
        throw new ApiError(opts.profileStatus, "Internal server error")
      }
      if (opts.sessionEmail === undefined) {
        throw new ApiError(404, "Not Found")
      }
      return {
        user_id: "33333333-3333-3333-3333-333333333333",
        email: opts.sessionEmail,
        full_name: null,
        has_password: true,
        claims_deployment: false,
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
        verification_pending: opts.verificationPending ?? false,
        can_sign_in: canSignIn || body.password !== undefined,
        ways_in: canSignIn || body.password !== undefined ? [] : waysIn,
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
  window.sessionStorage.clear()
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

    // Named by address, since the membership goes to the invited address
    // whoever accepted it.
    expect(await screen.findByText(/is now a member of/)).toHaveTextContent(
      "ada@example.com is now a member of Acme, with the admin role.",
    )
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
    expect(
      screen.queryByRole("link", {
        name: "Didn't sign up? Set a password instead",
      }),
    ).toBeNull()
    await user.click(screen.getByRole("button", { name: "Go to sign in" }))
    expect(window.location.hash).toBe("#/")
  })

  it("says to verify first when the address signed up and has not verified yet", async () => {
    // Accepting never verifies: whoever signed up may not hold the mailbox.
    mockApi({ verificationPending: true })
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123")

    await user.click(
      await screen.findByRole("button", { name: "Accept invitation" }),
    )

    expect(
      await screen.findByText(/Verify ada@example.com first/),
    ).toBeInTheDocument()
    expect(screen.queryByText("Sign in to get started.")).toBeNull()
    // The invitee may not be who signed up, and takes the address back here.
    expect(
      screen.getByRole("link", {
        name: "Didn't sign up? Set a password instead",
      }),
    ).toHaveAttribute("href", "#/recover-password")
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
    const accepted = mockApi({ needsPassword: true, waysIn: ["provider"] })
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

    // The providers are offered here, so the sign-in that follows is started
    // from the page that holds what it has to carry.
    expect(
      await screen.findByRole("button", { name: "Sign in with Google" }),
    ).toBeEnabled()
    expect(accepted).toEqual([{ token: "abc123" }])
  })

  describe("where the deployment publishes terms", () => {
    const terms = {
      mailReady: false,
      oauthProviders: ["google"],
      termsUrl: "https://otari.example.com/terms",
    }
    const realLocation = window.location

    afterEach(() => {
      Object.defineProperty(window, "location", {
        configurable: true,
        value: realLocation,
      })
    })

    it("holds the provider way in until the terms are accepted", async () => {
      const accepted = mockApi({ needsPassword: true })
      const user = userEvent.setup()
      renderPage("#/accept-invitation?token=abc123", terms)

      const instead = await screen.findByRole("button", {
        name: "Accept and sign in with a provider instead",
      })
      expect(instead).toBeDisabled()
      await user.click(instead)
      expect(accepted).toEqual([])

      await user.click(
        screen.getByRole("checkbox", { name: "I accept the terms of service" }),
      )
      await user.click(instead)

      await screen.findByRole("button", { name: "Sign in with Google" })
      expect(accepted).toEqual([{ token: "abc123", terms_accepted: true }])
    })

    it("carries the accepted terms into the provider round trip", async () => {
      mockApi({ needsPassword: true })
      const assign = vi.fn()
      Object.defineProperty(window, "location", {
        configurable: true,
        value: { ...window.location, assign, hash: "" },
      })
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(
          JSON.stringify({
            authorization_url: "https://accounts.google.com/o/oauth2/auth?x=1",
            state: "the-state",
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      )
      const user = userEvent.setup()
      renderPage("#/accept-invitation?token=abc123", terms)

      await user.click(
        await screen.findByRole("checkbox", {
          name: "I accept the terms of service",
        }),
      )
      await user.click(
        screen.getByRole("button", {
          name: "Accept and sign in with a provider instead",
        }),
      )
      await user.click(
        await screen.findByRole("button", { name: "Sign in with Google" }),
      )

      await vi.waitFor(() =>
        expect(assign).toHaveBeenCalledWith(
          "https://accounts.google.com/o/oauth2/auth?x=1",
        ),
      )
      expect(window.sessionStorage.getItem("otari.oauth.state")).toBe(
        "the-state",
      )
      expect(window.sessionStorage.getItem("otari.oauth.terms-accepted")).toBe(
        "true",
      )
    })
  })

  it("says which way in exists when the link may not set a password", async () => {
    // Another organization has the address too, so neither link may choose
    // its password, and "Sign in to get started" would be a dead end.
    mockApi({ canSignIn: false, waysIn: ["provider", "password_reset"] })
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123", {
      oauthProviders: ["github"],
    })

    expect(
      await screen.findByText(
        /This link can't set a password for ada@example\.com\. Accept the invitation, then sign in with GitHub, or choose a password with "Forgot your password\?"\./,
      ),
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Accept invitation" }))

    expect(await screen.findByText(/no password yet/)).toBeInTheDocument()
    expect(screen.queryByText("Sign in to get started.")).toBeNull()
    expect(
      screen.getByRole("link", { name: "Choose a password" }),
    ).toHaveAttribute("href", "#/recover-password")
  })

  it("sends a stranded invitee to whoever administers the gateway", async () => {
    mockApi({ canSignIn: false })
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123", { mailReady: false })

    await user.click(
      await screen.findByRole("button", { name: "Accept invitation" }),
    )

    expect(
      await screen.findByText(/ask whoever administers this gateway/),
    ).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: "Choose a password" })).toBeNull()
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
    mockApi({ sessionEmail: "Ada@Example.com" })
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

  it("tells a session for another address who the invitation is for", async () => {
    window.localStorage.setItem(SESSION_MARKER, "1")
    const accepted = mockApi({ sessionEmail: "bob@example.com" })
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123")

    expect(
      await screen.findByText(/This invitation is for ada@example.com/),
    ).toHaveTextContent("you're signed in as bob@example.com")
    expect(
      screen.queryByRole("button", { name: "Accept invitation" }),
    ).toBeNull()

    await user.click(
      screen.getByRole("button", { name: "Sign out and continue" }),
    )

    // Signed out, the page is the one the link was meant for.
    expect(
      await screen.findByRole("button", { name: "Accept invitation" }),
    ).toBeInTheDocument()
    expect(screen.queryByText(/signed in as/)).toBeNull()
    expect(accepted).toEqual([])
  })

  it("treats a session whose address cannot be read as someone else's", async () => {
    // The profile 404s on a gateway older than the route, so whose session
    // this is cannot be known, and "you're already signed in" would be a guess.
    window.localStorage.setItem(SESSION_MARKER, "1")
    const accepted = mockApi({})
    renderPage("#/accept-invitation?token=abc123")

    expect(
      await screen.findByText(/This invitation is for ada@example.com/),
    ).toHaveTextContent("you're signed in as someone else")
    expect(
      screen.getByRole("button", { name: "Sign out and continue" }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Accept invitation" }),
    ).toBeNull()
    expect(accepted).toEqual([])
  })

  it("holds every action when whose session this is cannot be read", async () => {
    window.localStorage.setItem(SESSION_MARKER, "1")
    mockApi({ profileStatus: 500 })
    renderPage("#/accept-invitation?token=abc123")

    expect(await screen.findByText("Internal server error")).toBeInTheDocument()
    expect(screen.getByText("Acme")).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Accept invitation" }),
    ).toBeNull()
    expect(
      screen.queryByRole("button", { name: "Sign out and continue" }),
    ).toBeNull()
    expect(screen.queryByText(/already signed in/i)).toBeNull()
  })

  it("offers no password for somebody else's address to a session", async () => {
    window.localStorage.setItem(SESSION_MARKER, "1")
    mockApi({ sessionEmail: "bob@example.com", needsPassword: true })
    renderPage("#/accept-invitation?token=abc123")

    expect(
      await screen.findByText(/This invitation is for ada@example.com/),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText("Password")).toBeNull()
  })

  it("hands the invited address to the sign-in screen", async () => {
    mockApi({ needsPassword: true })
    const user = userEvent.setup()
    renderPage("#/accept-invitation?token=abc123")

    await user.type(await screen.findByLabelText("Password"), "a-long-password")
    await user.type(
      screen.getByLabelText("Confirm password"),
      "a-long-password",
    )
    await user.click(
      screen.getByRole("button", { name: "Accept and set password" }),
    )
    await user.click(
      await screen.findByRole("button", { name: "Go to sign in" }),
    )

    expect(readSignInAddress()).toBe("ada@example.com")
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
    // spent token lands here.
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

  // The welcome guide is how an operator finds the bootstrap key, which an
  // invitee can do nothing with, so no state of this page links to it.
  it("offers no welcome guide, which is the operator's page", async () => {
    mockApi({})
    renderPage("#/accept-invitation?token=abc123")

    expect(await screen.findByText("Acme")).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: /welcome guide/ })).toBeNull()
  })
})
