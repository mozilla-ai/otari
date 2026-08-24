import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useAuth } from "@/features/auth/AuthContext"
import { Login } from "@/features/auth/Login"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

// The sign-in screen picks its form from the bootstrap, so every render here
// goes through a DeploymentProvider. Unclaimed (master key) is the default
// because that is what a fresh deployment serves; the password tests override
// `sign_in_methods` the way the gateway does once an operator has claimed it.
function Mounted({
  children,
  signInMethods = ["master_key"],
  mailReady = false,
}: {
  children: React.ReactNode
  signInMethods?: ("master_key" | "password")[]
  mailReady?: boolean
}) {
  return (
    <AppProviders>
      <DeploymentProvider
        value={bootstrap({
          sign_in_methods: signInMethods,
          mail_ready: mailReady,
        })}
      >
        {children}
      </DeploymentProvider>
    </AppProviders>
  )
}

function Harness() {
  const { isAuthenticated } = useAuth()
  return isAuthenticated ? <div>SIGNED IN</div> : <Login />
}

function SignOutThenLoginHarness() {
  const { isAuthenticated, logout } = useAuth()
  return isAuthenticated ? (
    <button type="button" onClick={logout}>
      Sign out
    </button>
  ) : (
    <Login />
  )
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

describe("Login", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("signs in by exchanging the master key for a session, never storing the key", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ expires_at: "2026-07-30T00:00:00Z" }))
    const user = userEvent.setup()

    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )

    await user.type(screen.getByLabelText("Master key"), "sk-correct")
    await user.click(screen.getByRole("button", { name: "Sign in" }))

    expect(await screen.findByText("SIGNED IN")).toBeInTheDocument()

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("/v1/auth/session")
    expect(init?.method).toBe("POST")
    expect(init?.body).toBe(JSON.stringify({ master_key: "sk-correct" }))
    // The raw key must not land in any JS-readable storage.
    expect(window.localStorage.getItem("otari.dashboard.hasSession")).toBe("1")
    expect(Object.values({ ...window.localStorage })).not.toContain(
      "sk-correct",
    )
    expect(Object.values({ ...window.sessionStorage })).not.toContain(
      "sk-correct",
    )
  })

  it("offers no signup or recovery link on a gateway that cannot send mail", () => {
    render(
      <Mounted signInMethods={["password"]}>
        <Harness />
      </Mounted>,
    )

    // Hidden rather than offered and then refused with a 503: all three flows
    // begin by sending a message.
    expect(
      screen.queryByRole("link", { name: /Claim your account/ }),
    ).toBeNull()
    expect(
      screen.queryByRole("link", { name: /Forgot your password/ }),
    ).toBeNull()
    expect(screen.queryByRole("link", { name: /verification link/ })).toBeNull()
  })

  it("links to signup, recovery and a fresh verification link once mail works", () => {
    render(
      <Mounted signInMethods={["password"]} mailReady>
        <Harness />
      </Mounted>,
    )

    expect(
      screen.getByRole("link", { name: /Claim your account/ }),
    ).toHaveAttribute("href", "#/signup")
    expect(
      screen.getByRole("link", { name: /Forgot your password/ }),
    ).toHaveAttribute("href", "#/recover-password")
    expect(
      screen.getByRole("link", { name: /verification link/ }),
    ).toHaveAttribute("href", "#/resend-verification")
  })

  it("hides recovery on an unclaimed deployment, where no password exists to reset", () => {
    render(
      <Mounted mailReady>
        <Harness />
      </Mounted>,
    )

    expect(
      screen.queryByRole("link", { name: /Forgot your password/ }),
    ).toBeNull()
    expect(screen.queryByRole("link", { name: /verification link/ })).toBeNull()
    // Signup still stands: a member an admin added by address claims it here.
    expect(
      screen.getByRole("link", { name: /Claim your account/ }),
    ).toBeInTheDocument()
  })

  it("links to the auth-free welcome page", () => {
    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )

    const link = screen.getByRole("link", { name: /welcome/i })
    expect(link).toHaveAttribute("href", "/welcome")
  })

  it("shows an error and stays on the form when the key is rejected", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Invalid master key" }, 401),
    )
    const user = userEvent.setup()

    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )

    await user.type(screen.getByLabelText("Master key"), "sk-wrong")
    await user.click(screen.getByRole("button", { name: "Sign in" }))

    // The gateway's own detail, verbatim, rather than a string this screen
    // guessed: it is the only party that knows which refusal happened.
    expect(await screen.findByText("Invalid master key")).toBeInTheDocument()
    expect(screen.queryByText("SIGNED IN")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Sign in" })).toBeInTheDocument()
  })

  it("asks for email and password once the deployment has been claimed", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ expires_at: "2026-07-30T00:00:00Z" }))
    const user = userEvent.setup()

    render(
      <Mounted signInMethods={["password"]}>
        <Harness />
      </Mounted>,
    )

    // The master-key box is gone, per otari-ai#1716: presenting it here would
    // ask for the one credential this deployment's sign-in no longer takes.
    expect(screen.queryByLabelText("Master key")).not.toBeInTheDocument()

    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("Password"), "a-real-password")
    await user.click(screen.getByRole("button", { name: "Sign in" }))

    expect(await screen.findByText("SIGNED IN")).toBeInTheDocument()

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("/v1/auth/session")
    expect(init?.body).toBe(
      JSON.stringify({
        email: "operator@example.com",
        password: "a-real-password",
      }),
    )
    // Neither half of the credential may land in JS-readable storage.
    const stored = [
      ...Object.values({ ...window.localStorage }),
      ...Object.values({ ...window.sessionStorage }),
    ]
    expect(stored).not.toContain("a-real-password")
    expect(stored).not.toContain("operator@example.com")
  })

  it("names the empty box on submit rather than posting a half credential", async () => {
    // The button used to be disabled until both halves were typed, which made
    // white on the brand tint at 1.95:1 the resting state of the whole screen.
    // It is enabled from the start now, so the guard has to hold here: an empty
    // box is refused locally, and the operator is told which one to fill.
    const fetchMock = vi.spyOn(globalThis, "fetch")
    const user = userEvent.setup()

    render(
      <Mounted signInMethods={["password"]}>
        <Harness />
      </Mounted>,
    )

    const submitButton = screen.getByRole("button", { name: "Sign in" })
    expect(submitButton).toBeEnabled()

    await user.click(submitButton)
    expect(await screen.findByText("Enter your email.")).toBeInTheDocument()
    expect(screen.getByLabelText("Email")).toHaveAttribute(
      "aria-describedby",
      "login-email-error",
    )
    expect(screen.getByRole("alert")).toHaveAttribute("id", "login-email-error")
    expect(fetchMock).not.toHaveBeenCalled()

    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.click(screen.getByRole("button", { name: "Sign in" }))
    expect(await screen.findByText("Enter your password.")).toBeInTheDocument()
    expect(screen.getByLabelText("Password")).toHaveAttribute(
      "aria-describedby",
      "login-password-error",
    )
    expect(fetchMock).not.toHaveBeenCalled()

    // Only once both halves are there does anything reach the gateway.
    fetchMock.mockResolvedValue(
      jsonResponse({ expires_at: "2026-07-30T00:00:00Z" }),
    )
    await user.type(screen.getByLabelText("Password"), "a-real-password")
    await user.click(screen.getByRole("button", { name: "Sign in" }))

    expect(await screen.findByText("SIGNED IN")).toBeInTheDocument()
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/v1/auth/session")
  })

  it("reports a malformed email locally instead of using browser validation", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
    const user = userEvent.setup()

    render(
      <Mounted signInMethods={["password"]}>
        <Harness />
      </Mounted>,
    )

    await user.type(screen.getByLabelText("Email"), "not-an-email")
    await user.type(screen.getByLabelText("Password"), "a-real-password")
    await user.click(screen.getByRole("button", { name: "Sign in" }))

    expect(
      await screen.findByText("Enter a valid email address."),
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Email")).toHaveAttribute(
      "aria-describedby",
      "login-email-error",
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("refuses an empty master key without posting it", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
    const user = userEvent.setup()

    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )

    // Whitespace is not a credential either: the read trims before it decides.
    await user.type(screen.getByLabelText("Master key"), "   ")
    await user.click(screen.getByRole("button", { name: "Sign in" }))

    expect(
      await screen.findByText("Enter your master key."),
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Master key")).toHaveAttribute(
      "aria-describedby",
      "login-master-key-error",
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("reveals the master key on request so a pasted one can be checked", async () => {
    const user = userEvent.setup()

    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )

    expect(screen.getByLabelText("Master key")).toHaveAttribute(
      "type",
      "password",
    )

    await user.click(screen.getByRole("button", { name: "Show master key" }))
    expect(screen.getByLabelText("Master key")).toHaveAttribute("type", "text")

    await user.click(screen.getByRole("button", { name: "Hide master key" }))
    expect(screen.getByLabelText("Master key")).toHaveAttribute(
      "type",
      "password",
    )
  })

  it("keeps the credential inputs unfocused on mount", () => {
    // autoFocus here raised the soft keyboard over half a phone screen on every
    // page load, and made a focus ring the screen's resting state.
    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )

    expect(screen.getByLabelText("Master key")).not.toHaveFocus()
  })

  it("shows the gateway's own wording when a password is rejected", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Incorrect email or password" }, 401),
    )
    const user = userEvent.setup()

    render(
      <Mounted signInMethods={["password"]}>
        <Harness />
      </Mounted>,
    )

    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("Password"), "wrong")
    await user.click(screen.getByRole("button", { name: "Sign in" }))

    expect(
      await screen.findByText("Incorrect email or password"),
    ).toBeInTheDocument()
    expect(screen.queryByText("SIGNED IN")).not.toBeInTheDocument()
  })

  it("surfaces the retirement message when a stale client posts a master key to a claimed deployment", async () => {
    // A 403 is not a wrong credential, and rendering "Invalid master key." over
    // it (what this screen did before it could post a password) tells the
    // operator to retry the one thing that cannot work.
    const retired =
      "Master-key sign-in is retired on this deployment: it has been claimed with a password."
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: retired }, 403),
    )
    const user = userEvent.setup()

    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )

    await user.type(screen.getByLabelText("Master key"), "sk-correct-but-late")
    await user.click(screen.getByRole("button", { name: "Sign in" }))

    expect(await screen.findByText(retired)).toBeInTheDocument()
    expect(screen.queryByText("Invalid master key.")).not.toBeInTheDocument()
  })

  it("offers no credential box when the gateway reports it cannot mint a session", async () => {
    // `/v1/bootstrap` answers [] when it cannot reach its database. A form here
    // could only ever be refused, and on a claimed deployment the fallback form
    // would be the master-key one, whose refusal reads as "wrong key".
    render(
      <Mounted signInMethods={[]}>
        <Harness />
      </Mounted>,
    )

    expect(screen.getByText("Otari sign-in is unavailable")).toBeInTheDocument()
    expect(screen.queryByLabelText("Master key")).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Email")).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Sign in" }),
    ).not.toBeInTheDocument()
  })

  it("refuses a new credential while a prior sign-out's revocation is still in flight (#557)", async () => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1")

    let resolveDelete!: () => void
    const deletePending = new Promise<Response>((resolve) => {
      resolveDelete = () => resolve(new Response(null, { status: 204 }))
    })
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation((_input, init) => {
        if (init?.method === "DELETE") {
          return deletePending
        }
        return Promise.resolve(
          jsonResponse({ expires_at: "2026-07-30T00:00:00Z" }),
        )
      })
    const user = userEvent.setup()

    render(
      <Mounted>
        <SignOutThenLoginHarness />
      </Mounted>,
    )

    await user.click(screen.getByRole("button", { name: "Sign out" }))

    // Local sign-out lands immediately, before the DELETE resolves.
    const keyField = await screen.findByLabelText("Master key")
    await user.type(keyField, "sk-new")

    const submitButton = screen.getByRole("button", {
      name: "Finishing sign-out…",
    })
    expect(submitButton).toBeDisabled()
    await user.click(submitButton)

    // Blocked: no sign-in POST was attempted while the old sign-out was pending.
    expect(
      fetchMock.mock.calls.some(([, init]) => init?.method === "POST"),
    ).toBe(false)

    resolveDelete()
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Sign in" })).toBeEnabled()
    })

    await user.click(screen.getByRole("button", { name: "Sign in" }))

    // Back to authenticated: SignOutThenLoginHarness renders the "Sign out"
    // button again once isAuthenticated flips true.
    expect(
      await screen.findByRole("button", { name: "Sign out" }),
    ).toBeInTheDocument()
    const postCall = fetchMock.mock.calls.find(
      ([, init]) => init?.method === "POST",
    )
    expect(postCall?.[1]?.body).toBe(JSON.stringify({ master_key: "sk-new" }))
  })
})
