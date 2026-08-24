import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useAuth } from "@/features/auth/AuthContext"
import { Login } from "@/features/auth/Login"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"
import { recordEvent, resetTelemetrySpy } from "@/tests/telemetry"

// The telemetry seam, replaced the way a superset build's alias replaces it.
// The base module records nothing, so the funnel this screen fires is only
// observable through a stand-in.
vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: () => telemetrySpy }
})

// The sign-in screen picks its form from the bootstrap, so every render here
// goes through a DeploymentProvider. Unclaimed (master key) is the default
// because that is what a fresh deployment serves; the password tests override
// `sign_in_methods` the way the gateway does once an operator has claimed it.
function Mounted({
  children,
  signInMethods = ["master_key"],
  mailReady = false,
  maintenanceMode = false,
}: {
  children: React.ReactNode
  signInMethods?: ("master_key" | "password")[]
  mailReady?: boolean
  maintenanceMode?: boolean
}) {
  return (
    <AppProviders>
      <DeploymentProvider
        value={bootstrap({
          sign_in_methods: signInMethods,
          mail_ready: mailReady,
          maintenance_mode: maintenanceMode,
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

  it("says the gateway is under maintenance instead of offering a doomed form", async () => {
    // The freeze refuses both credentials, so a form here could only ever be
    // refused, and a master-key refusal reads as "wrong key" to anyone who does
    // not already know a redeploy is under way.
    render(
      <Mounted maintenanceMode>
        <Harness />
      </Mounted>,
    )

    expect(screen.getByText("Otari is under maintenance")).toBeInTheDocument()
    expect(screen.queryByLabelText("Master key")).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Sign in" }),
    ).not.toBeInTheDocument()
  })

  it("renders the gateway's own 503 wording for a tab that was open before the freeze", async () => {
    // This tab loaded its bootstrap while sign-ins were still open, so it has
    // the form. The refusal has to arrive as a refusal and not as a fault.
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        {
          detail:
            "This gateway is in maintenance mode and is not starting new dashboard sessions right now.",
        },
        503,
      ),
    )
    const user = userEvent.setup()

    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )

    await user.type(screen.getByLabelText("Master key"), "sk-correct")
    await user.click(screen.getByRole("button", { name: "Sign in" }))

    expect(await screen.findByText(/maintenance mode/)).toBeInTheDocument()
    expect(screen.queryByText("SIGNED IN")).not.toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalled()
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

describe("the telemetry the sign-in screen records", () => {
  beforeEach(() => {
    resetTelemetrySpy()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("records a master-key sign-in under the credential it used", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ expires_at: "2026-07-30T00:00:00Z" }),
    )

    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )
    await userEvent.type(screen.getByLabelText("Master key"), "otari-mk-secret")
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }))

    await waitFor(() => {
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
        authentication_method: "master_key",
      })
    })
  })

  it("separates a password sign-in from a master-key one", async () => {
    // The two credentials a deployment can offer are one funnel with two paths,
    // and a claim (otari#649) moves every later sign-in from one to the other.
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ expires_at: "2026-07-30T00:00:00Z" }),
    )

    render(
      <Mounted signInMethods={["password"]}>
        <Harness />
      </Mounted>,
    )
    await userEvent.type(screen.getByLabelText("Email"), "ops@example.com")
    await userEvent.type(screen.getByLabelText("Password"), "hunter22")
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }))

    await waitFor(() => {
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
        authentication_method: "password",
      })
    })
  })

  it("records a refused credential without recording the credential", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Invalid master key." }, 401),
    )

    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )
    await userEvent.type(screen.getByLabelText("Master key"), "wrong-key")
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }))

    await waitFor(() => {
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: "master_key",
        error_code: "credential_rejected",
      })
    })
    // Nothing typed into the form reaches an event: not the key, and not the
    // gateway's own sentence about it.
    for (const [, properties] of recordEvent.mock.calls) {
      expect(JSON.stringify(properties ?? {})).not.toContain("wrong-key")
    }
  })

  it("records a failed request under its status rather than its message", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Gateway is unwell." }, 503),
    )

    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )
    await userEvent.type(screen.getByLabelText("Master key"), "otari-mk-secret")
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }))

    await waitFor(() => {
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: "master_key",
        error_code: "request_failed",
        status: 503,
      })
    })
  })

  it("records a refusal the form made itself, before any request", async () => {
    // This screen deliberately leaves its button enabled on an empty box and
    // validates on submit, which is the moment there is something to record.
    const fetchMock = vi.spyOn(globalThis, "fetch")

    render(
      <Mounted signInMethods={["password"]}>
        <Harness />
      </Mounted>,
    )
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }))

    expect(recordEvent).toHaveBeenCalledWith(
      TELEMETRY_EVENTS.FORM_VALIDATION_FAILED,
      { form_name: "login", errors: ["email_required"] },
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("names a malformed address as a different refusal from a missing one", async () => {
    render(
      <Mounted signInMethods={["password"]}>
        <Harness />
      </Mounted>,
    )
    await userEvent.type(screen.getByLabelText("Email"), "not-an-address")
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }))

    expect(recordEvent).toHaveBeenCalledWith(
      TELEMETRY_EVENTS.FORM_VALIDATION_FAILED,
      { form_name: "login", errors: ["email_invalid_format"] },
    )
  })

  it("names an empty master key on the unclaimed form", async () => {
    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }))

    expect(recordEvent).toHaveBeenCalledWith(
      TELEMETRY_EVENTS.FORM_VALIDATION_FAILED,
      { form_name: "login", errors: ["master_key_required"] },
    )
  })
})
