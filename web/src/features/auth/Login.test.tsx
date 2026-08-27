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
  return { useTelemetry: vi.fn(() => telemetrySpy) }
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
  oauthProviders = [],
}: {
  children: React.ReactNode
  signInMethods?: ("master_key" | "password" | "passkey")[]
  mailReady?: boolean
  maintenanceMode?: boolean
  oauthProviders?: string[]
}) {
  return (
    <AppProviders>
      <DeploymentProvider
        value={bootstrap({
          sign_in_methods: signInMethods,
          mail_ready: mailReady,
          maintenance_mode: maintenanceMode,
          oauth_providers: oauthProviders,
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
    expect(screen.queryByRole("link", { name: /Set your password/ })).toBeNull()
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
      screen.getByRole("link", { name: /Set your password/ }),
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
      screen.getByRole("link", { name: /Set your password/ }),
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
        error_code: "http_401",
      })
    })
    // Nothing typed into the form reaches an event: not the key, and not the
    // gateway's own sentence about it.
    for (const [, properties] of recordEvent.mock.calls) {
      expect(JSON.stringify(properties ?? {})).not.toContain("wrong-key")
    }
  })

  it("records a failed request under its status rather than its message", async () => {
    // 500 rather than 503: a 503 the gateway wrote is maintenance mode and
    // comes back as a refusal, not a throw (see `createSession`). This is the
    // fault path.
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Gateway is unwell." }, 500),
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
        error_code: "http_500",
        status: 500,
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

  it("names a missing password, the one reason with no coverage before", async () => {
    render(
      <Mounted signInMethods={["password"]}>
        <Harness />
      </Mounted>,
    )
    await userEvent.type(screen.getByLabelText("Email"), "ops@example.com")
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }))

    expect(recordEvent).toHaveBeenCalledWith(
      TELEMETRY_EVENTS.FORM_VALIDATION_FAILED,
      { form_name: "login", errors: ["password_required"] },
    )
  })

  it("separates a retired master key from a wrong one", async () => {
    // 401 is a wrong credential and 403 is a master key presented to a
    // deployment that has retired it, a distinction this screen already calls
    // load-bearing. One bucket for both would throw it away.
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Master-key sign-in has been retired." }, 403),
    )

    render(
      <Mounted>
        <Harness />
      </Mounted>,
    )
    await userEvent.type(screen.getByLabelText("Master key"), "otari-mk-old")
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }))

    await waitFor(() => {
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: "master_key",
        error_code: "http_403",
      })
    })
  })

  it("keeps a maintenance freeze apart from a rejected credential", async () => {
    // A 503 the gateway wrote is a deliberate refusal rather than a fault, and
    // it is the one refusal where nothing was wrong with the credential. Under
    // one bucket with a wrong password it would read as a spike in failed
    // sign-ins every time a deployment froze for a redeploy.
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "This gateway is paused for maintenance." }, 503),
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
        error_code: "http_503",
      })
    })
  })
})
describe("Login with a passkey", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    // A successful sign-in is remembered in localStorage, so without this the
    // next test in this block mounts already authenticated and renders no form
    // at all. The block above clears it for the same reason.
    window.localStorage.clear()
  })

  // A browser that can run the ceremony. `supportsPasskeys` is read on render,
  // so this has to be stubbed before one.
  function stubAuthenticator(get: ReturnType<typeof vi.fn>) {
    // A secure context as well as the API: `supportsPasskeys` gates on both, and
    // jsdom reports `isSecureContext` false by default.
    vi.stubGlobal("isSecureContext", true)
    vi.stubGlobal("PublicKeyCredential", function PublicKeyCredential() {})
    vi.stubGlobal("navigator", {
      ...globalThis.navigator,
      credentials: { get },
    })
  }

  function assertion() {
    return {
      id: "Y3JlZA",
      rawId: new Uint8Array([1, 2, 3]).buffer,
      type: "public-key",
      response: {
        clientDataJSON: new Uint8Array([4, 5]).buffer,
        authenticatorData: new Uint8Array([6, 7]).buffer,
        signature: new Uint8Array([8, 9]).buffer,
        userHandle: null,
      },
      getClientExtensionResults: () => ({}),
    }
  }

  function mockCeremony(verify: () => Response) {
    return vi
      .spyOn(globalThis, "fetch")
      .mockImplementation((input: RequestInfo | URL) => {
        const url = String(input)
        if (url === "/v1/auth/webauthn/authenticate/options") {
          return Promise.resolve(jsonResponse({ challenge: "Y2hhbGxlbmdl" }))
        }
        if (url === "/v1/auth/webauthn/authenticate") {
          return Promise.resolve(verify())
        }
        // Anything else the shell asks for (the build poll, say) answers
        // benignly rather than throwing: a throw here surfaces as an unhandled
        // rejection that fails whichever test happens to run next, which is a
        // failure with nothing to do with the one being written.
        return Promise.resolve(jsonResponse({}))
      })
  }

  it("blocks a password sign-in while a ceremony is still open", async () => {
    // The system sheet covers the page but the form behind it is still live and
    // Enter still submits, so without a guard a passkey and a password sign-in
    // race and whichever cookie lands second wins. The same hazard the
    // sign-out guard exists for (#557), arriving from the other direction.
    let releaseCeremony: (value: unknown) => void = () => {}
    const get = vi.fn(
      () =>
        new Promise((resolve) => {
          releaseCeremony = resolve
        }),
    )
    stubAuthenticator(get)
    const fetchMock = mockCeremony(() =>
      jsonResponse({ expires_at: "2026-09-01T00:00:00Z" }),
    )
    const user = userEvent.setup()

    render(
      <Mounted signInMethods={["passkey", "password"]}>
        <Harness />
      </Mounted>,
    )
    await user.click(screen.getByRole("button", { name: "Use a passkey" }))
    await waitFor(() => expect(get).toHaveBeenCalled())

    // The ceremony is open. Fill the form and submit it anyway.
    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("Password"), "a-real-password")
    expect(screen.getByRole("button", { name: /Sign in/ })).toBeDisabled()
    await user.keyboard("{Enter}")

    expect(
      fetchMock.mock.calls.some(([url]) => String(url) === "/v1/auth/session"),
    ).toBe(false)

    releaseCeremony(assertion())
  })

  it("is not offered when the gateway does not publish the method", () => {
    stubAuthenticator(vi.fn())
    render(
      <Mounted signInMethods={["password"]}>
        <Harness />
      </Mounted>,
    )
    expect(
      screen.queryByRole("button", { name: "Use a passkey" }),
    ).not.toBeInTheDocument()
  })

  it("is not offered in a browser that cannot run the ceremony", () => {
    vi.stubGlobal("PublicKeyCredential", undefined)
    render(
      <Mounted signInMethods={["password", "passkey"]}>
        <Harness />
      </Mounted>,
    )
    // Published by the gateway, but the button would be a dead end here.
    expect(
      screen.queryByRole("button", { name: "Use a passkey" }),
    ).not.toBeInTheDocument()
    // The form it sits beside is unaffected.
    expect(screen.getByLabelText("Email")).toBeInTheDocument()
  })

  it("signs in and leaves the password form in place beside it", async () => {
    const get = vi.fn().mockResolvedValue(assertion())
    stubAuthenticator(get)
    mockCeremony(() =>
      jsonResponse({
        expires_at: "2026-08-25T10:00:00Z",
        user_id: "11111111-1111-1111-1111-111111111111",
        active_organization_id: "22222222-2222-2222-2222-222222222222",
      }),
    )
    const user = userEvent.setup()
    render(
      <Mounted signInMethods={["password", "passkey"]}>
        <Harness />
      </Mounted>,
    )

    // Additive: the passkey never replaces the credential form.
    expect(screen.getByLabelText("Email")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Use a passkey" }))

    expect(await screen.findByText("SIGNED IN")).toBeInTheDocument()
    expect(get).toHaveBeenCalled()
  })

  it("reports the gateway's refusal without signing in", async () => {
    stubAuthenticator(vi.fn().mockResolvedValue(assertion()))
    mockCeremony(() =>
      jsonResponse({ detail: "That passkey did not sign you in" }, 401),
    )
    const user = userEvent.setup()
    render(
      <Mounted signInMethods={["password", "passkey"]}>
        <Harness />
      </Mounted>,
    )

    await user.click(screen.getByRole("button", { name: "Use a passkey" }))

    expect(
      await screen.findByText("That passkey did not sign you in"),
    ).toBeInTheDocument()
    expect(screen.queryByText("SIGNED IN")).not.toBeInTheDocument()
  })

  it("says nothing when the prompt is dismissed", async () => {
    const get = vi
      .fn()
      .mockRejectedValue(new DOMException("denied", "NotAllowedError"))
    stubAuthenticator(get)
    mockCeremony(() => jsonResponse({}, 200))
    const user = userEvent.setup()
    render(
      <Mounted signInMethods={["password", "passkey"]}>
        <Harness />
      </Mounted>,
    )

    await user.click(screen.getByRole("button", { name: "Use a passkey" }))

    await waitFor(() => expect(get).toHaveBeenCalled())
    // Pressing Escape is a decision, not a refused credential, so the screen
    // returns to its resting state with nothing said.
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Use a passkey" }),
      ).toBeInTheDocument(),
    )
    expect(screen.queryByText(/did not sign you in/)).not.toBeInTheDocument()
  })

  // The sign-in funnel landed while this branch was in flight, so the passkey
  // path has to report into it too or a whole authentication method is missing
  // from the one place sign-in success is measurable.
  describe("the telemetry it records", () => {
    beforeEach(() => {
      resetTelemetrySpy()
    })

    it("names the method on a success", async () => {
      const get = vi.fn().mockResolvedValue(assertion())
      stubAuthenticator(get)
      mockCeremony(() => jsonResponse({ expires_at: "2026-09-01T00:00:00Z" }))
      const user = userEvent.setup()

      render(
        <Mounted signInMethods={["passkey", "password"]}>
          <Harness />
        </Mounted>,
      )
      await user.click(screen.getByRole("button", { name: "Use a passkey" }))

      expect(await screen.findByText("SIGNED IN")).toBeInTheDocument()
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
        authentication_method: "passkey",
      })
    })

    it("records a refusal with the gateway's status and none of its wording", async () => {
      const get = vi.fn().mockResolvedValue(assertion())
      stubAuthenticator(get)
      mockCeremony(() =>
        jsonResponse({ detail: "That passkey did not sign you in" }, 401),
      )
      const user = userEvent.setup()

      render(
        <Mounted signInMethods={["passkey", "password"]}>
          <Harness />
        </Mounted>,
      )
      await user.click(screen.getByRole("button", { name: "Use a passkey" }))

      await waitFor(() =>
        expect(recordEvent).toHaveBeenCalledWith(
          TELEMETRY_EVENTS.LOGIN_FAILED,
          {
            authentication_method: "passkey",
            error_code: "http_401",
          },
        ),
      )
    })

    it("counts a dismissed prompt as its own outcome", async () => {
      // Neither silence nor a generic failure: it is the most common way this
      // button ends, and both alternatives misreport it.
      const get = vi
        .fn()
        .mockRejectedValue(new DOMException("dismissed", "NotAllowedError"))
      stubAuthenticator(get)
      mockCeremony(() => jsonResponse({}))
      const user = userEvent.setup()

      render(
        <Mounted signInMethods={["passkey", "password"]}>
          <Harness />
        </Mounted>,
      )
      await user.click(screen.getByRole("button", { name: "Use a passkey" }))

      await waitFor(() =>
        expect(recordEvent).toHaveBeenCalledWith(
          TELEMETRY_EVENTS.LOGIN_FAILED,
          {
            authentication_method: "passkey",
            error_code: "passkey_cancelled",
          },
        ),
      )
    })
  })

  describe("OAuth sign-in", () => {
    // `window.location.assign` is not implemented in jsdom, and the whole point
    // of the success path is that it leaves the page, so it is stubbed and
    // asserted on rather than allowed to run.
    function stubNavigation() {
      const assign = vi.fn()
      Object.defineProperty(window, "location", {
        configurable: true,
        value: { ...window.location, assign, hash: "" },
      })
      return assign
    }

    afterEach(() => {
      window.sessionStorage.clear()
    })

    it("offers one button per provider the gateway publishes, and none otherwise", () => {
      const { rerender } = render(
        <Mounted signInMethods={["password"]}>
          <Harness />
        </Mounted>,
      )
      // A deployment that registered no OAuth client carries no affordance at
      // all: absent rather than disabled.
      expect(
        screen.queryByRole("button", { name: /Sign in with/ }),
      ).not.toBeInTheDocument()

      rerender(
        <Mounted signInMethods={["password"]} oauthProviders={["google"]}>
          <Harness />
        </Mounted>,
      )
      expect(
        screen.getByRole("button", { name: "Sign in with Google" }),
      ).toBeInTheDocument()
      // GitHub is configured on some deployment, just not this one.
      expect(
        screen.queryByRole("button", { name: "Sign in with GitHub" }),
      ).not.toBeInTheDocument()
    })

    it("skips a provider this dashboard has no name for", () => {
      // The gateway's provider vocabulary is open, so a bootstrap may carry a
      // connection an overlay bound. Rendering it would produce a button with
      // no label rather than a sign-in.
      render(
        <Mounted
          signInMethods={["password"]}
          oauthProviders={["google", "acme-oidc"]}
        >
          <Harness />
        </Mounted>,
      )
      expect(
        screen.getAllByRole("button", { name: /Sign in with/ }),
      ).toHaveLength(1)
    })

    it("stores the state the gateway minted, then leaves for the provider", async () => {
      const assign = stubNavigation()
      const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse({
          authorization_url: "https://accounts.google.com/o/oauth2/v2/auth?x=1",
          state: "the-state",
        }),
      )
      const user = userEvent.setup()

      render(
        <Mounted signInMethods={["password"]} oauthProviders={["google"]}>
          <Harness />
        </Mounted>,
      )
      await user.click(
        screen.getByRole("button", { name: "Sign in with Google" }),
      )

      await waitFor(() => expect(assign).toHaveBeenCalled())
      expect(fetchMock.mock.calls[0]?.[0]).toBe(
        "/v1/auth/oauth/google/authorize",
      )
      // Stored *before* the navigation, or the callback would have nothing to
      // compare the returned state against.
      expect(window.sessionStorage.getItem("otari.oauth.state")).toBe(
        "the-state",
      )
      expect(assign).toHaveBeenCalledWith(
        "https://accounts.google.com/o/oauth2/v2/auth?x=1",
      )
      // Nothing has succeeded yet: the callback records the outcome.
      expect(recordEvent).not.toHaveBeenCalledWith(
        TELEMETRY_EVENTS.LOGIN_SUCCESS,
        expect.anything(),
      )
    })

    it("does not let a password submit while an OAuth redirect is still in flight", async () => {
      // The same hazard the passkey guard exists for (#557, from the other
      // direction), and worse here: an OAuth attempt ends by leaving the page,
      // so a password accepted meanwhile mints a session the provider's
      // callback then replaces with a session for whichever identity that
      // account resolves to.
      //
      // This pins the pair, not either half. The disabled submit button is what
      // actually stops the Enter below, since a browser will not submit a form
      // whose submit control is disabled; `submit()`'s own `pendingProvider`
      // check is belt-and-braces for a submit that does not go through the
      // button. Reverting one still passes here, which is worth knowing before
      // reading a green run as proof of the guard alone.
      const assign = stubNavigation()
      let releaseAuthorize: (value: Response) => void = () => {}
      const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
        () =>
          new Promise<Response>((resolve) => {
            releaseAuthorize = resolve
          }),
      )
      const user = userEvent.setup()

      render(
        <Mounted signInMethods={["password"]} oauthProviders={["google"]}>
          <Harness />
        </Mounted>,
      )
      await user.click(
        screen.getByRole("button", { name: "Sign in with Google" }),
      )
      await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

      // The redirect has not happened yet. Fill the form and submit it anyway.
      await user.type(screen.getByLabelText("Email"), "operator@example.com")
      await user.type(screen.getByLabelText("Password"), "a-real-password")
      expect(screen.getByRole("button", { name: /Sign in$/ })).toBeDisabled()
      await user.keyboard("{Enter}")

      expect(
        fetchMock.mock.calls.some(
          ([url]) => String(url) === "/v1/auth/session",
        ),
      ).toBe(false)
      expect(assign).not.toHaveBeenCalled()

      releaseAuthorize(
        jsonResponse({ authorization_url: "https://example.test", state: "s" }),
      )
    })

    it("renders the gateway's refusal beside the form, and does not navigate", async () => {
      const assign = stubNavigation()
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse({ detail: "Google sign-in is not configured." }, 503),
      )
      const user = userEvent.setup()

      render(
        <Mounted signInMethods={["password"]} oauthProviders={["google"]}>
          <Harness />
        </Mounted>,
      )
      await user.click(
        screen.getByRole("button", { name: "Sign in with Google" }),
      )

      expect(
        await screen.findByText("Google sign-in is not configured."),
      ).toBeInTheDocument()
      expect(assign).not.toHaveBeenCalled()
      expect(window.sessionStorage.getItem("otari.oauth.state")).toBeNull()
      await waitFor(() =>
        expect(recordEvent).toHaveBeenCalledWith(
          TELEMETRY_EVENTS.LOGIN_FAILED,
          {
            authentication_method: "google",
            error_code: "http_503",
          },
        ),
      )
    })
  })
})
