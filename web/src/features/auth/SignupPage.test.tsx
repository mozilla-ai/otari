import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { SignupPage } from "@/features/auth/SignupPage"
import { ApiError, apiFetch } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { ThemeProvider } from "@/shared/hooks/useTheme"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { bootstrap } from "@/tests/fixtures"
import { recordEvent, resetTelemetrySpy } from "@/tests/telemetry"

// The network boundary, not the hooks: the real hooks, their query keys, and
// the mutation state the page branches on all stay live.
vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

// The telemetry seam, replaced the way a superset build's alias replaces it: the
// base module records nothing, so the funnel is only observable through a
// stand-in.
vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: vi.fn(() => telemetrySpy) }
})

// `PublicAuthPage` passes the whole hash down, so the plain page is the one
// reached from the sign-in screen and a `?email=` one is the accept page's
// handoff (otari#835).
// The page reads `open_signup` and `terms_url` off the bootstrap, so every
// render goes through a DeploymentProvider. Closed signup and no published
// terms is the default, matching a deployment that configured neither.
function renderPage(
  hash = "#/signup",
  deployment: {
    openSignup?: boolean
    termsUrl?: string | null
    oauthProviders?: string[]
  } = {},
) {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider
        value={bootstrap({
          open_signup: deployment.openSignup ?? false,
          terms_url: deployment.termsUrl ?? null,
          oauth_providers: deployment.oauthProviders ?? [],
        })}
      >
        <ThemeProvider>
          <SignupPage hash={hash} />
        </ThemeProvider>
      </DeploymentProvider>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  resetTelemetrySpy()
  window.location.hash = ""
})

afterEach(() => {
  vi.restoreAllMocks()
  window.location.hash = ""
})

vi.mock("@/features/auth/overlayPublicAuthFields", () => ({
  PublicAuthFields: ({ page, isBusy }: { page: string; isBusy: boolean }) => (
    <p>{`fields for ${page}, ${isBusy ? "busy" : "idle"}`}</p>
  ),
}))

describe("SignupPage", () => {
  it("renders the edition's own fields ahead of the address", () => {
    renderPage()

    expect(screen.getByText("fields for signup, idle")).toBeInTheDocument()
  })

  it("claims the identity and lands on the check-email page", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    await vi.waitFor(() => {
      expect(window.location.hash).toBe("#/check-email?type=signup")
    })
    const [path, init] = vi.mocked(apiFetch).mock.calls[0] ?? []
    expect(path).toBe("/auth/signup")
    expect(JSON.parse(String(init?.body))).toEqual({
      email: "ada@example.com",
      password: "correct-horse",
    })
  })

  // otari-ai#2100: the same form, reading as registration where the deployment
  // takes an address nobody added.
  it("reads as registration where signup is open", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage("#/signup", { openSignup: true })

    expect(
      screen.getByRole("heading", { name: "Create your account" }),
    ).toBeInTheDocument()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Create account" }))

    await vi.waitFor(() => {
      expect(window.location.hash).toBe("#/check-email?type=signup")
    })
  })

  it("offers no terms checkbox on a deployment that published none", () => {
    renderPage()

    expect(screen.queryByRole("checkbox")).toBeNull()
  })

  it("requires the published terms, and records the acceptance", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage("#/signup", { termsUrl: "https://otari.example.com/terms" })

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    const submit = screen.getByRole("button", { name: "Claim account" })
    expect(submit).toBeDisabled()

    expect(
      screen.getByRole("link", { name: "terms of service" }),
    ).toHaveAttribute("href", "https://otari.example.com/terms")
    await user.click(screen.getByRole("checkbox"))
    await user.click(submit)

    const [, init] = vi.mocked(apiFetch).mock.calls[0] ?? []
    expect(JSON.parse(String(init?.body))).toEqual({
      email: "ada@example.com",
      password: "correct-horse",
      terms_accepted: true,
    })
  })

  it("opens the terms without ticking the box that links to them", async () => {
    // The link sits inside the checkbox's label, which react-aria makes
    // pressable: without the guard on the anchor, reading the terms accepted
    // them (otari-ai#2146).
    const opened = vi.spyOn(window, "open").mockReturnValue(null)
    const user = userEvent.setup()
    renderPage("#/signup", { termsUrl: "https://otari.example.com/terms" })

    await user.click(screen.getByRole("link", { name: "terms of service" }))

    expect(screen.getByRole("checkbox")).not.toBeChecked()
    // The half of the sentence that stayed in the label still toggles it.
    await user.click(screen.getByText("I accept the"))
    expect(screen.getByRole("checkbox")).toBeChecked()
    opened.mockRestore()
  })

  it("shows a password problem in the field's own message line", async () => {
    // One line, not two: the message takes the description's place rather than
    // stacking under it, so the card is the same height whether or not the
    // field is speaking. A card that changes height moves the animated
    // background measured against it (otari-ai#2146).
    const user = userEvent.setup()
    renderPage()
    const description = "At least 8 characters, and at most 72 bytes."
    expect(screen.getByText(description)).toBeInTheDocument()

    await user.type(screen.getByLabelText("Password"), "short")

    expect(
      await screen.findByText("At least 8 characters."),
    ).toBeInTheDocument()
    expect(screen.queryByText(description)).toBeNull()
    expect(screen.getByLabelText("Password")).toHaveAttribute(
      "aria-invalid",
      "true",
    )
  })

  it("refuses a password the gateway would refuse, without asking it", async () => {
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "short")

    expect(
      await screen.findByText("At least 8 characters."),
    ).toBeInTheDocument()
    expect(apiFetch).not.toHaveBeenCalled()
  })

  it("keeps its fill, blocks the press and says it is busy while the claim is out", async () => {
    let release!: () => void
    vi.mocked(apiFetch).mockReturnValue(
      new Promise((resolve) => {
        release = () => resolve({ message: "…" } as never)
      }),
    )
    const user = userEvent.setup()
    const { container } = renderPage()
    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")

    await user.click(screen.getByRole("button", { name: "Claim account" }))

    const pending = await screen.findByRole("button", {
      name: "Claiming account…",
    })
    // Not disabled, which would dim it to the refused treatment.
    expect(pending).not.toBeDisabled()
    expect(pending).toHaveClass("pointer-events-none")
    expect(container.querySelector("form")).toHaveAttribute("aria-busy", "true")
    release()
  })

  it("shows the gateway's own refusal and stays on the form", async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      new ApiError(503, "Outgoing mail is not configured on this deployment."),
    )
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    expect(
      await screen.findByText(
        "Outgoing mail is not configured on this deployment.",
      ),
    ).toBeInTheDocument()
    // No navigation: the page only leaves for #/check-email on a success.
    expect(window.location.hash).toBe("")
  })
})

describe("the telemetry the signup page records", () => {
  it("records the attempt and then the claim it produced", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.SIGNUP_STARTED, {
      authentication_method: "password",
    })
    await vi.waitFor(() => {
      expect(recordEvent).toHaveBeenCalledWith(
        TELEMETRY_EVENTS.SIGNUP_SUCCESS,
        // Always verification-bound: this endpoint is enumeration-safe, so the
        // page reads nothing back and neither does this.
        { authentication_method: "password", requires_verification: true },
      )
    })
  })

  it("records a refused claim under its status, not its message", async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      new ApiError(503, "Outgoing mail is not configured on this deployment."),
    )
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    await vi.waitFor(() => {
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.SIGNUP_FAILED, {
        authentication_method: "password",
        status: 503,
      })
    })
  })

  it("records nothing for a form its own button will not submit", async () => {
    // This page validates by disabling the submit rather than by refusing one,
    // so there is no moment at which a validation failure could be recorded and
    // none is manufactured. `FORM_VALIDATION_FAILED` comes from the sign-in
    // screen, which keeps its button live on purpose.
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "short")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    expect(recordEvent).not.toHaveBeenCalled()
  })

  it("prefills the invited address, read-only, and claims that one", async () => {
    // The address is the invitation's, not the visitor's to choose: another one
    // has nothing to claim, and signup answers the same enumeration-safe
    // sentence either way, so an editable field would fail silently.
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage("#/signup?email=ada%40example.com")

    const emailField = screen.getByLabelText("Email")
    expect(emailField).toHaveValue("ada@example.com")
    expect(emailField).toHaveAttribute("readonly")

    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    await vi.waitFor(() => {
      expect(window.location.hash).toBe("#/check-email?type=signup")
    })
    const [, init] = vi.mocked(apiFetch).mock.calls[0] ?? []
    expect(JSON.parse(String(init?.body)).email).toBe("ada@example.com")
  })

  it("offers the plain form to anyone who needs another address", () => {
    // The way out of the read-only field, so a prefill that is wrong for this
    // visitor is not a dead end of its own.
    renderPage("#/signup?email=ada%40example.com")

    expect(
      screen.getByRole("link", { name: "Claim a different address instead" }),
    ).toHaveAttribute("href", "#/signup")
  })

  it("asks for the address when the link carries none", () => {
    renderPage()

    const emailField = screen.getByLabelText("Email")
    expect(emailField).toHaveValue("")
    expect(emailField).not.toHaveAttribute("readonly")
    expect(
      screen.queryByRole("link", { name: "Claim a different address instead" }),
    ).toBeNull()
  })
})

describe("OAuth first", () => {
  function stubNavigation() {
    const assign = vi.fn()
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, assign, hash: "" },
    })
    return assign
  }

  function jsonResponse(body: unknown, status = 200) {
    return new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    })
  }

  const open = { openSignup: true, oauthProviders: ["github", "google"] }

  afterEach(() => {
    window.sessionStorage.clear()
  })

  it("folds the address form behind a row while a provider is offered", () => {
    renderPage("#/signup", open)

    expect(
      screen.getByRole("button", { name: "Sign up with GitHub" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Sign up with Google" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Sign up with email" }),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText("Email")).toBeNull()
    expect(screen.queryByLabelText("Password")).toBeNull()
  })

  it("opens the form in place, keeps the providers, and focuses the address", async () => {
    const user = userEvent.setup()
    renderPage("#/signup", open)

    await user.click(screen.getByRole("button", { name: "Sign up with email" }))

    expect(screen.getByLabelText("Email")).toHaveFocus()
    expect(screen.getByLabelText("Password")).toBeInTheDocument()
    // The row that opened it is gone and the providers are still there, now as
    // the two-up pair labelled with the provider alone.
    expect(
      screen.queryByRole("button", { name: "Sign up with email" }),
    ).toBeNull()
    expect(
      screen.getByRole("button", { name: "Sign up with GitHub" }),
    ).toHaveTextContent(/^GitHub$/)
  })

  it("offers neither the folded row nor a provider on a closed deployment", () => {
    // OAuth on a closed deployment only admits an address already on the
    // roster, so it is not a way to sign up and the page is the plain form.
    renderPage("#/signup", { oauthProviders: ["github", "google"] })

    expect(screen.queryByRole("button", { name: /Sign up with/ })).toBeNull()
    expect(screen.getByLabelText("Email")).toBeInTheDocument()
  })

  it("shows the form straight away when no provider is configured", () => {
    renderPage("#/signup", { openSignup: true })

    expect(screen.queryByRole("button", { name: /Sign up with/ })).toBeNull()
    expect(screen.getByLabelText("Email")).toBeInTheDocument()
    expect(screen.getByLabelText("Email")).not.toHaveFocus()
  })

  it("opens the form for a link that names an invited address", () => {
    renderPage("#/signup?email=ada%40example.com", open)

    expect(screen.getByLabelText("Email")).toHaveValue("ada@example.com")
    expect(screen.getByLabelText("Email")).not.toHaveFocus()
  })

  it("stores the state the gateway minted, then leaves for the provider", async () => {
    const assign = stubNavigation()
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        authorization_url: "https://github.com/login/oauth/authorize?x=1",
        state: "the-state",
      }),
    )
    const user = userEvent.setup()
    renderPage("#/signup", open)

    await user.click(
      screen.getByRole("button", { name: "Sign up with GitHub" }),
    )

    await vi.waitFor(() => expect(assign).toHaveBeenCalled())
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/v1/auth/oauth/github/authorize",
    )
    expect(window.sessionStorage.getItem("otari.oauth.state")).toBe("the-state")
    expect(assign).toHaveBeenCalledWith(
      "https://github.com/login/oauth/authorize?x=1",
    )
    expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.SIGNUP_STARTED, {
      authentication_method: "github",
    })
    // Nothing has succeeded yet: the callback page records the outcome.
    expect(recordEvent).not.toHaveBeenCalledWith(
      TELEMETRY_EVENTS.SIGNUP_SUCCESS,
      expect.anything(),
    )
  })

  it("says so, and stays, when the gateway cannot start the sign-up", async () => {
    const assign = stubNavigation()
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Google sign-in is not configured." }, 503),
    )
    const user = userEvent.setup()
    renderPage("#/signup", open)

    await user.click(
      screen.getByRole("button", { name: "Sign up with Google" }),
    )

    expect(await screen.findByRole("alert")).toBeInTheDocument()
    expect(assign).not.toHaveBeenCalled()
    expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.SIGNUP_FAILED, {
      authentication_method: "google",
      status: 503,
    })
    expect(
      screen.getByRole("button", { name: "Sign up with Google" }),
    ).toBeEnabled()
  })
})

describe("the password reveal", () => {
  it("shows what was typed on request, and hides it again", async () => {
    const user = userEvent.setup()
    renderPage()
    const field = screen.getByLabelText("Password")
    await user.type(field, "correct-horse")
    expect(field).toHaveAttribute("type", "password")

    // Guards, not checks: jsdom computes no cascade. The toggle is 32px at
    // every width, exempt from the phone-width 44px floor that would otherwise
    // widen the button and push its 44px bleed into the label above; the
    // measured result is a 32px box with a 6px bleed on every side.
    const toggle = screen.getByRole("button", { name: "Show password" })
    expect(toggle).toHaveClass("min-h-8!", "min-w-8!", "before:-inset-[7px]")

    await user.click(toggle)
    expect(field).toHaveAttribute("type", "text")
    expect(field).toHaveValue("correct-horse")

    await user.click(screen.getByRole("button", { name: "Hide password" }))
    expect(field).toHaveAttribute("type", "password")
  })
})
