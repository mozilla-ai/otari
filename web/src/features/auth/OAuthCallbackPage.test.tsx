import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { StrictMode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useAuth } from "@/features/auth/AuthContext"
import { OAuthCallbackPage } from "@/features/auth/OAuthCallbackPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"
import { recordEvent, resetTelemetrySpy } from "@/tests/telemetry"

vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: vi.fn(() => telemetrySpy) }
})

const STATE_KEY = "otari.oauth.state"
const RETURN_KEY = "otari.oauth.return-hash"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

/**
 * The page in isolation, with a signed-in marker to assert against.
 *
 * Deliberately **not** the way `DeploymentRoot` mounts it, and the comment that
 * once claimed otherwise is what let a real bug through: the app matches a
 * public auth path *ahead* of the auth gate, so signing in does not stop this
 * page rendering. This harness has the gate first, which makes `login()` look
 * like a navigation. That is fine for the assertions here, which are about what
 * the page sends and refuses; the navigation itself is pinned against the real
 * `App` tree in `src/app/App.test.tsx` ("sends a completed OAuth sign-in on to
 * the dashboard"), which is the only place the true ordering exists.
 */
function Harness({ hash }: { hash: string }) {
  const { isAuthenticated } = useAuth()
  return isAuthenticated ? (
    <div>SIGNED IN</div>
  ) : (
    <OAuthCallbackPage provider="google" hash={hash} />
  )
}

function renderCallback(hash: string) {
  return render(
    <AppProviders>
      <DeploymentProvider value={bootstrap({ oauth_providers: ["google"] })}>
        <Harness hash={hash} />
      </DeploymentProvider>
    </AppProviders>,
  )
}

beforeEach(() => {
  // The telemetry spy is one module-level `vi.fn()` shared by every test that
  // mocks the seam, so it accumulates across this file's tests unless it is
  // cleared. Without this a test asserting "exactly one outcome" counts the
  // refusals every earlier test recorded.
  resetTelemetrySpy()
  window.sessionStorage.clear()
  // `AuthContext` keeps a "there is a session" marker in localStorage, so a
  // test that signs in would leave every later one already signed in.
  window.localStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
  window.sessionStorage.clear()
  window.localStorage.clear()
  // The success path changes the hash, so a test that signs in would otherwise
  // leave it set for the next one.
  window.location.hash = ""
})

describe("OAuthCallbackPage", () => {
  it("spends the code for a session when the state matches the one this tab stored", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ expires_at: "2026-09-01T00:00:00Z" }))

    renderCallback("#/auth/google/callback?code=the-code&state=the-state")

    expect(await screen.findByText("SIGNED IN")).toBeInTheDocument()
    const [url, init] = fetchMock.mock.calls[0] ?? []
    expect(url).toBe(`${API_ROOT}/auth/oauth/google/callback`)
    // The code and the state, and no redirect URI: that one is the gateway's
    // own. The state goes back so the gateway can check it against the pending
    // authorization it recorded, which is the half this tab cannot do. No terms
    // were accepted before this flow left, so none are claimed.
    expect(JSON.parse(String(init?.body))).toEqual({
      code: "the-code",
      state: "the-state",
      terms_accepted: false,
    })
    await waitFor(() =>
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
        authentication_method: "google",
      }),
    )
  })

  it("clears the stored state, so one value answers exactly one callback", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ expires_at: "2026-09-01T00:00:00Z" }),
    )

    renderCallback("#/auth/google/callback?code=the-code&state=the-state")

    await screen.findByText("SIGNED IN")
    expect(window.sessionStorage.getItem(STATE_KEY)).toBeNull()
  })

  it("returns to the page the sign-in started from", async () => {
    // An emailed link to a page behind the sign-in screen, followed signed out:
    // the sign-in screen stood in for that page, and the provider's round trip
    // must not lose it.
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    window.sessionStorage.setItem(RETURN_KEY, "#/organization/billing")
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ expires_at: "2026-09-01T00:00:00Z" }),
    )

    renderCallback("#/auth/google/callback?code=the-code&state=the-state")

    await screen.findByText("SIGNED IN")
    expect(window.location.hash).toBe("#/organization/billing")
    expect(window.sessionStorage.getItem(RETURN_KEY)).toBeNull()
  })

  it.each([
    ["nothing was remembered", undefined],
    ["the page renders ahead of the router", "#/verify-email?token=t"],
    ["the value is not a hash path", "https://evil.example/#/keys"],
  ])("goes home when %s", async (_, remembered) => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    if (remembered !== undefined) {
      window.sessionStorage.setItem(RETURN_KEY, remembered)
    }
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ expires_at: "2026-09-01T00:00:00Z" }),
    )

    renderCallback("#/auth/google/callback?code=the-code&state=the-state")

    await screen.findByText("SIGNED IN")
    expect(window.location.hash).toBe("#/")
    expect(window.sessionStorage.getItem(RETURN_KEY)).toBeNull()
  })

  it.each([
    [
      "the provider declined",
      "#/auth/google/callback?error=access_denied&state=the-state",
      () => {},
    ],
    [
      "the gateway refused the code",
      "#/auth/google/callback?code=the-code&state=the-state",
      () =>
        vi
          .spyOn(globalThis, "fetch")
          .mockResolvedValue(
            jsonResponse({ detail: "Not registered here." }, 401),
          ),
    ],
  ])("keeps the page for a retry when %s", async (_, hash, stubFetch) => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    window.sessionStorage.setItem(RETURN_KEY, "#/organization/billing")
    stubFetch()

    renderCallback(hash)

    await screen.findByRole("heading", {
      name: "That sign-in did not complete",
    })
    // The sign-in screen stands in for that page, so a retry from there
    // remembers it afresh; storage keeps nothing for a later sign-in.
    expect(
      screen.getByRole("link", { name: "Back to sign in" }),
    ).toHaveAttribute("href", "#/organization/billing")
    expect(window.sessionStorage.getItem(RETURN_KEY)).toBeNull()
    expect(window.location.hash).toBe("")
  })

  it("sends a failed sign-in back home when the page is not one to return to", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    window.sessionStorage.setItem(RETURN_KEY, "#/reset-password?token=t")

    renderCallback("#/auth/google/callback?error=access_denied&state=the-state")

    expect(
      await screen.findByRole("link", { name: "Back to sign in" }),
    ).toHaveAttribute("href", "#/")
  })

  it("refuses a state that does not match, without spending the code", async () => {
    // The CSRF check, and the whole reason the value made the round trip: this
    // redirect did not come from a flow this tab started.
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    const fetchMock = vi.spyOn(globalThis, "fetch")

    renderCallback("#/auth/google/callback?code=the-code&state=somebody-elses")

    expect(
      await screen.findByRole("heading", {
        name: "That sign-in did not complete",
      }),
    ).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
    await waitFor(() =>
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: "google",
        error_code: "invalid_state",
      }),
    )
  })

  it("refuses a callback with no state at all", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")

    renderCallback("#/auth/google/callback?code=the-code")

    await screen.findByRole("heading", {
      name: "That sign-in did not complete",
    })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("says so when the provider reports the person declined, and calls nothing", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    const fetchMock = vi.spyOn(globalThis, "fetch")

    renderCallback(
      "#/auth/google/callback?error=access_denied&error_description=nope&state=the-state",
    )

    expect(
      await screen.findByText(/Google did not complete the sign-in/),
    ).toBeInTheDocument()
    // The provider's own error_description is attacker-influenceable text
    // arriving in a URL, and is deliberately not rendered.
    expect(screen.queryByText(/nope/)).toBeNull()
    expect(fetchMock).not.toHaveBeenCalled()
    await waitFor(() =>
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: "google",
        error_code: "provider_error",
      }),
    )
  })

  it("refuses a callback carrying a state but no code", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    const fetchMock = vi.spyOn(globalThis, "fetch")

    renderCallback("#/auth/google/callback?state=the-state")

    expect(
      await screen.findByText(
        /Google sent this browser back without an authorization code/,
      ),
    ).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("renders the gateway's own refusal, which is what tells a person what to do", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        {
          detail:
            "That Google account is not registered on this gateway. Ask whoever administers it to add your email address, then sign in again.",
        },
        401,
      ),
    )

    renderCallback("#/auth/google/callback?code=the-code&state=the-state")

    expect(
      await screen.findByText(/not registered on this gateway/),
    ).toBeInTheDocument()
    expect(screen.queryByText("SIGNED IN")).toBeNull()
  })

  describe("on a deployment that publishes terms", () => {
    const TERMS_KEY = "otari.oauth.terms-accepted"
    const realLocation = window.location

    function renderWithTerms(hash: string) {
      return render(
        <AppProviders>
          <DeploymentProvider
            value={bootstrap({
              oauth_providers: ["google"],
              terms_url: "https://otari.example.com/terms",
            })}
          >
            <Harness hash={hash} />
          </DeploymentProvider>
        </AppProviders>,
      )
    }

    const termsRefusal = () =>
      jsonResponse(
        {
          detail: "Accept the terms of service to continue",
          code: "terms_not_accepted",
        },
        400,
      )

    afterEach(() => {
      Object.defineProperty(window, "location", {
        configurable: true,
        value: realLocation,
      })
    })

    it("sends the acceptance the page that started the flow took, once", async () => {
      window.sessionStorage.setItem(STATE_KEY, "the-state")
      window.sessionStorage.setItem(TERMS_KEY, "true")
      const fetchMock = vi
        .spyOn(globalThis, "fetch")
        .mockResolvedValue(jsonResponse({ expires_at: "2026-09-01T00:00:00Z" }))

      renderWithTerms("#/auth/google/callback?code=the-code&state=the-state")

      expect(await screen.findByText("SIGNED IN")).toBeInTheDocument()
      const [, init] = fetchMock.mock.calls[0] ?? []
      expect(JSON.parse(String(init?.body))).toMatchObject({
        terms_accepted: true,
      })
      expect(window.sessionStorage.getItem(TERMS_KEY)).toBeNull()
    })

    it("asks for the terms a new account needs instead of ending at a refusal", async () => {
      window.sessionStorage.setItem(STATE_KEY, "the-state")
      window.sessionStorage.setItem(RETURN_KEY, "#/organization/billing")
      const assign = vi.fn()
      Object.defineProperty(window, "location", {
        configurable: true,
        value: { ...realLocation, assign },
      })
      const fetchMock = vi
        .spyOn(globalThis, "fetch")
        .mockResolvedValueOnce(termsRefusal())
        .mockResolvedValueOnce(
          jsonResponse({
            authorization_url: "https://accounts.google.com/o/oauth2/auth?x=1",
            state: "the-next-state",
          }),
        )
      const user = userEvent.setup()

      renderWithTerms("#/auth/google/callback?code=the-code&state=the-state")

      expect(
        await screen.findByRole("heading", {
          name: "Accept the terms to continue",
        }),
      ).toBeInTheDocument()
      expect(
        screen.getByRole("link", { name: "terms of service" }),
      ).toHaveAttribute("href", "https://otari.example.com/terms")
      // Either way on keeps the page the refused sign-in started from.
      expect(
        screen.getByRole("link", { name: "Back to sign in" }),
      ).toHaveAttribute("href", "#/organization/billing")
      const again = screen.getByRole("button", { name: "Sign in with Google" })
      expect(again).toBeDisabled()

      await user.click(
        screen.getByRole("checkbox", { name: "I accept the terms of service" }),
      )
      await user.click(again)

      await waitFor(() =>
        expect(assign).toHaveBeenCalledWith(
          "https://accounts.google.com/o/oauth2/auth?x=1",
        ),
      )
      // A fresh round trip, which this time carries the acceptance back.
      expect(String(fetchMock.mock.calls[1]?.[0])).toBe(
        `${API_ROOT}/auth/oauth/google/authorize`,
      )
      expect(window.sessionStorage.getItem(STATE_KEY)).toBe("the-next-state")
      expect(window.sessionStorage.getItem(TERMS_KEY)).toBe("true")
      expect(window.sessionStorage.getItem(RETURN_KEY)).toBe(
        "#/organization/billing",
      )
    })

    it("falls back to the gateway's wording where this tab knows no terms", async () => {
      window.sessionStorage.setItem(STATE_KEY, "the-state")
      vi.spyOn(globalThis, "fetch").mockResolvedValue(termsRefusal())

      renderCallback("#/auth/google/callback?code=the-code&state=the-state")

      expect(
        await screen.findByText("Accept the terms of service to continue"),
      ).toBeInTheDocument()
      expect(screen.queryByRole("checkbox")).toBeNull()
    })
  })

  it("spends the code once under StrictMode, which mounts every effect twice", async () => {
    // `main.tsx` renders the app inside StrictMode, so in development every
    // effect body runs twice on mount. A code is single-use, so without the
    // guard the second pass turns a sign-in the first pass completed into a
    // refusal.
    //
    // The assertion is the outcome, not the fetch count, and that distinction
    // is the whole test. `takeOAuthState` clears the stored state on the first
    // pass, so an unguarded second pass exits early on `invalid_state` and
    // never fetches: a count of one would pass with no guard at all. What only
    // the guard produces is a single recorded outcome with no `invalid_state`
    // beside it.
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ expires_at: "2026-09-01T00:00:00Z" }))

    render(
      <StrictMode>
        <AppProviders>
          <DeploymentProvider
            value={bootstrap({ oauth_providers: ["google"] })}
          >
            <Harness hash="#/auth/google/callback?code=the-code&state=the-state" />
          </DeploymentProvider>
        </AppProviders>
      </StrictMode>,
    )

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await waitFor(() =>
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
        authentication_method: "google",
      }),
    )
    expect(recordEvent).not.toHaveBeenCalledWith(
      TELEMETRY_EVENTS.LOGIN_FAILED,
      expect.objectContaining({ error_code: "invalid_state" }),
    )
    // One outcome in total, so the second pass produced neither a second
    // sign-in nor a refusal.
    expect(recordEvent).toHaveBeenCalledTimes(1)
  })

  it("tells the person when the sign-in removed their earlier password", async () => {
    // Rendered without the harness's gate, the way `DeploymentRoot` mounts it:
    // the page keeps rendering after `login()`, which is what lets it say this.
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    window.location.hash = "#/auth/google/callback"
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        expires_at: "2026-09-01T00:00:00Z",
        password_removed: true,
      }),
    )

    render(
      <AppProviders>
        <DeploymentProvider value={bootstrap({ oauth_providers: ["google"] })}>
          <OAuthCallbackPage
            provider="google"
            hash="#/auth/google/callback?code=the-code&state=the-state"
          />
        </DeploymentProvider>
      </AppProviders>,
    )

    expect(
      await screen.findByRole("heading", { name: "You're signed in" }),
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        /the password set on this account before then was removed/,
      ),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Set a new password" }),
    ).toHaveAttribute("href", "#/account")
    expect(screen.getByRole("link", { name: "Continue" })).toHaveAttribute(
      "href",
      "#/",
    )
    // The notice waits for the person instead of moving them on, even though
    // this tab now holds a session.
    expect(window.location.hash).toBe("#/auth/google/callback")
  })

  it.each([
    [
      "the page the sign-in started from",
      "#/organization/billing",
      "#/organization/billing",
    ],
    [
      "home when that page is not one to return to",
      "#/reset-password?token=t",
      "#/",
    ],
  ])("continues from that notice to %s", async (_, remembered, continueTo) => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    window.sessionStorage.setItem(RETURN_KEY, remembered)
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({
        expires_at: "2026-09-01T00:00:00Z",
        password_removed: true,
      }),
    )

    render(
      <AppProviders>
        <DeploymentProvider value={bootstrap({ oauth_providers: ["google"] })}>
          <OAuthCallbackPage
            provider="google"
            hash="#/auth/google/callback?code=the-code&state=the-state"
          />
        </DeploymentProvider>
      </AppProviders>,
    )

    expect(
      await screen.findByRole("link", { name: "Continue" }),
    ).toHaveAttribute("href", continueTo)
  })

  it("replaces its own history entry on success, so Back skips it", async () => {
    window.location.hash =
      "#/auth/google/callback?code=the-code&state=the-state"
    const entries = window.history.length
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ expires_at: "2026-09-01T00:00:00Z" }),
    )

    renderCallback("#/auth/google/callback?code=the-code&state=the-state")

    await screen.findByText("SIGNED IN")
    await waitFor(() => expect(window.location.hash).toBe("#/"))
    expect(window.history.length).toBe(entries)
  })

  it("sends a tab that is already signed in to the dashboard rather than refusing a spent state", async () => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1")
    window.location.hash = "#/auth/google/callback?code=the-code&state=spent"
    const fetchMock = vi.spyOn(globalThis, "fetch")

    // Mounted bare: the harness would hide the page behind the signed-in marker.
    render(
      <AppProviders>
        <DeploymentProvider value={bootstrap({ oauth_providers: ["google"] })}>
          <OAuthCallbackPage
            provider="google"
            hash="#/auth/google/callback?code=the-code&state=spent"
          />
        </DeploymentProvider>
      </AppProviders>,
    )

    await waitFor(() => expect(window.location.hash).toBe("#/"))
    expect(
      screen.queryByRole("heading", { name: "That sign-in did not complete" }),
    ).toBeNull()
    expect(fetchMock).not.toHaveBeenCalled()
    expect(recordEvent).not.toHaveBeenCalled()
  })
})
