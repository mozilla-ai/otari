import { render, screen, waitFor } from "@testing-library/react"
import { StrictMode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useAuth } from "@/features/auth/AuthContext"
import { OAuthCallbackPage } from "@/features/auth/OAuthCallbackPage"
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
  // The success path now hands the tab back to "#/", so a test that signs in
  // would otherwise leave that hash set for the next one.
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
    expect(url).toBe("/v1/auth/oauth/google/callback")
    // The code alone: the redirect URI is the gateway's own, and the state was
    // already checked here against the value this tab stored.
    expect(JSON.parse(String(init?.body))).toEqual({ code: "the-code" })
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
})
