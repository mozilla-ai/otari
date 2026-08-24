import { render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useAuth } from "@/features/auth/AuthContext"
import { OAuthCallbackPage } from "@/features/auth/OAuthCallbackPage"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"
import { recordEvent } from "@/tests/telemetry"

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

/** Mounted the way `DeploymentRoot` mounts it: ahead of the auth gate. */
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
  window.sessionStorage.clear()
  // `AuthContext` keeps a "there is a session" marker in localStorage, so a
  // test that signs in would leave every later one already signed in.
  window.localStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
  window.sessionStorage.clear()
  window.localStorage.clear()
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

  it("posts the code once, even though an effect may mount twice", async () => {
    // A code is single-use, so a second post would refuse a sign-in the first
    // one completed. StrictMode's double mount is the shape that produces it.
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ expires_at: "2026-09-01T00:00:00Z" }))
    const { rerender } = renderCallback(
      "#/auth/google/callback?code=the-code&state=the-state",
    )

    rerender(
      <AppProviders>
        <DeploymentProvider value={bootstrap({ oauth_providers: ["google"] })}>
          <Harness hash="#/auth/google/callback?code=the-code&state=the-state" />
        </DeploymentProvider>
      </AppProviders>,
    )

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
  })
})
