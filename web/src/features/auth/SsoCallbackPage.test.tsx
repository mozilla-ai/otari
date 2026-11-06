import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { StrictMode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { SsoAuthorizeResponse, SsoSessionPublic } from "@/client"
import { useAuth } from "@/features/auth/AuthContext"
import { SsoCallbackPage } from "@/features/auth/SsoCallbackPage"
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

const STATE_KEY = "otari.sso.state"
const SLUG_KEY = "otari.sso.slug"
const TERMS_KEY = "otari.sso.terms-accepted"
const RETURN_KEY = "otari.sso.return-hash"
const COMPANY_KEY = "otari.dashboard.rememberedCompany"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function sessionBody(
  overrides: Partial<SsoSessionPublic> = {},
): SsoSessionPublic {
  return {
    active_organization_id: null,
    expires_at: "2026-09-01T00:00:00Z",
    password_removed: false,
    user_id: "11111111-1111-1111-1111-111111111111",
    ...overrides,
  }
}

/** The page in isolation, with a signed-in marker to assert against. */
function Harness({ hash }: { hash: string }) {
  const { isAuthenticated } = useAuth()
  return isAuthenticated ? (
    <div>SIGNED IN</div>
  ) : (
    <SsoCallbackPage hash={hash} />
  )
}

function renderCallback(hash: string) {
  return render(
    <AppProviders>
      <DeploymentProvider value={bootstrap()}>
        <Harness hash={hash} />
      </DeploymentProvider>
    </AppProviders>,
  )
}

beforeEach(() => {
  resetTelemetrySpy()
  window.sessionStorage.clear()
  window.localStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
  window.sessionStorage.clear()
  window.localStorage.clear()
  window.location.hash = ""
})

describe("SsoCallbackPage", () => {
  it("spends the code for a session when the state matches the one this tab stored, and remembers the company", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    window.sessionStorage.setItem(SLUG_KEY, "acme")
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(sessionBody()))

    renderCallback("#/auth/sso/callback?code=the-code&state=the-state")

    expect(await screen.findByText("SIGNED IN")).toBeInTheDocument()
    const [url, init] = fetchMock.mock.calls[0] ?? []
    expect(url).toBe(`${API_ROOT}/auth/sso/callback`)
    expect(JSON.parse(String(init?.body))).toEqual({
      code: "the-code",
      state: "the-state",
      terms_accepted: false,
    })
    expect(window.localStorage.getItem(COMPANY_KEY)).toBe("acme")
    await waitFor(() =>
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
        authentication_method: "sso",
      }),
    )
  })

  it("forwards the iss the redirect carried, and substitutes none of its own", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    window.sessionStorage.setItem(SLUG_KEY, "acme")
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(sessionBody()))

    renderCallback(
      "#/auth/sso/callback?code=the-code&state=the-state&iss=https://idp.acme.example",
    )

    await screen.findByText("SIGNED IN")
    const [, init] = fetchMock.mock.calls[0] ?? []
    expect(JSON.parse(String(init?.body))).toMatchObject({
      iss: "https://idp.acme.example",
    })
  })

  it("sends no iss when the redirect carried none", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    window.sessionStorage.setItem(SLUG_KEY, "acme")
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(sessionBody()))

    renderCallback("#/auth/sso/callback?code=the-code&state=the-state")

    await screen.findByText("SIGNED IN")
    const [, init] = fetchMock.mock.calls[0] ?? []
    expect(JSON.parse(String(init?.body))).not.toHaveProperty("iss")
  })

  it("does not remember the company when the sign-in fails", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    window.sessionStorage.setItem(SLUG_KEY, "acme")
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Not registered here." }, 401),
    )

    renderCallback("#/auth/sso/callback?code=the-code&state=the-state")

    await screen.findByRole("heading", {
      name: "That sign-in did not complete",
    })
    expect(window.localStorage.getItem(COMPANY_KEY)).toBeNull()
  })

  it("refuses a state that does not match, without spending the code", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    const fetchMock = vi.spyOn(globalThis, "fetch")

    renderCallback("#/auth/sso/callback?code=the-code&state=somebody-elses")

    expect(
      await screen.findByRole("heading", {
        name: "That sign-in did not complete",
      }),
    ).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
    await waitFor(() =>
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: "sso",
        error_code: "invalid_state",
      }),
    )
  })

  it("refuses a callback carrying a state but no code", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    const fetchMock = vi.spyOn(globalThis, "fetch")

    renderCallback("#/auth/sso/callback?state=the-state")

    expect(
      await screen.findByText(
        /sent this browser back without an authorization code/,
      ),
    ).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("says so when the provider reports the person declined, and calls nothing", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    const fetchMock = vi.spyOn(globalThis, "fetch")

    renderCallback("#/auth/sso/callback?error=access_denied&state=the-state")

    expect(await screen.findByText(/You may have declined/)).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("sends a tab that is already signed in to the dashboard rather than refusing a spent state", async () => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1")
    const fetchMock = vi.spyOn(globalThis, "fetch")

    render(
      <AppProviders>
        <DeploymentProvider value={bootstrap()}>
          <SsoCallbackPage hash="#/auth/sso/callback?code=the-code&state=spent" />
        </DeploymentProvider>
      </AppProviders>,
    )

    await waitFor(() => expect(window.location.hash).toBe("#/"))
    expect(fetchMock).not.toHaveBeenCalled()
    expect(recordEvent).not.toHaveBeenCalled()
  })

  it("returns to the page the sign-in started from", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    window.sessionStorage.setItem(SLUG_KEY, "acme")
    window.sessionStorage.setItem(RETURN_KEY, "#/organization/billing")
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(sessionBody()))

    renderCallback("#/auth/sso/callback?code=the-code&state=the-state")

    await screen.findByText("SIGNED IN")
    expect(window.location.hash).toBe("#/organization/billing")
  })

  it("tells the person when the sign-in removed their earlier password", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    window.sessionStorage.setItem(SLUG_KEY, "acme")
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(sessionBody({ password_removed: true })),
    )

    render(
      <AppProviders>
        <DeploymentProvider value={bootstrap()}>
          <SsoCallbackPage hash="#/auth/sso/callback?code=the-code&state=the-state" />
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
  })

  describe("on a deployment that publishes terms", () => {
    const realLocation = window.location
    const TERMS_URL = "https://otari.example.com/terms"

    function renderWithTerms(hash: string) {
      return render(
        <AppProviders>
          <DeploymentProvider value={bootstrap({ terms_url: TERMS_URL })}>
            <Harness hash={hash} />
          </DeploymentProvider>
        </AppProviders>,
      )
    }

    afterEach(() => {
      Object.defineProperty(window, "location", {
        configurable: true,
        value: realLocation,
      })
    })

    it("asks for the terms a new account needs, and retries through the same connection", async () => {
      window.sessionStorage.setItem(STATE_KEY, "the-state")
      window.sessionStorage.setItem(SLUG_KEY, "acme")
      window.sessionStorage.setItem(RETURN_KEY, "#/organization/billing")
      const assign = vi.fn()
      Object.defineProperty(window, "location", {
        configurable: true,
        value: { ...realLocation, assign },
      })
      const authorized: SsoAuthorizeResponse = {
        authorization_url: "https://idp.acme.example/authorize?x=1",
        state: "the-next-state",
      }
      const fetchMock = vi
        .spyOn(globalThis, "fetch")
        .mockResolvedValueOnce(
          jsonResponse(
            {
              detail: "Accept the terms of service to continue",
              code: "terms_not_accepted",
            },
            400,
          ),
        )
        .mockResolvedValueOnce(jsonResponse(authorized))
      const user = userEvent.setup()

      renderWithTerms("#/auth/sso/callback?code=the-code&state=the-state")

      expect(
        await screen.findByRole("heading", {
          name: "Accept the terms to continue",
        }),
      ).toBeInTheDocument()
      const again = screen.getByRole("button", {
        name: "Continue signing in",
      })
      expect(again).toBeDisabled()

      await user.click(
        screen.getByRole("checkbox", { name: "I accept the terms of service" }),
      )
      await user.click(again)

      await waitFor(() =>
        expect(assign).toHaveBeenCalledWith(
          "https://idp.acme.example/authorize?x=1",
        ),
      )
      expect(String(fetchMock.mock.calls[1]?.[0])).toBe(
        `${API_ROOT}/auth/sso/acme/authorize`,
      )
      expect(window.sessionStorage.getItem(STATE_KEY)).toBe("the-next-state")
      expect(window.sessionStorage.getItem(TERMS_KEY)).toBe("true")
      expect(window.sessionStorage.getItem(RETURN_KEY)).toBe(
        "#/organization/billing",
      )
    })
  })

  it("spends the code once under StrictMode, which mounts every effect twice", async () => {
    window.sessionStorage.setItem(STATE_KEY, "the-state")
    window.sessionStorage.setItem(SLUG_KEY, "acme")
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(sessionBody()))

    render(
      <StrictMode>
        <AppProviders>
          <DeploymentProvider value={bootstrap()}>
            <Harness hash="#/auth/sso/callback?code=the-code&state=the-state" />
          </DeploymentProvider>
        </AppProviders>
      </StrictMode>,
    )

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await waitFor(() =>
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
        authentication_method: "sso",
      }),
    )
    expect(recordEvent).not.toHaveBeenCalledWith(
      TELEMETRY_EVENTS.LOGIN_FAILED,
      expect.objectContaining({ error_code: "invalid_state" }),
    )
  })
})
