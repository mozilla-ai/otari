import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { SsoAuthorizeResponse } from "@/client"
import { SsoAuthorizePage } from "@/features/auth/SsoAuthorizePage"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"
import { recordEvent, resetTelemetrySpy } from "@/tests/telemetry"

vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: vi.fn(() => telemetrySpy) }
})

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function renderPage(slug: string) {
  return render(
    <AppProviders>
      <DeploymentProvider value={bootstrap()}>
        <SsoAuthorizePage slug={slug} />
      </DeploymentProvider>
    </AppProviders>,
  )
}

const realLocation = window.location

function stubNavigation() {
  const assign = vi.fn()
  Object.defineProperty(window, "location", {
    configurable: true,
    value: { ...window.location, assign },
  })
  return assign
}

beforeEach(() => {
  resetTelemetrySpy()
  window.sessionStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
  window.sessionStorage.clear()
  window.location.hash = ""
  Object.defineProperty(window, "location", {
    configurable: true,
    value: realLocation,
  })
})

describe("SsoAuthorizePage", () => {
  it("names the organization and offers a control rather than leaving on its own", () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")

    renderPage("acme")

    expect(screen.getByText("acme")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Continue with acme" }),
    ).toBeInTheDocument()
    // Nothing starts before the control is pressed.
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("starts the round trip when the control is pressed", async () => {
    const assign = stubNavigation()
    const authorized: SsoAuthorizeResponse = {
      authorization_url: "https://idp.acme.example/authorize?x=1",
      state: "the-state",
    }
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(authorized))
    renderPage("acme")
    const user = userEvent.setup()

    await user.click(screen.getByRole("button", { name: "Continue with acme" }))

    await waitFor(() =>
      expect(assign).toHaveBeenCalledWith(
        "https://idp.acme.example/authorize?x=1",
      ),
    )
    expect(window.sessionStorage.getItem("otari.sso.state")).toBe("the-state")
    expect(window.sessionStorage.getItem("otari.sso.slug")).toBe("acme")
  })

  it("shows a neutral refusal and the ordinary sign-in options for a slug with no sign-in to start", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "No sign-in is available at this address" }, 404),
    )
    renderPage("no-such-org")
    const user = userEvent.setup()

    await user.click(
      screen.getByRole("button", { name: "Continue with no-such-org" }),
    )

    expect(
      await screen.findByRole("heading", {
        name: "That sign-in is not available",
      }),
    ).toBeInTheDocument()
    // Neutral: it does not say whether the slug is unknown or just disabled.
    expect(screen.queryByText(/no-such-org/)).not.toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Back to sign in" }),
    ).toBeInTheDocument()
    await waitFor(() =>
      expect(recordEvent).toHaveBeenCalledWith(TELEMETRY_EVENTS.LOGIN_FAILED, {
        authentication_method: "sso",
        error_code: "http_404",
        status: 404,
      }),
    )
  })
})
