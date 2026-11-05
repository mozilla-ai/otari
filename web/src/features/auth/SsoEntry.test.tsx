import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { SsoAuthorizeResponse, SsoLookupResponse } from "@/client"
import { SsoEntry } from "@/features/auth/SsoEntry"
import { AppProviders } from "@/tests/providers"

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

function renderEntry() {
  return render(
    <AppProviders>
      <SsoEntry />
    </AppProviders>,
  )
}

const realLocation = window.location

function stubNavigation(hash = "") {
  const assign = vi.fn()
  Object.defineProperty(window, "location", {
    configurable: true,
    value: { ...window.location, assign, hash },
  })
  return assign
}

beforeEach(() => {
  window.localStorage.clear()
  window.sessionStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
  window.sessionStorage.clear()
  Object.defineProperty(window, "location", {
    configurable: true,
    value: realLocation,
  })
})

describe("SsoEntry", () => {
  it("opens one address field and a Continue button", async () => {
    renderEntry()
    const user = userEvent.setup()

    await user.click(
      screen.getByRole("button", { name: "Sign in with your company (SSO)" }),
    )

    expect(screen.getByLabelText("Work email")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Continue" })).toBeInTheDocument()
  })

  it("offers the other sign-in options back when no company matches", async () => {
    const notFound: SsoLookupResponse = { found: false }
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(notFound))
    renderEntry()
    const user = userEvent.setup()

    await user.click(
      screen.getByRole("button", { name: "Sign in with your company (SSO)" }),
    )
    await user.type(screen.getByLabelText("Work email"), "nobody@example.com")
    await user.click(screen.getByRole("button", { name: "Continue" }))

    expect(
      await screen.findByText(
        "We could not find a company sign-in for that address.",
      ),
    ).toBeInTheDocument()
    const back = screen.getByRole("button", { name: "Back to sign in" })
    await user.click(back)

    // Collapsed back to the closed row, the other sign-in options' own level.
    expect(
      screen.getByRole("button", { name: "Sign in with your company (SSO)" }),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText("Work email")).not.toBeInTheDocument()
  })

  it("names the remembered company and skips the address step", async () => {
    window.localStorage.setItem("otari.dashboard.rememberedCompany", "acme")
    renderEntry()
    const user = userEvent.setup()

    await user.click(
      screen.getByRole("button", { name: "Sign in with your company (SSO)" }),
    )

    expect(screen.queryByLabelText("Work email")).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Continue with acme" }),
    ).toBeInTheDocument()
  })

  it("renders the card normally when localStorage throws", async () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked")
    })
    renderEntry()
    const user = userEvent.setup()

    expect(
      screen.getByRole("button", { name: "Sign in with your company (SSO)" }),
    ).toBeInTheDocument()

    await user.click(
      screen.getByRole("button", { name: "Sign in with your company (SSO)" }),
    )

    // A read that threw is read as nothing remembered, so the address step
    // opens rather than the row failing to render at all.
    expect(screen.getByLabelText("Work email")).toBeInTheDocument()
  })

  it("looks up the address, then leaves for the organization's connection when it matches", async () => {
    const assign = stubNavigation()
    const found: SsoLookupResponse = {
      found: true,
      organization_name: "Acme Corp",
      slug: "acme",
    }
    const authorized: SsoAuthorizeResponse = {
      authorization_url: "https://idp.acme.example/authorize?x=1",
      state: "the-state",
    }
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(found))
      .mockResolvedValueOnce(jsonResponse(authorized))
    renderEntry()
    const user = userEvent.setup()

    await user.click(
      screen.getByRole("button", { name: "Sign in with your company (SSO)" }),
    )
    await user.type(screen.getByLabelText("Work email"), "ana@acme.example")
    await user.click(screen.getByRole("button", { name: "Continue" }))

    await waitFor(() =>
      expect(assign).toHaveBeenCalledWith(
        "https://idp.acme.example/authorize?x=1",
      ),
    )
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(window.sessionStorage.getItem("otari.sso.state")).toBe("the-state")
    expect(window.sessionStorage.getItem("otari.sso.slug")).toBe("acme")
  })

  it("falls back to the address step when a remembered company's connection is gone", async () => {
    window.localStorage.setItem("otari.dashboard.rememberedCompany", "acme")
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Not available." }, 404),
    )
    renderEntry()
    const user = userEvent.setup()

    await user.click(
      screen.getByRole("button", { name: "Sign in with your company (SSO)" }),
    )
    await user.click(screen.getByRole("button", { name: "Continue with acme" }))

    expect(
      await screen.findByText(
        "We could not find a company sign-in for that address.",
      ),
    ).toBeInTheDocument()
    expect(
      window.localStorage.getItem("otari.dashboard.rememberedCompany"),
    ).toBeNull()
  })
})
