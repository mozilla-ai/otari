import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AuthProvider } from "@/features/auth/AuthContext"
import { MaintenanceModeCard } from "@/features/settings/MaintenanceModeCard"

const SWITCH = "Freeze new dashboard sign-ins"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

/** Serves the freeze flag and records what a PATCH asked for. */
function mockApi(enabled: boolean) {
  let current = enabled
  const patched: boolean[] = []
  const fetchMock = vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (_input, init) => {
      if ((init?.method ?? "GET").toUpperCase() === "PATCH") {
        const body = JSON.parse(String(init?.body)) as { enabled: boolean }
        patched.push(body.enabled)
        current = body.enabled
      }
      return jsonResponse({ enabled: current })
    })
  return { fetchMock, patched }
}

function renderCard() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <AuthProvider>
        <MaintenanceModeCard />
      </AuthProvider>
    </QueryClientProvider>,
  )
}

describe("MaintenanceModeCard", () => {
  beforeEach(() => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1")
  })

  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("claims nothing about the freeze until the server has answered", async () => {
    // An "off" switch drawn while the request is in flight would state the
    // opposite of the truth on exactly the deployment being checked on.
    mockApi(true)
    renderCard()

    expect(screen.queryByRole("switch")).not.toBeInTheDocument()

    await waitFor(() =>
      expect(screen.getByRole("switch")).toHaveAttribute(
        "aria-checked",
        "true",
      ),
    )
  })

  it("turns the freeze on", async () => {
    const { patched } = mockApi(false)
    const user = userEvent.setup()
    renderCard()

    const toggle = await screen.findByRole("switch", { name: SWITCH })
    expect(toggle).toHaveAttribute("aria-checked", "false")

    await user.click(toggle)

    await waitFor(() => expect(patched).toEqual([true]))
    await waitFor(() =>
      expect(screen.getByRole("switch")).toHaveAttribute(
        "aria-checked",
        "true",
      ),
    )
  })

  it("turns the freeze back off", async () => {
    const { patched } = mockApi(true)
    const user = userEvent.setup()
    renderCard()

    await user.click(await screen.findByRole("switch", { name: SWITCH }))

    await waitFor(() => expect(patched).toEqual([false]))
  })

  it("says the operator's own session and the API are unaffected", async () => {
    // The two facts someone about to flip this needs, and the reason they can
    // flip it back: neither their session nor anyone's API traffic is frozen.
    mockApi(false)
    renderCard()

    await screen.findByRole("switch", { name: SWITCH })

    expect(
      screen.getByText(/Sessions already open keep working/),
    ).toBeInTheDocument()
    expect(screen.getByText(/does not touch the API/)).toBeInTheDocument()
  })

  it("surfaces a refusal rather than silently leaving the switch where it was", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "Database error" }, 500),
    )
    renderCard()

    expect(await screen.findByText(/Database error/)).toBeInTheDocument()
  })
})
