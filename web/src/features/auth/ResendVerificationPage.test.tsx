import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ResendVerificationPage } from "@/features/auth/ResendVerificationPage"
import { apiFetch } from "@/shared/api/client"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
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
  return { useTelemetry: () => telemetrySpy }
})

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <ResendVerificationPage />
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

describe("ResendVerificationPage", () => {
  it("sends a fresh link and lands on the check-email page", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.click(screen.getByRole("button", { name: "Send link" }))

    await vi.waitFor(() => {
      expect(window.location.hash).toBe("#/check-email?type=resend")
    })
    expect(vi.mocked(apiFetch).mock.calls[0]?.[0]).toBe(
      "/v1/auth/resend-verification",
    )
  })
})

describe("the telemetry the resend page records", () => {
  it("records the link once the gateway has sent it", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.click(screen.getByRole("button", { name: "Send link" }))

    await vi.waitFor(() => {
      expect(recordEvent).toHaveBeenCalledWith(
        TELEMETRY_EVENTS.RESEND_VERIFICATION_CLICKED,
      )
    })
  })

  it("records nothing when the gateway refused to send one", async () => {
    // On the send rather than on the click: a request the gateway refused is
    // not a link anybody was sent.
    vi.mocked(apiFetch).mockRejectedValue(new Error("nope"))
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.click(screen.getByRole("button", { name: "Send link" }))

    await vi.waitFor(() => {
      expect(apiFetch).toHaveBeenCalled()
    })
    expect(recordEvent).not.toHaveBeenCalled()
  })
})
