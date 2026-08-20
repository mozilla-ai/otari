import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { MailSettings, SendTestMailResponse } from "@/client"
import { AuthProvider } from "@/features/auth/AuthContext"
import { MailDeliveryCard } from "@/features/settings/MailDeliveryCard"

const UNCONFIGURED: MailSettings = {
  transport: "none",
  enabled: false,
  ready: false,
  from_email: null,
  from_name: "Otari",
  public_base_url: null,
  missing: ["smtp_host", "mail_from_email", "public_base_url"],
}

const READY: MailSettings = {
  transport: "smtp",
  enabled: true,
  ready: true,
  from_email: "otari@example.com",
  from_name: "Otari",
  public_base_url: "https://otari.example.com",
  missing: [],
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockApi(
  settings: MailSettings,
  testResult: SendTestMailResponse | { status: number; detail: string } = {
    ok: true,
    transport: "smtp",
    reason: null,
  },
) {
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      if ((init?.method ?? "GET").toUpperCase() === "POST") {
        return "status" in testResult
          ? jsonResponse({ detail: testResult.detail }, testResult.status)
          : jsonResponse(testResult)
      }
      if (url.includes("/v1/settings/mail")) return jsonResponse(settings)
      return jsonResponse([])
    })
}

function renderCard() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <AuthProvider>
        <MailDeliveryCard />
      </AuthProvider>
    </QueryClientProvider>,
  )
}

describe("MailDeliveryCard", () => {
  beforeEach(() => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1")
  })

  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("claims nothing about mail until the server has answered", async () => {
    mockApi(UNCONFIGURED)
    renderCard()

    // The unconfigured copy is the wrong thing to show before the answer
    // arrives, since it would be wrong on every deployment that does have mail.
    expect(
      screen.queryByText(/Unavailable until a transport/),
    ).not.toBeInTheDocument()
    expect(screen.getByText(/Checking whether/)).toBeInTheDocument()

    expect(await screen.findByText("None")).toBeInTheDocument()
    expect(
      screen.getByText(/Unavailable until a transport/),
    ).toBeInTheDocument()
  })

  it("names the settings that would turn mail on when there is no transport", async () => {
    mockApi(UNCONFIGURED)
    renderCard()

    expect(await screen.findByText("None")).toBeInTheDocument()
    expect(screen.getByText("OTARI_SMTP_HOST")).toBeInTheDocument()
    expect(screen.getByText("OTARI_MAIL_FROM_EMAIL")).toBeInTheDocument()
    expect(screen.getByText("OTARI_PUBLIC_BASE_URL")).toBeInTheDocument()
  })

  it("disables the test send with a reason rather than offering one that would fail", async () => {
    mockApi(UNCONFIGURED)
    renderCard()

    await screen.findByText("None")
    await userEvent.type(screen.getByLabelText("Recipient"), "ada@example.com")

    expect(
      screen.getByRole("button", { name: "Send test email" }),
    ).toBeDisabled()
    expect(
      screen.getByText(/Unavailable until a transport/),
    ).toBeInTheDocument()
  })

  it("reports the configured transport and sender", async () => {
    mockApi(READY)
    renderCard()

    expect(await screen.findByText("SMTP")).toBeInTheDocument()
    expect(screen.getByText("Otari <otari@example.com>")).toBeInTheDocument()
    expect(screen.queryByText("OTARI_SMTP_HOST")).not.toBeInTheDocument()
  })

  it("keeps the test send disabled until an address is typed", async () => {
    mockApi(READY)
    renderCard()

    await screen.findByText("SMTP")
    const button = screen.getByRole("button", { name: "Send test email" })
    expect(button).toBeDisabled()

    await userEvent.type(screen.getByLabelText("Recipient"), "ada@example.com")
    expect(button).toBeEnabled()
  })

  it("confirms a delivered test message", async () => {
    mockApi(READY)
    renderCard()

    await screen.findByText("SMTP")
    await userEvent.type(screen.getByLabelText("Recipient"), "ada@example.com")
    await userEvent.click(
      screen.getByRole("button", { name: "Send test email" }),
    )

    expect(await screen.findByText(/Sent over smtp/)).toBeInTheDocument()
  })

  it("does not tell an operator to check an inbox for a console send", async () => {
    // The console transport writes to the log and delivers to nobody, so the
    // inbox wording would send them looking for a message that never left.
    mockApi(
      { ...READY, transport: "console" },
      { ok: true, transport: "console", reason: null },
    )
    renderCard()

    await screen.findByText("Console (logged, not delivered)")
    await userEvent.type(screen.getByLabelText("Recipient"), "ada@example.com")
    await userEvent.click(
      screen.getByRole("button", { name: "Send test email" }),
    )

    expect(
      await screen.findByText(/Written to the gateway log/),
    ).toBeInTheDocument()
    expect(screen.queryByText(/recipient's inbox/)).not.toBeInTheDocument()
  })

  it("shows the transport's own reason when a configured send fails", async () => {
    mockApi(READY, {
      ok: false,
      transport: "smtp",
      // A neutral sentinel, deliberately not a real exception string: the
      // contract is "whatever reason the server reported reaches the operator",
      // and pinning an SMTP exception format here would make one library's
      // wording into a UI contract this component never promised.
      reason: "the-reason-the-server-gave",
    })
    renderCard()

    await screen.findByText("SMTP")
    await userEvent.type(screen.getByLabelText("Recipient"), "ada@example.com")
    await userEvent.click(
      screen.getByRole("button", { name: "Send test email" }),
    )

    expect(
      await screen.findByText(/the-reason-the-server-gave/),
    ).toBeInTheDocument()
  })

  it("surfaces the server's refusal when mail is turned off between load and send", async () => {
    // The button is disabled in that state, so reaching the 503 means the
    // configuration changed under an open page; the refusal still has to read
    // as an error rather than as a delivered message.
    mockApi(READY, {
      status: 503,
      detail:
        "Outgoing mail is not configured on this deployment (missing: smtp_host).",
    })
    renderCard()

    await screen.findByText("SMTP")
    await userEvent.type(screen.getByLabelText("Recipient"), "ada@example.com")
    await userEvent.click(
      screen.getByRole("button", { name: "Send test email" }),
    )

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/not configured/),
    )
  })
})
