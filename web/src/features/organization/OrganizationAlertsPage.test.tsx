import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { AlertRule, OrganizationContext } from "@/client"
import { OrganizationAlertsPage } from "@/features/organization/OrganizationAlertsPage"
import { alertRule, organizationContext } from "@/tests/fixtures"

// Mocked at the `apiFetch` boundary (`globalThis.fetch`) rather than at the
// hooks, per the frontend standards: the query keys, the invalidations and the
// request bodies are part of what these tests are for.

interface Request {
  url: string
  method: string
  body: unknown
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

interface MockOpts {
  rules?: AlertRule[]
  context?: OrganizationContext
  /** What the test-send call answers, so a test can drive the refusal path. */
  test?: () => Response
}

function mockApi(opts: MockOpts = {}) {
  const rules = opts.rules ?? []
  const requests: Request[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    requests.push({
      url,
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })

    if (url.includes("/test")) {
      return opts.test
        ? opts.test()
        : jsonResponse({ delivered: true, detail: null })
    }
    if (url.includes("/alert-rules")) {
      if (method === "GET") {
        return jsonResponse({ count: rules.length, data: rules })
      }
      // Every write is re-read through the invalidated list, so one row serves.
      return jsonResponse(rules[0] ?? alertRule())
    }
    return jsonResponse(opts.context ?? organizationContext())
  })
  return requests
}

function renderPage(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("OrganizationAlertsPage", () => {
  it("lists the organization's rules", async () => {
    mockApi({ rules: [alertRule({ name: "Platform team Slack" })] })
    renderPage(<OrganizationAlertsPage />)

    expect(await screen.findByText("Platform team Slack")).toBeInTheDocument()
    // The kind is derived from the redacted scheme, which is all the client has.
    expect(screen.getByText("Slack")).toBeInTheDocument()
    expect(screen.getByText("80%")).toBeInTheDocument()
    expect(screen.getByText("Limit reached")).toBeInTheDocument()
    expect(screen.getByText("Active")).toBeInTheDocument()
  })

  it("never renders anything but the server's redaction", async () => {
    // The security assertion of this file. The page has no way to obtain a real
    // Apprise URL, and this fails if a future change starts echoing one.
    mockApi({ rules: [alertRule({ destination: "slack://***" })] })
    renderPage(<OrganizationAlertsPage />)

    await screen.findByText("Platform team Slack")
    expect(screen.getByText("slack://***")).toBeInTheDocument()
    expect(document.body.textContent).not.toContain("xoxb")
  })

  it("shows an empty state rather than a bare table", async () => {
    mockApi({ rules: [] })
    renderPage(<OrganizationAlertsPage />)

    expect(
      await screen.findByText(/No alert destinations yet/),
    ).toBeInTheDocument()
  })

  it("creates a rule with the thresholds the form collected", async () => {
    const requests = mockApi({ rules: [] })
    renderPage(<OrganizationAlertsPage />)

    await screen.findByText(/No alert destinations yet/)
    await userEvent.click(
      screen.getByRole("button", { name: "Add destination" }),
    )

    await userEvent.type(screen.getByLabelText("Name"), "Ops webhook")
    await userEvent.type(
      screen.getByLabelText("Destination"),
      "json://hooks.example.com/incoming",
    )
    await userEvent.click(
      screen.getByRole("button", { name: "Add destination" }),
    )

    await waitFor(() => {
      expect(
        requests.some(
          (request) =>
            request.method === "POST" && request.url.includes("/alert-rules"),
        ),
      ).toBe(true)
    })
    const created = requests.find((request) => request.method === "POST")
    expect(created?.body).toMatchObject({
      name: "Ops webhook",
      destination: "json://hooks.example.com/incoming",
      warn_at_percent: 80,
      notify_on_exceeded: true,
      enabled: true,
    })
  })

  it("will not submit a rule that could never fire", async () => {
    mockApi({ rules: [] })
    renderPage(<OrganizationAlertsPage />)

    await screen.findByText(/No alert destinations yet/)
    await userEvent.click(
      screen.getByRole("button", { name: "Add destination" }),
    )
    await userEvent.type(screen.getByLabelText("Name"), "Inert")
    await userEvent.type(screen.getByLabelText("Destination"), "slack://a/b/c")

    // Turn off the refusal alert, then clear the warning threshold: together
    // these leave a rule with nothing to send.
    await userEvent.click(
      screen.getByRole("checkbox", {
        name: /Also alert when a budget starts refusing requests/,
      }),
    )
    await userEvent.click(screen.getByRole("button", { name: /Warn early at/ }))
    await userEvent.click(
      await screen.findByRole("option", { name: "No early warning" }),
    )

    expect(await screen.findByText(/would never/)).toBeInTheDocument()
    const submit = screen
      .getAllByRole("button", { name: "Add destination" })
      .at(-1)
    expect(submit).toBeDisabled()
  })

  it("pauses a rule without touching its destination", async () => {
    const requests = mockApi({ rules: [alertRule({ enabled: true })] })
    renderPage(<OrganizationAlertsPage />)

    await screen.findByText("Platform team Slack")
    await userEvent.click(screen.getByRole("button", { name: "Pause" }))

    await waitFor(() => {
      expect(requests.some((request) => request.method === "PATCH")).toBe(true)
    })
    const patch = requests.find((request) => request.method === "PATCH")
    // Only `enabled`. Sending `destination` back would post the redaction as a
    // real value, which is why the update body is built field by field.
    expect(patch?.body).toEqual({ enabled: false })
  })

  it("reports a successful test send", async () => {
    mockApi({ rules: [alertRule()] })
    renderPage(<OrganizationAlertsPage />)

    await screen.findByText("Platform team Slack")
    await userEvent.click(screen.getByRole("button", { name: "Send test" }))

    expect(await screen.findByText("Test delivered")).toBeInTheDocument()
  })

  it("reports a refused test send rather than claiming success", async () => {
    // The case the whole test action exists for: a destination that parses,
    // stores, and then quietly accepts nothing.
    mockApi({
      rules: [alertRule()],
      test: () =>
        jsonResponse({ delivered: false, detail: "Timed out after 15s" }),
    })
    renderPage(<OrganizationAlertsPage />)

    await screen.findByText("Platform team Slack")
    await userEvent.click(screen.getByRole("button", { name: "Send test" }))

    expect(await screen.findByText("Test failed")).toBeInTheDocument()
  })

  it("tells a plain member they may not manage alerts, and offers no controls", async () => {
    mockApi({
      rules: [],
      context: organizationContext({ role: "member" }),
    })
    renderPage(<OrganizationAlertsPage />)

    expect(
      await screen.findByText(/Only organization owners and admins/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Add destination" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Send test" }),
    ).not.toBeInTheDocument()
  })
})
