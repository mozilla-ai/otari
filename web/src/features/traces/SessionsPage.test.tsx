import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { ScopedBudget } from "@/client"
import { SessionsPage } from "@/features/traces/SessionsPage"
import { mockApi, renderPage } from "@/tests/activity"
import { traceDetail, traceSummary } from "@/tests/traceFixtures"

afterEach(() => {
  vi.restoreAllMocks()
})

function keyCeiling(overrides: Partial<ScopedBudget> = {}): ScopedBudget {
  return {
    id: "ceiling-1",
    budget_id: "budget-1",
    scope_type: "api_token",
    scope_id: "key-1",
    provider_key_id: null,
    name: "Key ceiling",
    max_budget: 10,
    current_spend: 0,
    reserved_spend: 0,
    token_limit: null,
    current_tokens: 0,
    reserved_tokens: 0,
    request_limit: null,
    current_requests: 0,
    reserved_requests: 0,
    budget_duration_sec: null,
    reset_alignment: null,
    period_start: null,
    period_end: null,
    created_at: "2026-10-01T00:00:00Z",
    updated_at: "2026-10-01T00:00:00Z",
    ...overrides,
  }
}

async function openFirstSession() {
  const sessions = await screen.findByRole("grid", { name: "Sessions" })
  await userEvent.click(
    await within(sessions).findByRole("row", { name: /claude-code/ }),
  )
  return screen.findByRole("complementary", { name: "Trace claude-code" })
}

describe("SessionsPage", () => {
  it("lists sessions from the trace API and opens one into its spans", async () => {
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    expect(await within(peek).findByText("Bash")).toBeInTheDocument()
    expect(within(peek).getAllByText("gpt-5").length).toBeGreaterThan(0)
  })

  it("reads an opened session in the workspace its list entry names", async () => {
    const { calls } = mockApi({
      traces: [traceSummary()],
      traceDetail: traceDetail(),
    })

    renderPage(<SessionsPage />, "/sessions")
    await openFirstSession()

    await vi.waitFor(() =>
      expect(
        calls.some((call) =>
          call.url.includes(
            `/traces/${traceSummary().trace_id}?workspace_id=${traceSummary().workspace_id}`,
          ),
        ),
      ).toBe(true),
    )
  })

  it("leads an opened session with its spend against the key's ceiling", async () => {
    const summary = traceSummary({ api_key_id: "key-1", cost: 0.5 })
    mockApi({
      traces: [summary],
      traceDetail: traceDetail({ summary }),
      ceilings: [keyCeiling({ max_budget: 10, current_spend: 6 })],
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    expect(
      await within(peek).findByText(
        /This session cost \$0\.50\. That is 5\.0% of this key's \$10\.00 ceiling, which has \$6\.00 spent this period\./,
      ),
    ).toBeInTheDocument()
    expect(
      within(peek).getByRole("progressbar", {
        name: "This key's spend against its ceiling",
      }),
    ).toBeInTheDocument()
  })

  it("says when a key has no ceiling of its own", async () => {
    const summary = traceSummary({ api_key_id: "key-2" })
    mockApi({
      traces: [summary],
      traceDetail: traceDetail({ summary }),
      ceilings: [keyCeiling()],
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    expect(
      await within(peek).findByText(/This key has no spend ceiling of its own/),
    ).toBeInTheDocument()
  })

  it("does not ask a tenant for the deployment's ceilings", async () => {
    const summary = traceSummary({ api_key_id: "key-1" })
    const { calls } = mockApi({
      traces: [summary],
      traceDetail: traceDetail({ summary }),
      deploymentOperator: false,
    })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()

    expect(
      await within(peek).findByText("This session cost $0.02."),
    ).toBeInTheDocument()
    expect(calls.some((call) => call.url.includes("/scoped-budgets"))).toBe(
      false,
    )
  })

  it("opens the span a gateway decision is about", async () => {
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<SessionsPage />, "/sessions")
    const peek = await openFirstSession()
    await userEvent.click(
      await within(peek).findByRole("button", {
        name: /gpt-5 failed, a later attempt answered/,
      }),
    )

    expect(
      await within(peek).findByRole("button", { name: "Attributes" }),
    ).toBeInTheDocument()
    expect(
      within(peek).getByText("error, recovered by a fallback"),
    ).toBeInTheDocument()
  })

  it("says so when the session list cannot be read", async () => {
    const { mock } = mockApi()
    const answer = mock.getMockImplementation()
    mock.mockImplementation(async (input, init) =>
      /\/traces\?/.test(String(input))
        ? new Response(JSON.stringify({ detail: "Trace store unavailable" }), {
            status: 500,
            headers: { "Content-Type": "application/json" },
          })
        : (answer as typeof fetch)(input, init),
    )

    renderPage(<SessionsPage />, "/sessions")

    expect(
      await screen.findByText(/Trace store unavailable/),
    ).toBeInTheDocument()
  })
})
