import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { SessionsPage } from "@/features/traces/SessionsPage"
import { mockApi, renderPage } from "@/tests/activity"
import { traceDetail, traceSummary } from "@/tests/traceFixtures"

afterEach(() => {
  vi.restoreAllMocks()
})

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
