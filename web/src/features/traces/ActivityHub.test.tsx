import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ActivityHub } from "@/features/traces/ActivityHub"
import { mockApi, renderPage } from "@/tests/activity"
import { STANDALONE_SURFACES } from "@/tests/fixtures"
import { traceDetail, traceSummary } from "@/tests/traceFixtures"

const WITH_TRACES = { surfaces: [...STANDALONE_SURFACES, "traces"] }

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ActivityHub", () => {
  it("is the request log where the deployment records no traces", async () => {
    mockApi()

    renderPage(<ActivityHub />)

    expect(await screen.findByText(/A per-request log/)).toBeInTheDocument()
    expect(
      screen.queryByRole("radiogroup", { name: "Activity view" }),
    ).not.toBeInTheDocument()
  })

  it("opens on agent sessions, and opens one into its turns", async () => {
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<ActivityHub />, "/activity", WITH_TRACES)

    const sessions = await screen.findByRole("grid", { name: "Sessions" })
    await userEvent.click(
      await within(sessions).findByRole("row", { name: /claude-code/ }),
    )

    const panel = await screen.findByRole("dialog", { name: "Session" })
    expect(
      await within(panel).findByRole("region", { name: "Turn 1" }),
    ).toBeInTheDocument()
    await userEvent.click(within(panel).getByRole("button", { name: "Close" }))
    expect(
      screen.queryByRole("dialog", { name: "Session" }),
    ).not.toBeInTheDocument()
  })

  it("reads an opened session in the workspace its list entry names", async () => {
    const { calls } = mockApi({
      traces: [traceSummary()],
      traceDetail: traceDetail(),
    })

    renderPage(<ActivityHub />, "/activity", WITH_TRACES)
    const sessions = await screen.findByRole("grid", { name: "Sessions" })
    await userEvent.click(
      await within(sessions).findByRole("row", { name: /claude-code/ }),
    )

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

  it("names a single request's panel as a request", async () => {
    const request = traceSummary({
      trace_id: "req-9",
      session_source: "none",
      harness: null,
    })
    mockApi({
      traces: [request],
      traceDetail: traceDetail({ summary: request }),
    })

    renderPage(<ActivityHub />, "/activity?trace=req-9", WITH_TRACES)

    const panel = await screen.findByRole("dialog", { name: "Request" })
    expect(
      within(panel).getByRole("button", { name: "Copy request ID" }),
    ).toBeInTheDocument()
  })

  it("switches to the request log and keeps the switch", async () => {
    mockApi({ traces: [] })

    renderPage(<ActivityHub />, "/activity", WITH_TRACES)

    await userEvent.click(
      await screen.findByRole("radio", { name: "Requests" }),
    )

    expect(await screen.findByText(/A per-request log/)).toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "Requests" })).toBeChecked()
  })

  it("scopes the sessions, their count and the chart to the selected workspace", async () => {
    const { calls } = mockApi({ traces: [traceSummary()], workspace: "ws-1" })

    renderPage(<ActivityHub />, "/activity", WITH_TRACES)

    await screen.findByRole("grid", { name: "Sessions" })
    const reads = (path: string) =>
      calls
        .filter((call) => call.method === "GET")
        .map((call) => new URL(call.url, "http://x"))
        .filter((url) => url.pathname.endsWith(path))
    await vi.waitFor(() => {
      for (const path of ["/traces", "/traces/count", "/traces/series"]) {
        const scoped = reads(path)
        expect(scoped.length).toBeGreaterThan(0)
        expect(scoped.at(-1)?.searchParams.get("workspace_id")).toBe("ws-1")
      }
    })
  })

  it("explains how a client's requests become one session when there are none", async () => {
    mockApi({ traces: [] })

    renderPage(<ActivityHub />, "/activity", WITH_TRACES)

    expect(await screen.findByText(/sends a session id/)).toBeInTheDocument()
  })
})

describe("ActivityHub links", () => {
  it("opens the request log a link names, and keeps it once its filters are cleared", async () => {
    mockApi({ traces: [] })

    renderPage(
      <ActivityHub />,
      "/activity?view=requests&status=error",
      WITH_TRACES,
    )

    expect(await screen.findByText(/A per-request log/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole("button", { name: "Clear all" }))
    expect(screen.getByRole("radio", { name: "Requests" })).toBeChecked()
    expect(screen.getByText(/A per-request log/)).toBeInTheDocument()
  })

  it("opens on sessions when a link names no view, whatever filters it carries", async () => {
    mockApi({ traces: [] })

    renderPage(<ActivityHub />, "/activity?status=error", WITH_TRACES)

    expect(await screen.findByRole("radio", { name: "Sessions" })).toBeChecked()
  })
})
