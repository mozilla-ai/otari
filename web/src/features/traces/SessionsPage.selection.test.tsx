import { useNavigate } from "@tanstack/react-router"
import { screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { SessionsPage } from "@/features/traces/SessionsPage"
import { jsonResponse, mockApi, renderPage } from "@/tests/activity"
import { traceSummary } from "@/tests/traceFixtures"

const START = "2026-10-01T00:00:00.000Z"
const END = "2026-10-01T00:30:00.000Z"

// The chart's drag is the timeline's own concern; a button stands in for it.
vi.mock("@/features/activity/ActivityTimeline", () => ({
  ActivityTimeline: ({
    onSelectRange,
  }: {
    onSelectRange: (start: string, end: string) => void
  }) => (
    <button type="button" onClick={() => onSelectRange(START, END)}>
      Select a bucket
    </button>
  ),
}))

afterEach(() => {
  vi.restoreAllMocks()
})

describe("SessionsPage timeline selection", () => {
  it("opens the session when the selected window holds exactly one", async () => {
    mockApi({ traces: [traceSummary()] })

    renderPage(<SessionsPage />, "/sessions")
    await userEvent.click(await screen.findByText("Select a bucket"))

    expect(
      await screen.findByRole("complementary", { name: "Trace claude-code" }),
    ).toBeInTheDocument()
  })

  it("leaves the list alone when the window holds several", async () => {
    mockApi({
      traces: [
        traceSummary(),
        traceSummary({ trace_id: "trace-b", name: "other" }),
      ],
    })

    renderPage(<SessionsPage />, "/sessions")
    await userEvent.click(await screen.findByText("Select a bucket"))

    await screen.findByRole("grid", { name: "Sessions" })
    expect(
      screen.queryByRole("complementary", { name: /Trace/ }),
    ).not.toBeInTheDocument()
  })

  it("falls back to the preset when a link's start date does not parse", async () => {
    mockApi({ traces: [traceSummary()] })

    renderPage(<SessionsPage />, "/sessions?start_date=not-a-date")

    expect(
      await screen.findByRole("grid", { name: "Sessions" }),
    ).toBeInTheDocument()
  })

  it("does not open a lone session just because the page opened on one", async () => {
    mockApi({ traces: [traceSummary()] })

    renderPage(<SessionsPage />, "/sessions?status=error")

    await screen.findByRole("grid", { name: "Sessions" })
    expect(
      screen.queryByRole("complementary", { name: /Trace/ }),
    ).not.toBeInTheDocument()
  })

  it("drops a selection whose list failed, so a later refresh opens nothing", async () => {
    const { mock } = mockApi({ traces: [traceSummary()] })
    const answer = mock.getMockImplementation()
    let failing = true
    mock.mockImplementation(async (input, init) => {
      const url = String(input)
      const isList =
        /\/traces\?/.test(url) && !/\/traces\/(count|series)/.test(url)
      if (failing && isList) return jsonResponse({ detail: "boom" }, 500)
      return answer?.(input, init) ?? jsonResponse([])
    })
    function Recover() {
      return (
        <button
          type="button"
          onClick={() => {
            failing = false
          }}
        >
          Recover
        </button>
      )
    }

    renderPage(
      <>
        <SessionsPage />
        <Recover />
      </>,
      "/sessions",
    )
    await userEvent.click(await screen.findByText("Select a bucket"))
    await vi.waitFor(() =>
      expect(screen.getByRole("alert")).toBeInTheDocument(),
    )
    await userEvent.click(screen.getByText("Recover"))
    await userEvent.click(screen.getByRole("button", { name: /refresh/i }))

    await screen.findByRole("grid", { name: "Sessions" })
    expect(
      screen.queryByRole("complementary", { name: /Trace/ }),
    ).not.toBeInTheDocument()
  })

  it("drops a selection whose list failed, so a later filter change opens nothing", async () => {
    const { mock } = mockApi({ traces: [traceSummary()] })
    const answer = mock.getMockImplementation()
    let failing = true
    mock.mockImplementation(async (input, init) => {
      const url = String(input)
      const isList =
        /\/traces\?/.test(url) && !/\/traces\/(count|series)/.test(url)
      if (failing && isList) return jsonResponse({ detail: "boom" }, 500)
      return answer?.(input, init) ?? jsonResponse([])
    })
    function Narrow() {
      const navigate = useNavigate()
      return (
        <button
          type="button"
          onClick={() => {
            failing = false
            void navigate({
              to: "/sessions",
              search: { start_date: START, end_date: END, status: "error" },
            })
          }}
        >
          Narrow by status
        </button>
      )
    }

    renderPage(
      <>
        <SessionsPage />
        <Narrow />
      </>,
      "/sessions",
    )
    await userEvent.click(await screen.findByText("Select a bucket"))
    await vi.waitFor(() =>
      expect(screen.getByRole("alert")).toBeInTheDocument(),
    )
    await userEvent.click(screen.getByText("Narrow by status"))

    await screen.findByRole("grid", { name: "Sessions" })
    expect(
      screen.queryByRole("complementary", { name: /Trace/ }),
    ).not.toBeInTheDocument()
  })
})
