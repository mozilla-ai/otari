import { useNavigate, useRouter } from "@tanstack/react-router"
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { SessionsPage } from "@/features/traces/SessionsPage"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { mockApi, renderPage } from "@/tests/activity"
import { recordEvent, resetTelemetrySpy } from "@/tests/telemetry"
import { traceDetail, traceSummary } from "@/tests/traceFixtures"

vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: vi.fn(() => telemetrySpy) }
})

// The chart's drag is the timeline's own concern; a button stands in for it.
vi.mock("@/features/activity/ActivityTimeline", () => ({
  ActivityTimeline: ({
    onSelectRange,
  }: {
    onSelectRange: (start: string, end: string) => void
  }) => (
    <button
      type="button"
      onClick={() =>
        onSelectRange("2026-10-01T00:00:00.000Z", "2026-10-01T00:30:00.000Z")
      }
    >
      Select a bucket
    </button>
  ),
}))

beforeEach(() => {
  resetTelemetrySpy()
})

afterEach(() => {
  vi.restoreAllMocks()
})

function opened() {
  return recordEvent.mock.calls.filter(
    ([event]) => event === TELEMETRY_EVENTS.SESSION_OPENED,
  )
}

describe("SessionsPage telemetry", () => {
  it("records a session opened from the list, and again when it is reopened", async () => {
    const user = userEvent.setup()
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<SessionsPage />, "/sessions")
    const sessions = await screen.findByRole("grid", { name: "Sessions" })
    await user.click(
      await within(sessions).findByRole("row", { name: /claude-code/ }),
    )
    const peek = await screen.findByRole("complementary", {
      name: "Trace claude-code",
    })
    await user.click(within(peek).getByRole("button", { name: "Close" }))
    await waitFor(() =>
      expect(
        screen.queryByRole("complementary", { name: "Trace claude-code" }),
      ).not.toBeInTheDocument(),
    )
    await user.click(
      await within(sessions).findByRole("row", { name: /claude-code/ }),
    )
    await screen.findByRole("complementary", { name: "Trace claude-code" })

    // The session's id is never sent: it is unbounded.
    expect(opened()).toEqual([
      [TELEMETRY_EVENTS.SESSION_OPENED, { source: "list" }],
      [TELEMETRY_EVENTS.SESSION_OPENED, { source: "list" }],
    ])
  })

  it("records a session an address already named as opened by a link", async () => {
    const summary = traceSummary()
    mockApi({ traces: [], traceDetail: traceDetail() })

    renderPage(
      <SessionsPage />,
      `/sessions?peek=${summary.trace_id}&peek_workspace=${summary.workspace_id}`,
    )
    await screen.findByRole("complementary", { name: "Trace claude-code" })

    expect(opened()).toEqual([
      [TELEMETRY_EVENTS.SESSION_OPENED, { source: "link" }],
    ])
  })

  it("records a session a timeline bar opened", async () => {
    mockApi({ traces: [traceSummary()], traceDetail: traceDetail() })

    renderPage(<SessionsPage />, "/sessions")
    await userEvent.click(await screen.findByText("Select a bucket"))
    await screen.findByRole("complementary", { name: "Trace claude-code" })

    expect(opened()).toEqual([
      [TELEMETRY_EVENTS.SESSION_OPENED, { source: "timeline" }],
    ])
  })

  it("credits a link, not the timeline, for a session a bar found already open", async () => {
    const user = userEvent.setup()
    const summary = traceSummary()
    mockApi({ traces: [summary], traceDetail: traceDetail() })

    function LinkBack() {
      const navigate = useNavigate()
      return (
        <button
          type="button"
          onClick={() =>
            void navigate({
              to: "/sessions",
              search: {
                peek: summary.trace_id,
                peek_workspace: summary.workspace_id,
              },
            })
          }
        >
          Follow a link to it
        </button>
      )
    }

    renderPage(
      <>
        <SessionsPage />
        <LinkBack />
      </>,
      "/sessions",
    )
    const sessions = await screen.findByRole("grid", { name: "Sessions" })
    await user.click(
      await within(sessions).findByRole("row", { name: /claude-code/ }),
    )
    const peek = await screen.findByRole("complementary", {
      name: "Trace claude-code",
    })
    // The bar holds the one session already open, so nothing new opens.
    await user.click(screen.getByText("Select a bucket"))
    await user.click(within(peek).getByRole("button", { name: "Close" }))
    await waitFor(() =>
      expect(
        screen.queryByRole("complementary", { name: "Trace claude-code" }),
      ).not.toBeInTheDocument(),
    )
    await user.click(screen.getByText("Follow a link to it"))
    await screen.findByRole("complementary", { name: "Trace claude-code" })

    expect(opened()).toEqual([
      [TELEMETRY_EVENTS.SESSION_OPENED, { source: "list" }],
      [TELEMETRY_EVENTS.SESSION_OPENED, { source: "link" }],
    ])
  })

  it("credits a link, not the list, when Back overtook the press and left no session open", async () => {
    const user = userEvent.setup()
    const other = traceSummary({ trace_id: "s-other", name: "other" })
    const pressed = traceSummary()
    mockApi({ traces: [other, pressed], traceDetail: traceDetail() })
    const history: { back: () => void } = { back: () => {} }

    function Navigation() {
      const navigate = useNavigate()
      history.back = useRouter().history.back
      const open = (id: string) =>
        void navigate({
          to: "/sessions",
          search: { peek: id, peek_workspace: pressed.workspace_id },
        })
      return (
        <>
          <button type="button" onClick={() => open(other.trace_id)}>
            Link to the other session
          </button>
          <button type="button" onClick={() => open(pressed.trace_id)}>
            Link to the pressed session
          </button>
        </>
      )
    }

    renderPage(
      <>
        <SessionsPage />
        <Navigation />
      </>,
      "/sessions",
    )
    const sessions = await screen.findByRole("grid", { name: "Sessions" })
    const row = await within(sessions).findByRole("row", {
      name: /claude-code/,
    })
    // A pushed entry with a session open, so Back has somewhere to go that
    // holds none.
    await user.click(screen.getByText("Link to the other session"))
    await screen.findByRole("complementary", { name: /Trace/ })
    // Pressed, and Back taken before the press's navigation lands.
    act(() => {
      row.focus()
      fireEvent.keyDown(row, { key: "Enter" })
      fireEvent.keyUp(row, { key: "Enter" })
      history.back()
    })
    await waitFor(() =>
      expect(
        screen.queryByRole("complementary", { name: /Trace/ }),
      ).not.toBeInTheDocument(),
    )
    await user.click(screen.getByText("Link to the pressed session"))
    await screen.findByRole("complementary", { name: /Trace/ })

    expect(opened()).toEqual([
      [TELEMETRY_EVENTS.SESSION_OPENED, { source: "link" }],
      [TELEMETRY_EVENTS.SESSION_OPENED, { source: "link" }],
    ])
  })

  it("records nothing while no session is open", async () => {
    mockApi({ traces: [traceSummary()] })

    renderPage(<SessionsPage />, "/sessions")
    await screen.findByRole("grid", { name: "Sessions" })

    expect(opened()).toEqual([])
  })
})
