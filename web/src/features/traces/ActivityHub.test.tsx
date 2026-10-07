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

    const sessions = await screen.findByRole("region", { name: "Sessions" })
    await userEvent.click(
      await within(sessions).findByRole("button", { name: /claude-code/ }),
    )

    expect(
      await screen.findByRole("region", { name: "Turn 1" }),
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

  it("explains how a client's requests become one session when there are none", async () => {
    mockApi({ traces: [] })

    renderPage(<ActivityHub />, "/activity", WITH_TRACES)

    expect(await screen.findByText(/sends a session id/)).toBeInTheDocument()
  })
})

describe("ActivityHub links", () => {
  it("opens the request log when a link carries its filters", async () => {
    mockApi({ traces: [] })

    renderPage(<ActivityHub />, "/activity?status=error", WITH_TRACES)

    expect(await screen.findByText(/A per-request log/)).toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "Requests" })).toBeChecked()
  })

  it("keeps an explicit choice of sessions over the filters a link carries", async () => {
    mockApi({ traces: [] })

    renderPage(
      <ActivityHub />,
      "/activity?status=error&view=sessions",
      WITH_TRACES,
    )

    expect(await screen.findByRole("radio", { name: "Sessions" })).toBeChecked()
  })
})
