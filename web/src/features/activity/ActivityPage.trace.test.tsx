import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ActivityPage } from "@/features/activity/ActivityPage"
import { entry, mockApi, renderPage } from "@/tests/activity"
import { SURFACES } from "@/tests/fixtures"
import { traceSummary } from "@/tests/traceFixtures"

afterEach(() => {
  vi.restoreAllMocks()
})

const WITH_TRACES = { surfaces: [...SURFACES, "traces"] }

const lookups = (calls: ReturnType<typeof mockApi>["calls"]) =>
  calls.filter((call) => call.url.includes("/traces/requests/"))

async function openRow() {
  const user = userEvent.setup()
  await user.click((await screen.findByText("gpt-4o")).closest("tr")!)
}

describe("ActivityPage trace link", () => {
  it("links a traced request to its session, opened on this row's LLM call", async () => {
    const { calls } = mockApi({
      rows: [entry({ id: "attempt-1", request_group_id: "req-1" })],
      requestTrace: traceSummary(),
    })
    renderPage(<ActivityPage />, "/activity", WITH_TRACES)

    await openRow()
    const link = await screen.findByRole("link", { name: "View trace" })

    const href = new URL(link.getAttribute("href")!, "http://localhost")
    expect(href.pathname).toBe("/sessions")
    expect(Object.fromEntries(href.searchParams)).toEqual({
      peek: traceSummary().trace_id,
      peek_workspace: traceSummary().workspace_id,
      span: "attempt-1",
    })
    expect(lookups(calls).map((call) => call.url)).toEqual([
      expect.stringMatching(/\/traces\/requests\/req-1$/),
    ])
  })

  it("offers no link for a request with no trace stored", async () => {
    const { calls } = mockApi({
      rows: [entry({ request_group_id: "req-1" })],
    })
    renderPage(<ActivityPage />, "/activity", WITH_TRACES)

    await openRow()

    await waitFor(() => expect(lookups(calls)).toHaveLength(1))
    expect(screen.getByText("Request detail")).toBeInTheDocument()
    expect(
      screen.queryByRole("link", { name: "View trace" }),
    ).not.toBeInTheDocument()
  })

  it("asks nothing of a deployment that records no traces", async () => {
    const { calls } = mockApi({
      rows: [entry({ request_group_id: "req-1" })],
      requestTrace: traceSummary(),
    })
    renderPage(<ActivityPage />)

    await openRow()

    expect(await screen.findByText("Request detail")).toBeInTheDocument()
    expect(lookups(calls)).toEqual([])
    expect(
      screen.queryByRole("link", { name: "View trace" }),
    ).not.toBeInTheDocument()
  })
})
