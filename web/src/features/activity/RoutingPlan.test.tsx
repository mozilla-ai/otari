import { screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { entry, mockApi, renderPage } from "@/tests/activity"
import { flushRouter } from "@/tests/router"
import { RoutingPlan } from "./RoutingPlan"

afterEach(() => {
  vi.restoreAllMocks()
})

const failed = entry({
  id: "a",
  request_group_id: "grp-1",
  policy_name: "cheap-first",
  status: "absorbed",
  status_code: 429,
  attempt_position: 1,
  attempt_count: 2,
  selection_reason: "default",
  model: "gpt-4o-mini",
  cost: null,
})

const served = entry({
  id: "b",
  request_group_id: "grp-1",
  policy_name: "cheap-first",
  status: "success",
  attempt_position: 2,
  attempt_count: 2,
  selection_reason: "on_failure",
  model: "gpt-4o",
  latency_ms: 1_200,
  cost: 0.02,
})

function planTable() {
  return screen.getByRole("table", {
    name: "Routing plan for policy cheap-first",
  })
}

describe("RoutingPlan", () => {
  it("reassembles the whole plan from the group, in attempt order", async () => {
    mockApi({ groupRows: [served, failed] })
    renderPage(<RoutingPlan entry={failed} />)
    await flushRouter()

    expect(
      await screen.findByText("Served by attempt 2 of 2: openai:gpt-4o"),
    ).toBeInTheDocument()
    const rows = within(planTable()).getAllByRole("row").slice(1)
    expect(
      rows.map((row) => within(row).getAllByRole("cell")[0].textContent),
    ).toEqual(["1", "2"])
    expect(
      within(rows[0]).getByText("failed 429, fell back"),
    ).toBeInTheDocument()
    expect(within(rows[1]).getByText("served the request")).toBeInTheDocument()
  })

  it("marks which attempt the open row is", async () => {
    mockApi({ groupRows: [served, failed] })
    renderPage(<RoutingPlan entry={failed} />)
    await flushRouter()

    // Await the assembled plan, not the table: the fallback table renders the
    // single row it was handed while the group lookup is still in flight.
    await screen.findByText("Served by attempt 2 of 2: openai:gpt-4o")
    const rows = within(planTable()).getAllByRole("row").slice(1)
    expect(within(rows[0]).getByText("this row")).toBeInTheDocument()
    expect(within(rows[1]).queryByText("this row")).not.toBeInTheDocument()
  })

  it("says the request ended in an error when no candidate served", async () => {
    const terminal = entry({
      ...served,
      id: "b",
      status: "error",
      status_code: 500,
    })
    mockApi({ groupRows: [failed, terminal] })
    renderPage(<RoutingPlan entry={failed} />)
    await flushRouter()

    expect(
      await screen.findByText("No candidate served this request."),
    ).toBeInTheDocument()
  })

  it("narrates the row itself rather than flashing empty while the lookup runs", async () => {
    mockApi({ groupRows: [] })
    renderPage(<RoutingPlan entry={failed} />)
    await flushRouter()

    expect(
      screen.getByText("Loading the rest of this request's attempts…"),
    ).toBeInTheDocument()
    // The row it was given, so the section is never blank.
    expect(
      within(planTable()).getByText("openai:gpt-4o-mini"),
    ).toBeInTheDocument()
  })

  it("says so for a row that carries no group to look up", async () => {
    mockApi()
    renderPage(
      <RoutingPlan entry={entry({ ...failed, request_group_id: null })} />,
    )
    await flushRouter()

    expect(
      screen.getByText(
        "This row carries no request group, so its other attempts cannot be found.",
      ),
    ).toBeInTheDocument()
  })
})
