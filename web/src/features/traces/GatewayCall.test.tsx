import { screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { entry, mockApi, renderPage } from "@/tests/activity"
import { GatewayCall } from "./GatewayCall"

const START = Date.parse("2026-10-23T10:00:00Z")

function renderCall() {
  return renderPage(
    <GatewayCall
      attemptId="row-2"
      requestGroupId="group-1"
      startMs={START}
      endMs={START + 5_000}
      summary={<p>The span's own summary</p>}
    />,
    "/sessions",
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("GatewayCall", () => {
  it("shows the call's own usage row, not another attempt of its request", async () => {
    mockApi({
      groupRows: [
        entry({ id: "row-1", request_group_id: "group-1", model: "first" }),
        entry({ id: "row-2", request_group_id: "group-1", model: "served" }),
      ],
    })

    renderCall()

    expect(await screen.findByText("served")).toBeInTheDocument()
    expect(screen.queryByText("first")).toBeNull()
    expect(screen.queryByText("The span's own summary")).toBeNull()
  })

  it("falls back to the span's summary and the log when the row is out of reach", async () => {
    mockApi({ groupRows: [] })

    renderCall()

    expect(
      await screen.findByRole("link", { name: "View request" }),
    ).toBeInTheDocument()
    expect(screen.getByText("The span's own summary")).toBeInTheDocument()
  })
})
