import { screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { renderWithRouter } from "@/tests/router"
import { RequestLogLink } from "./RequestLogLink"

describe("RequestLogLink", () => {
  it("opens the request on the log, in a window around the call", async () => {
    const start = Date.parse("2026-10-23T10:00:00Z")
    await renderWithRouter(
      <RequestLogLink
        requestId="row-1"
        startMs={start}
        endMs={start + 5_000}
      />,
    )

    const link = screen.getByRole("link", { name: "View request" })
    const href = new URL(link.getAttribute("href")!, "http://localhost")
    expect(href.pathname).toBe("/requests")
    expect(Object.fromEntries(href.searchParams)).toEqual({
      request: "row-1",
      start_date: "2026-10-23T09:59:00.000Z",
      end_date: "2026-10-23T10:01:05.000Z",
    })
  })
})
