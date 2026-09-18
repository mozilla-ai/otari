import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ActivityPage } from "@/features/activity/ActivityPage"
import { API_ROOT } from "@/shared/api/client"
import { entry, listCalls, mockApi, renderPage } from "@/tests/activity"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ActivityPage time range and timeline", () => {
  it("re-anchors a rolling window when its preset is re-picked", async () => {
    // Refresh no longer moves the window, so re-selecting the active preset is the
    // gesture that advances a rolling range to "now".
    const { calls } = mockApi({ rows: [entry()] })
    const user = userEvent.setup()
    renderPage(<ActivityPage />)
    await screen.findByText("gpt-4o")

    const startOf = (url: string): string | null =>
      new URL(url, "http://x").searchParams.get("start_date")
    const before = startOf(listCalls(calls).at(-1)!)
    expect(before).not.toBeNull()

    await user.click(screen.getByRole("button", { name: "24h" }))

    await waitFor(() =>
      expect(startOf(listCalls(calls).at(-1)!)).not.toBe(before),
    )
  })

  it("queries an unbounded window for the truthful 'All' preset", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />)
    await screen.findByText("gpt-4o")

    await user.click(screen.getByRole("button", { name: "All" }))

    // Activity's list endpoint applies no default lookback, so "All" really omits
    // the start bound rather than silently scoping to a recent window.
    await waitFor(() =>
      expect(listCalls(calls).some((url) => !url.includes("start_date"))).toBe(
        true,
      ),
    )

    // The histogram, however, sends an explicit start bound: without one the
    // summary endpoint would apply a hidden 30-day default, so the bars would show
    // a rolling month while the caption reads "All time". The list stays all-time.
    expect(
      calls.some(
        (c) =>
          c.url.includes(`${API_ROOT}/usage/summary`) &&
          c.url.includes("start_date="),
      ),
    ).toBe(true)
  })

  it("rewrites an unrecognized range to the one it actually applied", async () => {
    mockApi({ rows: [entry()] })
    // `90d` is a Usage preset and not an Activity one, so a URL copied between
    // the two pages arrives with a range this page cannot honor. It falls back
    // to the default window either way; the point here is that the address bar
    // stops claiming ninety days over a list showing one, which the preset row
    // reads back: no tab is pressed while the bogus key stands.
    renderPage(<ActivityPage />, "/activity?range=90d")
    await screen.findByText("gpt-4o")

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "24h" })).toHaveAttribute(
        "aria-pressed",
        "true",
      ),
    )
  })

  it("leaves an unrecognized range alone when explicit bounds are set", async () => {
    mockApi({ rows: [entry()] })
    // A drill-down carries its own window, so the range is not being read and
    // is not lying about anything. Rewriting it here would fight the bounds.
    renderPage(
      <ActivityPage />,
      "/activity?range=90d&start_date=2026-08-01T00:00:00.000Z",
    )
    await screen.findByText("gpt-4o")

    expect(
      screen.queryByRole("button", { name: "24h", pressed: true }),
    ).not.toBeInTheDocument()
  })

  it("gives the histogram an explicit start for the custom-range sentinel", async () => {
    const { calls } = mockApi({ rows: [entry()] })
    // `?range=custom` has no rolling window of its own, so without an explicit
    // extent the summary would fall back to the server's hidden 30-day default.
    renderPage(<ActivityPage />, "/activity?range=custom")
    await screen.findByText("gpt-4o")

    await waitFor(() =>
      expect(
        calls.some(
          (c) =>
            c.url.includes(`${API_ROOT}/usage/summary`) &&
            c.url.includes("start_date="),
        ),
      ).toBe(true),
    )
  })

  it("frames a drill-down window that reaches outside the preset extent", async () => {
    const { calls } = mockApi({ rows: [entry()] })
    // A Usage-page drill-down: explicit multi-week bounds while `range` stays the
    // 24h default. The timeline must frame the drilled window (daily buckets over
    // its bounds), not the unrelated 24h extent.
    renderPage(
      <ActivityPage />,
      "/activity?start_date=2020-07-01T00:00:00.000Z&end_date=2020-07-15T00:00:00.000Z",
    )
    await screen.findByText("gpt-4o")

    await waitFor(() =>
      expect(
        calls.some(
          (c) =>
            c.url.includes(`${API_ROOT}/usage/summary`) &&
            c.url.includes("bucket=day") &&
            c.url.includes("start_date=2020-07-01") &&
            c.url.includes("end_date=2020-07-15"),
        ),
      ).toBe(true),
    )
    // The caption reflects the drilled window (end shown inclusively). Assert on
    // day numbers and the UTC marker, not a month abbreviation, since the caption
    // formats with the runtime locale ("Jul" would fail outside en-US).
    const caption = (screen.getByText(/Showing/).textContent ?? "").replace(
      /\s+/g,
      " ",
    )
    expect(caption).toMatch(/\b1\b/)
    expect(caption).toMatch(/\b14\b/)
    expect(caption).toContain("UTC")
  })

  it("buckets the timeline histogram by the active preset's extent", async () => {
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />) // default 24h
    await screen.findByText("gpt-4o")

    // The 24h extent buckets hourly, so the timeline's context summary is fetched
    // with bucket=hour (distinct from the day-bucketed model-suggestion summary).
    await waitFor(() =>
      expect(
        calls.some(
          (c) =>
            c.url.includes(`${API_ROOT}/usage/summary`) &&
            c.url.includes("bucket=hour"),
        ),
      ).toBe(true),
    )
  })

  it("frames the active preset when the two initial windows are read a millisecond apart", async () => {
    // `win` and `extentWin` each derive a rolling start from their own clock read,
    // and `extentWin` is initialized second, so its start is the later of the two
    // whenever the render straddles a millisecond. That is not a drill-down, and
    // must not be read as one: doing so framed the window instead of the preset,
    // dropping the preset highlight and bucketing a 24h extent by day. A monotonic
    // clock makes the tick certain instead of leaving it to machine load, which is
    // what made the assertion above flake on CI while passing locally.
    let now = Date.parse("2026-08-11T12:00:00.000Z")
    vi.spyOn(Date, "now").mockImplementation(() => ++now)

    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />) // default 24h
    await screen.findByText("gpt-4o")

    await waitFor(() =>
      expect(
        calls.some(
          (c) =>
            c.url.includes(`${API_ROOT}/usage/summary`) &&
            c.url.includes("bucket=hour"),
        ),
      ).toBe(true),
    )
    // The visible half of the same bug: the preset row still marks 24h active
    // rather than falling back to the custom sentinel, which marks nothing. The
    // presets are tabs now, so the assertion is on the state a tab reports
    // rather than on a button variant class.
    expect(screen.getByRole("button", { name: "24h" })).toHaveAttribute(
      "aria-pressed",
      "true",
    )
  })
})
