import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ActivityPage } from "@/features/activity/ActivityPage"
import { API_ROOT } from "@/shared/api/client"
import {
  entry,
  jsonResponse,
  listCalls,
  mockApi,
  operatorContext,
  renderPage,
} from "@/tests/activity"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ActivityPage", () => {
  it("renders a request row with humanized latency, tokens, and status", async () => {
    mockApi({
      rows: [entry({ total_tokens: 1500, latency_ms: 842, cost: 0.0123 })],
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("gpt-4o")).closest("tr")!
    expect(within(row).getByText("1,500")).toBeInTheDocument()
    expect(within(row).getByText("842 ms")).toBeInTheDocument()
    expect(within(row).getByText("$0.0123")).toBeInTheDocument()
    // Status is a dot plus an uppercase word now, not a pill.
    expect(within(row).getByText("Success")).toBeInTheDocument()
  })

  it("shows the api key column, and an em-dash for master-key rows", async () => {
    mockApi({
      rows: [
        entry({ id: "g", model: "gateway-model", api_key_id: "key-1" }),
        entry({ id: "x", model: "imported-model", api_key_id: null }),
      ],
    })
    renderPage(<ActivityPage />)

    const importedRow = (await screen.findByText("imported-model")).closest(
      "tr",
    )!
    expect(within(importedRow).getByText("—")).toBeInTheDocument()
  })

  it("renders latency over a second as seconds and null latency as an em-dash", async () => {
    mockApi({
      rows: [
        entry({ id: "a", model: "slow-model", latency_ms: 1420 }),
        entry({ id: "b", model: "batch-model", latency_ms: null }),
      ],
    })
    renderPage(<ActivityPage />)

    const slow = (await screen.findByText("slow-model")).closest("tr")!
    expect(within(slow).getByText("1.42 s")).toBeInTheDocument()
    const batch = screen.getByText("batch-model").closest("tr")!
    expect(within(batch).getByText("—")).toBeInTheDocument()
  })

  it("opens an error row's detail and shows the diagnostic with its status code", async () => {
    const user = userEvent.setup()
    mockApi({
      rows: [
        entry({
          status: "error",
          error_message: "provider exploded: quota exceeded",
          status_code: 502,
        }),
      ],
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("gpt-4o")).closest("tr")!
    expect(within(row).getByText("Error")).toBeInTheDocument()

    await user.click(row)
    // The dashboard is admin-only, so the stored error text is shown verbatim,
    // with the classifying HTTP status alongside the "Error" heading.
    expect(
      screen.getByText("provider exploded: quota exceeded"),
    ).toBeInTheDocument()
    expect(screen.getByText("Error (502)")).toBeInTheDocument()
  })

  it("omits the status code from the error heading when none was recorded", async () => {
    const user = userEvent.setup()
    mockApi({
      rows: [
        entry({
          status: "error",
          error_message: "stream completed without usage data",
          status_code: null,
        }),
      ],
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("gpt-4o")).closest("tr")!
    await user.click(row)
    expect(
      screen.getByText("stream completed without usage data"),
    ).toBeInTheDocument()
    // Bare heading, no "(code)" suffix. Scoped to the overline, which is the
    // heading's own class: a plain span now also matches the status filter's
    // <option>Error</option> and the row's own status word, which reads "Error"
    // rather than "ERROR" since it took a label map.
    expect(
      screen.getByText("Error", { selector: "span.text-overline" }),
    ).toBeInTheDocument()
    expect(screen.queryByText(/Error \(/)).not.toBeInTheDocument()
  })

  it("copies a request id out of the detail panel", async () => {
    // The id an operator pastes into a log search or a support thread, where a
    // mistyped character makes it useless.
    const user = userEvent.setup()
    mockApi({ rows: [entry({ id: "3ba12b77-8841-42a5-b776-a0a1aacb347f" })] })
    renderPage(<ActivityPage />)

    await user.click((await screen.findByText("gpt-4o")).closest("tr")!)
    await user.click(screen.getByRole("button", { name: "Copy request id" }))

    expect(await navigator.clipboard.readText()).toBe(
      "3ba12b77-8841-42a5-b776-a0a1aacb347f",
    )
  })

  it("opens the detail inline directly under the clicked row, and Close collapses it", async () => {
    // Regression: the shared-table migration rendered the detail below the
    // whole table, so on a full page a row click looked like it did nothing.
    const user = userEvent.setup()
    mockApi({
      rows: [entry({ id: "r1" }), entry({ id: "r2", model: "gpt-4o-mini" })],
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("gpt-4o")).closest("tr")!
    await user.click(row)

    expect(screen.getByText("Request detail")).toBeInTheDocument()
    // The panel is the row's next sibling (accordion), not a card after the table.
    expect(row.nextElementSibling?.textContent).toContain("Request detail")

    await user.click(screen.getByRole("button", { name: "Close" }))
    expect(screen.queryByText("Request detail")).not.toBeInTheDocument()
  })

  it("shows a row's token composition rather than one uninformative total", async () => {
    // A cached agent request: the total is ~98% cache read, so the total alone
    // makes every row look alike. The bar carries the split.
    mockApi({
      rows: [
        entry({
          prompt_tokens: 100_000,
          completion_tokens: 500,
          total_tokens: 100_500,
          cache_read_tokens: 98_000,
          cache_write_tokens: 1_500,
          billing_meters: {
            total_input_tokens: 100_000,
            fresh_input_tokens: 500,
            cache_read_tokens: 98_000,
            cache_write_tokens: 1_500,
            cache_write_1h_tokens: 0,
            completion_tokens: 500,
          },
        }),
      ],
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("gpt-4o")).closest("tr")!
    expect(within(row).getByText("100,500")).toBeInTheDocument()
    const bar = within(row).getByRole("img", { name: /Token composition/ })
    expect(bar).toHaveAccessibleName(
      "Token composition: Fresh input 500, Cache read 98,000, Cache write 1,500, Output 500",
    )

    // Four segments, widest being the cache read, so the shape is what the eye
    // compares between rows.
    const widths = [...bar.querySelectorAll("rect")].map((r) =>
      Number(r.getAttribute("width")),
    )
    expect(widths).toHaveLength(4)
    expect(Math.max(...widths)).toBeCloseTo((98_000 / 100_500) * 100, 5)
    expect(widths.reduce((a, b) => a + b, 0)).toBeCloseTo(100, 5)
  })

  it("explains the column's total in the detail panel when it exceeds the raw one", async () => {
    // An additive-convention row reports its cache buckets outside the prompt, so
    // the billed total the column shows is far above the stored `total_tokens`.
    // Both are spelled out, or the two numbers look like a bug.
    const user = userEvent.setup()
    mockApi({
      rows: [
        entry({
          prompt_tokens: 1_000,
          completion_tokens: 200,
          total_tokens: 1_200,
          cache_read_tokens: 98_000,
          cache_write_tokens: 1_500,
          billing_meters: {
            total_input_tokens: 100_500,
            fresh_input_tokens: 1_000,
            cache_read_tokens: 98_000,
            cache_write_tokens: 1_500,
            cache_write_1h_tokens: 0,
            completion_tokens: 200,
          },
        }),
      ],
    })
    renderPage(<ActivityPage />)

    const row = (await screen.findByText("gpt-4o")).closest("tr")!
    await user.click(row)

    const field = (label: string): string =>
      screen.getByText(label).parentElement!.textContent ?? ""
    expect(field("Total tokens")).toContain("1,200")
    expect(field("Billed tokens")).toContain("100,700")
    // Which is the number the row itself shows.
    expect(within(row).getByText("100,700")).toBeInTheDocument()
  })

  it("splits an unmetered row from its raw columns, and shows no bar without usage", async () => {
    // An unpriced row carries no billing meters, so the composition falls back to
    // the raw columns (cache read counted inside the prompt).
    mockApi({
      rows: [
        entry({
          id: "unpriced",
          model: "unpriced-model",
          prompt_tokens: 1_000,
          completion_tokens: 200,
          total_tokens: 1_200,
          cache_read_tokens: 400,
          billing_meters: null,
          cost: null,
        }),
        entry({
          id: "failed",
          model: "failed-model",
          status: "error",
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null,
        }),
      ],
    })
    renderPage(<ActivityPage />)

    const unpriced = (await screen.findByText("unpriced-model")).closest("tr")!
    expect(
      within(unpriced).getByRole("img", { name: /Token composition/ }),
    ).toHaveAccessibleName(
      "Token composition: Fresh input 600, Cache read 400, Output 200",
    )

    // A request that failed before the provider reported usage has nothing to
    // compose, so the cell stays an em-dash instead of drawing an empty bar.
    const failed = screen.getByText("failed-model").closest("tr")!
    expect(
      within(failed).queryByRole("img", { name: /Token composition/ }),
    ).not.toBeInTheDocument()
    expect(within(failed).getByText("—")).toBeInTheDocument()
  })

  it("opens a bookmarked deep page on that page", async () => {
    // The URL is the source of truth for `page`, but the mount effect used to
    // re-anchor the rolling window a few milliseconds later, which changed the
    // filter set and reset the page: every shared `?page=3` link opened on page 1.
    const { calls } = mockApi({
      rows: Array.from({ length: 50 }, (_, i) => entry({ id: `r${i}` })),
      total: 500,
    })
    renderPage(<ActivityPage />, "/activity?page=2")

    await screen.findAllByText("gpt-4o")
    expect(await screen.findByText("101–150 of 500")).toBeInTheDocument()
    expect(listCalls(calls).every((url) => url.includes("skip=100"))).toBe(true)
  })

  it("keeps the current page when refreshing", async () => {
    // Refresh used to re-anchor the rolling window, which changed the filter set,
    // which reset the page: pressing it on page 3 dropped you back to page 1.
    const { calls } = mockApi({
      rows: Array.from({ length: 50 }, (_, i) => entry({ id: `r${i}` })),
      total: 500,
    })
    const user = userEvent.setup()
    renderPage(<ActivityPage />, "/activity?page=2")
    await screen.findAllByText("gpt-4o")
    expect(await screen.findByText("101–150 of 500")).toBeInTheDocument()

    const before = listCalls(calls).length
    const entitySummaryBefore = calls.filter(
      (call) =>
        call.url.includes(`${API_ROOT}/usage/summary`) &&
        call.url.includes("dimensions=user"),
    ).length
    const button = screen.getByRole("button", { name: "Refresh" })
    await waitFor(() => expect(button).toBeEnabled())
    await user.click(button)

    // The list is refetched, and every fetch stays on the third page's offset.
    await waitFor(() => expect(listCalls(calls).length).toBeGreaterThan(before))
    await waitFor(() =>
      expect(
        calls.filter(
          (call) =>
            call.url.includes(`${API_ROOT}/usage/summary`) &&
            call.url.includes("dimensions=user"),
        ).length,
      ).toBeGreaterThan(entitySummaryBefore),
    )
    expect(listCalls(calls).every((url) => url.includes("skip=100"))).toBe(true)
    expect(screen.getByText("101–150 of 500")).toBeInTheDocument()
  })

  it("distinguishes filtered-empty from never-used", async () => {
    const user = userEvent.setup()
    mockApi({ rows: [], total: 0 })
    renderPage(<ActivityPage />)

    // The default 24h preset is not itself a filter (mirroring UsagePage), so an
    // empty result on a brand-new gateway reads as "never used", not "filtered".
    expect(
      await screen.findByText("No requests recorded yet."),
    ).toBeInTheDocument()

    // The unbounded "All" applies no window either, so it stays "never used"
    // rather than flipping to filtered-empty.
    await user.click(screen.getByRole("button", { name: "All" }))
    expect(
      await screen.findByText("No requests recorded yet."),
    ).toBeInTheDocument()

    // Narrowing to a bounded non-default preset is a real time filter, so an empty
    // result then reads as filtered-to-empty.
    await user.click(screen.getByRole("button", { name: "7d" }))
    expect(
      await screen.findByText("No requests match these filters."),
    ).toBeInTheDocument()
  })

  it("keeps Next reachable when the count request fails", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) return operatorContext()
      if (url.includes(`${API_ROOT}/usage/count`)) {
        return jsonResponse({ detail: "boom" }, 500)
      }
      if (url.includes(`${API_ROOT}/usage/summary`)) {
        return jsonResponse({
          by_model: [],
          by_user: [],
          by_api_key: [],
          series: [],
        })
      }
      if (url.includes(`${API_ROOT}/usage/in-flight`)) {
        return jsonResponse({ requests: [], total: 0 })
      }
      if (url.includes(`${API_ROOT}/usage`)) {
        return jsonResponse(
          Array.from({ length: 50 }, (_, i) => entry({ id: `r${i}` })),
        )
      }
      return jsonResponse([])
    })
    renderPage(<ActivityPage />)

    await screen.findAllByText("gpt-4o")
    expect(screen.getByRole("button", { name: "Next page" })).toBeEnabled()
    expect(screen.getByText("1–50")).toBeInTheDocument()
    expect(screen.queryByText("0 of 0")).not.toBeInTheDocument()
  })

  it("snaps URL-supplied page sizes to the nearest offered option", async () => {
    // An old size=500 bookmark must not resurrect second-long selection
    // clicks, and a hand-edited size=-5 must not reach the API as a bad limit.
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />, "/activity?size=500")

    await screen.findByText("gpt-4o")
    expect(listCalls(calls).at(-1)).toContain("limit=100")

    const { calls: negativeCalls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />, "/activity?size=-5")
    await waitFor(() =>
      expect(listCalls(negativeCalls).length).toBeGreaterThan(0),
    )
    expect(listCalls(negativeCalls).at(-1)).toContain("limit=25")
  })

  it("shows the paginator range and total", async () => {
    mockApi({
      rows: Array.from({ length: 50 }, (_, i) => entry({ id: `r${i}` })),
      total: 120,
    })
    renderPage(<ActivityPage />)

    expect(await screen.findByText("1–50 of 120")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Previous page" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Next page" })).toBeEnabled()
  })
})
