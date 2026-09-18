import { act, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { InFlightRequest, InFlightResponse } from "@/client"
import { ActivityPage } from "@/features/activity/ActivityPage"
import { API_ROOT } from "@/shared/api/client"
import {
  countCalls,
  entry,
  jsonResponse,
  listCalls,
  mockApi,
  operatorContext,
  renderPage,
} from "@/tests/activity"

// ---------------------------------------------------------------------------
// Requests in flight, and the frozen log (issue #526)
// ---------------------------------------------------------------------------

function inFlightRequest(
  overrides: Partial<InFlightRequest> = {},
): InFlightRequest {
  return {
    id: "live-1",
    endpoint: "/v1/chat/completions",
    model: "ollama:qwen3",
    provider: "ollama",
    user_id: "alice",
    api_key_id: "key-1",
    policy_name: null,
    started_at: new Date().toISOString(),
    elapsed_ms: 12_000,
    ...overrides,
  }
}

// The live control, found by role: its label is assembled from the count and the
// word "in flight" as separate text nodes, so no single text node carries it.
function liveControl(): HTMLElement | null {
  return screen.queryByRole("button", { name: /in flight/ })
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ActivityPage live traffic", () => {
  it("reports what is running as a count beside refresh, not as rows in the log", async () => {
    // The reason the count exists: a usage row is only written once a request
    // settles, so on a slow backend the log stays empty for a whole 30s call and
    // reads as "nothing is happening".
    //
    // The reason it is not a row: the poll behind it runs every two seconds, and
    // rows that re-derived themselves on that timer reordered the top of the table
    // continuously on any gateway with real traffic.
    mockApi({
      rows: [entry()],
      inFlight: { requests: [inFlightRequest()], total: 1 },
    })
    renderPage(<ActivityPage />, "/activity?range=24h")

    await waitFor(() => expect(liveControl()).toBeInTheDocument())
    expect(liveControl()).toHaveAccessibleName(/1 in flight/)

    // One body row, the settled one. The live request is not among them.
    const bodyRows = screen.getAllByRole("row").slice(1)
    expect(bodyRows).toHaveLength(1)
    expect(within(bodyRows[0]).getByText("gpt-4o")).toBeInTheDocument()
    expect(screen.queryByText("ollama:qwen3")).not.toBeInTheDocument()
  })

  it("lists the running requests, with the wait so far, when the count is opened", async () => {
    const user = userEvent.setup()
    mockApi({
      rows: [],
      inFlight: {
        requests: [
          inFlightRequest({ policy_name: "cheap-first", elapsed_ms: 95_000 }),
        ],
        total: 1,
      },
    })
    renderPage(<ActivityPage />, "/activity?range=24h")

    await waitFor(() => expect(liveControl()).toBeInTheDocument())
    await user.click(liveControl()!)

    const panel = await screen.findByRole("dialog")
    expect(within(panel).getByText("ollama:qwen3")).toBeInTheDocument()
    expect(within(panel).getByText(/alice/)).toBeInTheDocument()
    expect(within(panel).getByText(/cheap-first/)).toBeInTheDocument()
    // Seeded from the server's own measurement, so the wait does not depend on the
    // browser clock agreeing with the gateway's, and long enough to read in minutes
    // because a stuck local model is the case this exists for.
    expect(within(panel).getByText(/^1m 35s$/)).toBeInTheDocument()
  })

  it("says how many running requests the response left out", async () => {
    // The endpoint caps what it serializes, so the count and the list can differ;
    // reading the list length as the total would under-report live traffic.
    const user = userEvent.setup()
    mockApi({ rows: [], inFlight: { requests: [inFlightRequest()], total: 7 } })
    renderPage(<ActivityPage />, "/activity?range=24h")

    await waitFor(() =>
      expect(liveControl()).toHaveAccessibleName(/7 in flight/),
    )
    await user.click(liveControl()!)

    const panel = await screen.findByRole("dialog")
    expect(
      within(panel).getByText(
        /6 further requests are in flight beyond the 1 listed/,
      ),
    ).toBeInTheDocument()
  })

  it("keeps an opened list open when the last request lands", async () => {
    // Otherwise the list an operator opened to watch a slow request is torn out
    // from under them at the moment it finishes, which is the moment they were
    // waiting for. It stays, reading "0 in flight", until they close it.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      let live: InFlightResponse = { requests: [inFlightRequest()], total: 1 }
      mockApi({ rows: [], inFlight: () => live })
      renderPage(<ActivityPage />, "/activity?range=24h")

      await waitFor(() => expect(liveControl()).toBeInTheDocument())
      // Held as a node: react-aria marks the rest of the page `aria-hidden` while
      // the popover is open, so the trigger is unreachable by role until it closes.
      const control = liveControl()!
      await user.click(control)
      const panel = await screen.findByRole("dialog")
      expect(within(panel).getByText("ollama:qwen3")).toBeInTheDocument()

      live = { requests: [], total: 0 }
      await vi.advanceTimersByTimeAsync(3_000)

      await waitFor(() => expect(control).toHaveTextContent(/0 in flight/))
      expect(screen.getByRole("dialog")).toBeInTheDocument()
      expect(screen.getByText(/Nothing running right now/)).toBeInTheDocument()

      // Closed by the operator, and only then does the control go.
      await user.keyboard("{Escape}")
      await waitFor(() => expect(liveControl()).not.toBeInTheDocument())
    } finally {
      vi.useRealTimers()
    }
  })

  it("shows no live control while the gateway is idle", async () => {
    mockApi({ rows: [entry()], inFlight: { requests: [], total: 0 } })
    renderPage(<ActivityPage />, "/activity?range=24h")

    await screen.findByText("gpt-4o")
    expect(liveControl()).not.toBeInTheDocument()
  })

  it("drops the live control when the in-flight poll starts failing", async () => {
    // TanStack keeps the last successful payload after a failed refetch, so without
    // an explicit error arm the count would sit there with its waits climbing
    // against a frozen anchor, claiming work is running that may have landed
    // minutes ago. That is the state the hook already refuses to cache across
    // mounts, so it must not be reachable this way either.
    let failing = false

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) return operatorContext()
      if (url.includes(`${API_ROOT}/usage/in-flight`)) {
        return failing
          ? jsonResponse({ detail: "gateway restarting" }, 503)
          : jsonResponse({ requests: [inFlightRequest()], total: 1 })
      }
      if (url.includes(`${API_ROOT}/usage/count`))
        return jsonResponse({ total: 0 })
      if (url.includes(`${API_ROOT}/usage/summary`)) {
        return jsonResponse({
          by_model: [],
          by_user: [],
          by_api_key: [],
          by_source: [],
          series: [],
        })
      }
      return jsonResponse([])
    })

    // The wait is jumped rather than slept through. `useInFlightRequests`
    // declares its own `retry` (three attempts, since a 503 is a gateway
    // restarting and worth re-asking), which overrides the harness's
    // `retry: false`, so reaching the error arm costs the 2s poll plus
    // TanStack's 1s/2s/4s backoffs. On real timers that was 9.1s, a third of
    // this whole suite's wall clock in one case, and the 20s and 30s ceilings
    // above were sized to survive it.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      renderPage(<ActivityPage />, "/activity?range=24h")
      await waitFor(() => expect(liveControl()).toBeInTheDocument())

      failing = true
      // Past the poll and all three backoffs. `...Async` rather than the
      // synchronous form because each attempt is a fetch: the awaits between
      // timers are what let those promises settle and schedule the next one.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2_000 + 1_000 + 2_000 + 4_000 + 500)
      })
      expect(liveControl()).not.toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it("reports live traffic gateway-wide, whatever the table is filtered to", async () => {
    // The endpoint takes no filters (a request in progress has no status, cost, or
    // token count to filter on), so the count is not narrowed to the current view
    // and the request itself must stay bare. A status filter that empties the table
    // therefore leaves the live count alone rather than hiding it.
    const { calls } = mockApi({
      rows: [],
      inFlight: { requests: [inFlightRequest()], total: 1 },
    })
    renderPage(<ActivityPage />, "/activity?range=7d&status=error&model=gpt-4o")

    await screen.findByText("No requests match these filters.")
    await waitFor(() => expect(liveControl()).toBeInTheDocument())

    const requested = calls
      .map((c) => c.url)
      .filter((url) => url.includes(`${API_ROOT}/usage/in-flight`))
    expect(requested.length).toBeGreaterThan(0)
    for (const url of requested) {
      expect(url).toBe(`${API_ROOT}/usage/in-flight`)
    }
  })

  it("leaves the paginator counting settled rows only", async () => {
    // The live request is not part of any page's slice, so folding it into "N of M"
    // would make the count disagree with the log the operator can page through.
    mockApi({
      rows: [entry()],
      total: 1,
      inFlight: { requests: [inFlightRequest()], total: 1 },
    })
    renderPage(<ActivityPage />, "/activity?range=24h")

    await waitFor(() => expect(liveControl()).toBeInTheDocument())
    expect(screen.getByText(/1\s*[–-]\s*1 of 1/)).toBeInTheDocument()
  })

  it("does not re-read the log when a tracked request settles", async () => {
    // The freeze, and the whole point of it: on a busy gateway requests settle
    // continuously, and re-reading the log on each one reshuffled the table every
    // few seconds under whoever was trying to read it. The settled request appears
    // at the next refresh instead.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      let live: InFlightResponse = { requests: [inFlightRequest()], total: 1 }
      const { calls } = mockApi({ rows: [entry()], inFlight: () => live })
      renderPage(<ActivityPage />, "/activity?range=24h")

      await waitFor(() => expect(liveControl()).toBeInTheDocument())
      const before = listCalls(calls).length

      // The request settles: the next poll no longer carries it.
      live = { requests: [], total: 0 }
      await vi.advanceTimersByTimeAsync(3_000)
      await waitFor(() => expect(liveControl()).not.toBeInTheDocument())

      // Several further polls, so this is not just a question of timing.
      await vi.advanceTimersByTimeAsync(10_000)
      expect(listCalls(calls).length).toBe(before)
    } finally {
      vi.useRealTimers()
    }
  })

  it("leaves an expanded row alone while it polls for in-flight requests", async () => {
    // Regression: the poll runs every 2s and its result used to re-derive the
    // table's rows, so DataTable rebuilt its detail host to match and an operator
    // who expanded a row watched the panel flash and slide open again every couple
    // of seconds. The poll no longer touches the rows array at all.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      const { calls } = mockApi({
        rows: [entry({ id: "settled-1" })],
        inFlight: { requests: [inFlightRequest()], total: 1 },
      })
      renderPage(<ActivityPage />, "/activity?range=24h")

      const row = (await screen.findByText("gpt-4o")).closest("tr")!
      await user.click(row)
      const panel = screen
        .getByText("Request detail")
        .closest(".otari-detail-row")
      expect(panel).not.toBeNull()

      const polls = () =>
        calls.filter((c) => c.url.includes(`${API_ROOT}/usage/in-flight`))
          .length
      const before = polls()
      await vi.advanceTimersByTimeAsync(5_000)
      await waitFor(() => expect(polls()).toBeGreaterThan(before + 1))

      // Same node, still open: the panel was never torn down and rebuilt.
      expect(
        screen.getByText("Request detail").closest(".otari-detail-row"),
      ).toBe(panel)
      expect(document.querySelectorAll(".otari-detail-row")).toHaveLength(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it("offers newer rows as a badge, and loads them only when it is pressed", async () => {
    // The freeze's other half: a page that never moves must still be able to say it
    // has fallen behind, or a quiet gateway and a flooded one look identical.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      let serverTotal = 4
      const { calls } = mockApi({ rows: [entry()], total: () => serverTotal })
      renderPage(<ActivityPage />, "/activity?range=24h")

      await screen.findByText("gpt-4o")
      expect(
        screen.queryByRole("button", { name: /new/ }),
      ).not.toBeInTheDocument()

      // 87 requests land while the operator reads the page.
      serverTotal = 91
      const listsBefore = listCalls(calls).length
      await vi.advanceTimersByTimeAsync(16_000)
      const badge = await screen.findByRole("button", { name: /87 new/ })

      // Nothing was re-read to discover that: the log is still the one on screen.
      expect(listCalls(calls).length).toBe(listsBefore)

      await user.click(badge)
      await waitFor(() =>
        expect(listCalls(calls).length).toBeGreaterThan(listsBefore),
      )
      // Loaded: the pinned count has caught up, so there is nothing left to offer.
      await waitFor(() =>
        expect(
          screen.queryByRole("button", { name: /new/ }),
        ).not.toBeInTheDocument(),
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it("re-reads the total when the operator pages, so newer rows do not strand the old ones", async () => {
    // The total is not in the count's key, so a frozen page would keep whichever
    // value it loaded with. `TablePagination` derives `isLast` from the total
    // whenever it has one, so an understated total disables Next short of the real
    // end and leaves the oldest rows unreachable. On main the settle-refetch hid
    // this by re-reading the count on any traffic; nothing does now except this.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      let serverTotal = 100
      mockApi({ rows: [entry()], total: () => serverTotal })
      renderPage(<ActivityPage />, "/activity?range=24h&size=50")

      await screen.findByText("gpt-4o")
      await waitFor(() =>
        expect(screen.getByText(/of 100/)).toBeInTheDocument(),
      )

      // Twenty land, taking the log to three pages of fifty.
      serverTotal = 120
      await user.click(screen.getByRole("button", { name: /next page/i }))

      await waitFor(() =>
        expect(screen.getByText(/of 120/)).toBeInTheDocument(),
      )
      // The third page is reachable, so the oldest twenty are not stranded behind a
      // boundary computed from a total that has moved on.
      expect(screen.getByRole("button", { name: /next page/i })).toBeEnabled()
    } finally {
      vi.useRealTimers()
    }
  })

  it("says so when it cannot tell whether newer rows exist", async () => {
    // A badge that is simply absent reads as "nothing has landed". On a table that
    // no longer moves by itself, that makes a flooded gateway look like an idle one.
    let countAsks = 0
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) return operatorContext()
      // The pinned count and the polled one share a URL, so fail every count after
      // the first: the page keeps a total and loses any way to tell if it is current.
      if (url.includes(`${API_ROOT}/usage/count`)) {
        countAsks += 1
        return countAsks === 1
          ? jsonResponse({ total: 1 })
          : jsonResponse({ detail: "nope" }, 500)
      }
      if (url.includes(`${API_ROOT}/usage/in-flight`))
        return jsonResponse({ requests: [], total: 0 })
      if (url.includes(`${API_ROOT}/usage/summary`)) {
        return jsonResponse({
          by_model: [],
          by_user: [],
          by_api_key: [],
          by_source: [],
          series: [],
        })
      }
      return jsonResponse([entry()])
    })

    renderPage(<ActivityPage />, "/activity?range=24h")

    await screen.findByText("gpt-4o")
    await waitFor(() =>
      expect(screen.getByText("Newer rows unknown")).toBeInTheDocument(),
    )
    // Still no false badge, and the log itself is not reported as broken.
    expect(
      screen.queryByRole("button", { name: /new · load/ }),
    ).not.toBeInTheDocument()
  })

  it("does not poll for newer rows on a page that cannot show them", async () => {
    // Newer rows land at the top of page 1, so on page 3 a badge offering to load
    // them would be a promise the refresh does not keep.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const { calls } = mockApi({ rows: [entry()], total: 500 })
      renderPage(<ActivityPage />, "/activity?range=24h&page=2")

      await screen.findByText("gpt-4o")
      const before = countCalls(calls).length
      await vi.advanceTimersByTimeAsync(40_000)

      expect(countCalls(calls).length).toBe(before)
      expect(
        screen.queryByRole("button", { name: /new/ }),
      ).not.toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it("drops the badge when the operator pages past the rows it offers", async () => {
    // `page` is not part of the live count's key, so paging forward disables the
    // poll but leaves its last payload in the cache. Read unguarded, that keeps
    // the badge on screen for pages where pressing it loads the current page and
    // the newer rows stay at the top of page 1, out of sight.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      let serverTotal = 100
      mockApi({ rows: [entry()], total: () => serverTotal })
      renderPage(<ActivityPage />, "/activity?range=24h")

      await screen.findByText("gpt-4o")
      serverTotal = 120
      await vi.advanceTimersByTimeAsync(16_000)
      await screen.findByRole("button", { name: /20 new/ })

      await user.click(screen.getByRole("button", { name: "Next page" }))
      await waitFor(() =>
        expect(
          screen.queryByRole("button", { name: /new/ }),
        ).not.toBeInTheDocument(),
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it("does not poll for newer rows in a window that has already ended", async () => {
    // A window bounded in the past can gain no rows, so the poll would be pure cost.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const { calls } = mockApi({ rows: [entry()], total: 1 })
      renderPage(
        <ActivityPage />,
        "/activity?start_date=2026-01-01T00:00:00Z&end_date=2026-01-02T00:00:00Z",
      )

      await screen.findByText("gpt-4o")
      const before = countCalls(calls).length
      await vi.advanceTimersByTimeAsync(40_000)

      expect(countCalls(calls).length).toBe(before)
    } finally {
      vi.useRealTimers()
    }
  })
})
