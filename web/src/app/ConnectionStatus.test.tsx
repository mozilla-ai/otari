import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query"
import { act, render, screen, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { announce } from "@/app/announce"
import { ConnectionStatus, RECONNECTED_MS } from "@/app/ConnectionStatus"
import { apiFetch, setUnauthorizedHandler } from "@/shared/api/client"

// Drives one management request so the query cache carries a real success/error,
// exactly what ConnectionStatus watches. No retry, like the build poll every page
// runs, so a single dropped fetch lands in the cache as an error.
function Probe() {
  useQuery({
    queryKey: ["probe"],
    queryFn: () => apiFetch("/settings"),
    retry: false,
  })
  return null
}

function renderWithProbe(retryDelay = 0): { client: QueryClient } {
  // `retryDelay` reaches the liveness probe, which sets its own `retry` but
  // takes the client's delay, so by default its retries do not wait out a real
  // backoff.
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, retryDelay } },
  })
  const wrap = (children: ReactNode) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  render(
    wrap(
      <>
        <Probe />
        <ConnectionStatus />
      </>,
    ),
  )
  return { client }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function isLiveness(input: RequestInfo | URL): boolean {
  return String(input).endsWith("/health/liveness")
}

vi.mock("@/app/announce", () => ({ announce: vi.fn() }))

const DOWN = /Can’t reach the gateway\./
const band = () => document.querySelector("[data-connection-band]")
const findDown = () => screen.findByText(DOWN)
const getDown = () => screen.getByText(DOWN)
const queryDown = () => screen.queryByText(DOWN)

describe("ConnectionStatus", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.mocked(announce).mockClear()
  })

  it("shows a band when the gateway is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new TypeError("Failed to fetch"),
    )
    renderWithProbe()

    await findDown()
    expect(band()).toHaveTextContent(
      /^Can’t reach the gateway\. Showing what loaded before \d\d:\d\d\. Changes won’t save until it answers\.Retrying$/,
    )
    // A band in the flow, not a toast floated over the page, and no live
    // region of its own: its sentence is announced once.
    expect(band()).toHaveClass("sticky", "top-0")
    expect(band()).not.toHaveClass("fixed")
    expect(screen.queryByRole("status")).toBeNull()
    expect(screen.queryByRole("alert")).toBeNull()
    expect(vi.mocked(announce).mock.calls).toEqual([
      [
        expect.stringMatching(
          /^Can’t reach the gateway\. Showing what loaded before \d\d:\d\d\. Changes won’t save until it answers\.$/,
        ),
      ],
    ])
    // One failed request is only the suspicion; the alarm follows the probe and
    // its retries failing too. The failed request itself is not sent again: it
    // may be a heavy read, and the probe is what rechecks.
    const fetchMock = vi.mocked(globalThis.fetch)
    // Three probes, each with its no-cors second look, since nothing answered.
    expect(
      fetchMock.mock.calls.filter(
        ([url, init]) => isLiveness(url) && init?.mode !== "no-cors",
      ),
    ).toHaveLength(3)
    expect(
      fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/settings")),
    ).toHaveLength(1)
  })

  it("keeps the band through a cache write while the gateway is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new TypeError("Failed to fetch"),
    )
    const { client } = renderWithProbe()
    await findDown()

    // `setQueryData` reaches no gateway, so it is not an answer from one.
    await act(async () => {
      client.setQueryData(["optimistic"], { ok: true })
    })

    expect(getDown()).toBeInTheDocument()
  })

  it("stays quiet when one request drops but the gateway answers", async () => {
    // A tab waking from sleep or an edge proxy dropping a pooled connection
    // fails one fetch against a healthy backend. The banner used to show on that
    // alone and hold until the next poll.
    let dropped = false
    let livenessAnswered = false
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (!isLiveness(input)) {
        if (!dropped) {
          dropped = true
          throw new TypeError("Failed to fetch")
        }
        return jsonResponse({ ok: true })
      }
      livenessAnswered = true
      return jsonResponse("I'm alive!")
    })
    renderWithProbe()

    await waitFor(() => expect(livenessAnswered).toBe(true))
    // Let the probe's answer settle into the cache before reading the DOM.
    await act(async () => {})
    expect(queryDown()).not.toBeInTheDocument()
  })

  it("clears itself once the gateway responds again, and re-runs what failed", async () => {
    // Recovery is the probe's to report; the query that failed to reach the
    // gateway then runs again, so its page's own error clears too.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      let online = false
      vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
        if (!online) throw new TypeError("Failed to fetch")
        return jsonResponse("I'm alive!")
      })
      const { client } = renderWithProbe()

      await findDown()

      online = true
      // Past the probe's recheck interval (`GATEWAY_LIVENESS_RECHECK_MS`).
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5_000)
      })

      await waitFor(() => expect(queryDown()).not.toBeInTheDocument())
      // It says so for a moment while what failed refetches, then goes.
      expect(band()).toHaveTextContent("Reconnected. Refreshing this page.")
      await waitFor(() =>
        expect(client.getQueryState(["probe"])?.status).toBe("success"),
      )
      await act(async () => {
        await vi.advanceTimersByTimeAsync(RECONNECTED_MS - 100)
      })
      expect(band()).toHaveTextContent("Reconnected.")
      await act(async () => {
        await vi.advanceTimersByTimeAsync(200)
      })
      expect(band()).toBeNull()
      expect(vi.mocked(announce).mock.calls.at(-1)).toEqual([
        "Reconnected. Refreshing this page.",
      ])
    } finally {
      vi.useRealTimers()
    }
  })

  it("holds the band through each recheck while the gateway stays down", async () => {
    // The probe has never held data, so a recheck puts it back in `pending` for
    // the length of its retries. A real delay keeps that window open long enough
    // to see.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      vi.spyOn(globalThis, "fetch").mockRejectedValue(
        new TypeError("Failed to fetch"),
      )
      renderWithProbe(1_000)

      // Past the probe's two retries.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2_500)
      })
      expect(getDown()).toBeInTheDocument()
      for (let elapsed = 0; elapsed < 12_000; elapsed += 250) {
        await act(async () => {
          await vi.advanceTimersByTimeAsync(250)
        })
        expect(getDown()).toBeInTheDocument()
      }
    } finally {
      vi.useRealTimers()
    }
  })

  it("shows the band within seconds when the gateway hangs rather than refusing", async () => {
    // A stalled gateway times requests out instead of failing them, so the
    // alarm waits on the probe's own deadline, not `apiFetch`'s 30s.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      // `AbortSignal.timeout` runs on a timer fake timers do not reach, so it is
      // rebuilt on `setTimeout` for the test.
      vi.spyOn(AbortSignal, "timeout").mockImplementation((ms) => {
        const controller = new AbortController()
        setTimeout(
          () => controller.abort(new DOMException("", "TimeoutError")),
          ms,
        )
        return controller.signal
      })
      vi.spyOn(globalThis, "fetch").mockImplementation(
        (_input, init) =>
          new Promise((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () =>
              reject(init.signal?.reason),
            )
          }),
      )
      renderWithProbe()

      // The page's own request times out at 30s, then three 5s probe attempts.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(30_000 + 3 * 5_000 + 500)
      })
      expect(getDown()).toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it("holds the time it went down while later requests keep failing", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      vi.setSystemTime(new Date(2026, 10, 25, 14, 2, 0))
      vi.spyOn(globalThis, "fetch").mockRejectedValue(
        new TypeError("Failed to fetch"),
      )
      renderWithProbe()
      await findDown()
      expect(band()).toHaveTextContent("before 14:02.")

      await act(async () => {
        await vi.advanceTimersByTimeAsync(3 * 60_000)
      })
      expect(band()).toHaveTextContent("before 14:02.")
    } finally {
      vi.useRealTimers()
    }
  })

  it("re-runs what failed to connect when the gateway answers the first check, so a blip leaves no stale error", async () => {
    // One dropped request, then a probe that answers at once: no outage was
    // ever confirmed, so there is no recovery to hang the refetch on.
    let dropped = false
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (isLiveness(input)) return jsonResponse("I'm alive!")
      if (!dropped) {
        dropped = true
        throw new TypeError("Failed to fetch")
      }
      return jsonResponse({ ok: true })
    })
    const { client } = renderWithProbe()

    await waitFor(() =>
      expect(client.getQueryState(["probe"])?.status).toBe("success"),
    )
    expect(queryDown()).not.toBeInTheDocument()
  })

  it("re-runs only what is on screen, not a page the operator has left", async () => {
    let settingsDropped = false
    let livenessAnswered = false
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (isLiveness(input)) {
        livenessAnswered = true
        return jsonResponse("I'm alive!")
      }
      if (String(input).endsWith("/left")) {
        throw new TypeError("Failed to fetch")
      }
      // The page on screen drops its first request too, so a recovery runs.
      if (!settingsDropped) {
        settingsDropped = true
        throw new TypeError("Failed to fetch")
      }
      return jsonResponse({ ok: true })
    })
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false, retryDelay: 0 } },
    })
    // A query from a page no longer mounted: in the cache, with no observer.
    await client
      .fetchQuery({ queryKey: ["left"], queryFn: () => apiFetch("/left") })
      .catch(() => {})
    render(
      <QueryClientProvider client={client}>
        <Probe />
        <ConnectionStatus />
      </QueryClientProvider>,
    )
    await waitFor(() => expect(livenessAnswered).toBe(true))
    await waitFor(() =>
      expect(client.getQueryState(["probe"])?.status).toBe("success"),
    )
    const settings = vi
      .mocked(globalThis.fetch)
      .mock.calls.filter(([url]) => String(url).endsWith("/settings"))
    expect(settings).toHaveLength(2)
    const left = vi
      .mocked(globalThis.fetch)
      .mock.calls.filter(([url]) => String(url).endsWith("/left"))
    expect(left).toHaveLength(1)
  })

  it("does not keep re-running a request that fails on its own while the gateway answers", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (isLiveness(input)) return jsonResponse("I'm alive!")
      // Blocked in the browser, say, every time.
      throw new TypeError("Failed to fetch")
    })
    renderWithProbe()

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 300))
    })
    const settings = vi
      .mocked(globalThis.fetch)
      .mock.calls.filter(([url]) => String(url).endsWith("/settings"))
    expect(settings).toHaveLength(2)
    expect(queryDown()).not.toBeInTheDocument()
  })

  it("counts a liveness check redirected to a sign-in page on another origin as the gateway answering", async () => {
    let answered = 0
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      if (!isLiveness(input)) throw new TypeError("Failed to fetch")
      // Followed with CORS, the redirect to another origin is refused; read
      // without CORS, something there answers.
      if (init?.mode !== "no-cors") throw new TypeError("Failed to fetch")
      answered += 1
      return { type: "opaque", status: 0, ok: false, body: null } as Response
    })
    renderWithProbe()

    await waitFor(() => expect(answered).toBeGreaterThan(0))
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 100))
    })
    expect(queryDown()).not.toBeInTheDocument()
  })

  it.each([401, 404, 500])(
    "counts a %s from the liveness check as the gateway answering, without signing out",
    async (status) => {
      const onUnauthorized = vi.fn()
      setUnauthorizedHandler(onUnauthorized)
      try {
        let probed = 0
        vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
          if (!isLiveness(input)) throw new TypeError("Failed to fetch")
          probed += 1
          return jsonResponse({ detail: "no" }, status)
        })
        renderWithProbe()

        await waitFor(() => expect(probed).toBeGreaterThan(0))
        await act(async () => {
          await new Promise((resolve) => setTimeout(resolve, 100))
        })
        expect(queryDown()).not.toBeInTheDocument()
        expect(onUnauthorized).not.toHaveBeenCalled()
      } finally {
        setUnauthorizedHandler(null)
      }
    },
  )

  it("stays quiet for an error that is not a lost connection", async () => {
    // A 500 means the backend answered; that is a page-level error, not "offline".
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "boom" }, 500),
    )
    renderWithProbe()

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())
    await waitFor(() => expect(queryDown()).not.toBeInTheDocument())
    expect(
      vi.mocked(globalThis.fetch).mock.calls.some(([url]) => isLiveness(url)),
    ).toBe(false)
  })
})
