import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query"
import { act, render, screen, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ConnectionStatus } from "@/app/ConnectionStatus"
import { apiFetch } from "@/shared/api/client"

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

describe("ConnectionStatus", () => {
  afterEach(() => vi.restoreAllMocks())

  it("alerts when the gateway is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new TypeError("Failed to fetch"),
    )
    renderWithProbe()

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent(/Can’t reach the gateway/)
    // One failed request is only the suspicion; the alarm follows the probe and
    // its retries failing too. The failed request itself is not sent again: it
    // may be a heavy read, and the probe is what rechecks.
    const fetchMock = vi.mocked(globalThis.fetch)
    expect(
      fetchMock.mock.calls.filter(([url]) => isLiveness(url)),
    ).toHaveLength(3)
    expect(
      fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/settings")),
    ).toHaveLength(1)
  })

  it("keeps the alert through a cache write while the gateway is unreachable", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new TypeError("Failed to fetch"),
    )
    const { client } = renderWithProbe()
    await screen.findByRole("alert")

    // `setQueryData` reaches no gateway, so it is not an answer from one.
    await act(async () => {
      client.setQueryData(["optimistic"], { ok: true })
    })

    expect(screen.getByRole("alert")).toBeInTheDocument()
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
    const { client } = renderWithProbe()

    await waitFor(() =>
      expect(client.getQueryState(["probe"])?.status).toBe("error"),
    )
    await waitFor(() => expect(livenessAnswered).toBe(true))
    // Let the probe's answer settle into the cache before reading the DOM.
    await act(async () => {})
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("clears itself once the gateway responds again, without the failed query refetching", async () => {
    // The failed query stays in the cache in its error state, as one on a page
    // the operator has left does; recovery is the probe's to report.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      let online = false
      vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
        if (!online) throw new TypeError("Failed to fetch")
        return jsonResponse("I'm alive!")
      })
      const { client } = renderWithProbe()

      await screen.findByRole("alert")

      online = true
      // Past the probe's recheck interval (`GATEWAY_LIVENESS_RECHECK_MS`).
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5_000)
      })

      await waitFor(() =>
        expect(screen.queryByRole("alert")).not.toBeInTheDocument(),
      )
      expect(client.getQueryState(["probe"])?.status).toBe("error")
    } finally {
      vi.useRealTimers()
    }
  })

  it("holds the alert through each recheck while the gateway stays down", async () => {
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
      expect(screen.getByRole("alert")).toBeInTheDocument()
      for (let elapsed = 0; elapsed < 12_000; elapsed += 250) {
        await act(async () => {
          await vi.advanceTimersByTimeAsync(250)
        })
        expect(screen.getByRole("alert")).toBeInTheDocument()
      }
    } finally {
      vi.useRealTimers()
    }
  })

  it("alerts within seconds when the gateway hangs rather than refusing", async () => {
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
      expect(screen.getByRole("alert")).toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it("stays quiet for an error that is not a lost connection", async () => {
    // A 500 means the backend answered; that is a page-level error, not "offline".
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "boom" }, 500),
    )
    renderWithProbe()

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())
    await waitFor(() =>
      expect(screen.queryByRole("alert")).not.toBeInTheDocument(),
    )
    expect(
      vi.mocked(globalThis.fetch).mock.calls.some(([url]) => isLiveness(url)),
    ).toBe(false)
  })
})
