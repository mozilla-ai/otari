import { afterEach, describe, expect, it, vi } from "vitest"

import { ApiError, apiFetch, createSession, deleteSession } from "./client"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("apiFetch", () => {
  it("bounds a request that never settles", async () => {
    // A hung request holds one of the browser's ~6 sockets per origin. Enough of
    // them and everything an operator clicks afterwards queues behind them, which
    // reads as the click doing nothing. The deadline is ours, not the server's.
    vi.useFakeTimers()
    try {
      vi.spyOn(globalThis, "fetch").mockImplementation(
        (_input, init) =>
          new Promise((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () => {
              reject(new DOMException("timed out", "TimeoutError"))
            })
          }),
      )

      const pending = apiFetch("/v1/models")
      const assertion = expect(pending).rejects.toMatchObject({
        status: 0,
        message: expect.stringContaining("did not respond within 30s"),
      })
      await vi.advanceTimersByTimeAsync(30_000)
      await assertion
    } finally {
      vi.useRealTimers()
    }
  })

  it("bounds a response whose body stalls after the headers arrive", async () => {
    // fetch() resolves on headers, so a stalled body trips the deadline on the
    // JSON read rather than on the fetch. Callers only handle ApiError.
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      status: 200,
      ok: true,
      json: () => Promise.reject(new DOMException("timed out", "TimeoutError")),
    } as unknown as Response)

    await expect(apiFetch("/v1/models")).rejects.toBeInstanceOf(ApiError)
    await expect(apiFetch("/v1/models")).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining("did not respond within 30s"),
    })
  })

  it("passes a caller's signal through instead of imposing its own", async () => {
    const controller = new AbortController()
    const seen: (AbortSignal | null | undefined)[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation((_input, init) => {
      seen.push(init?.signal)
      return Promise.resolve(new Response("{}", { status: 200 }))
    })

    await apiFetch("/v1/models", { signal: controller.signal })

    expect(seen[0]).toBe(controller.signal)
  })

  it("does not quote its own 30s deadline at a caller that set a longer one", async () => {
    // The bulk usage delete and reprice run on longRequestSignal(); telling an
    // operator who waited five minutes that nothing answered "within 30s" would
    // point them at the wrong thing.
    const controller = new AbortController()
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new DOMException("timed out", "TimeoutError"),
    )

    await expect(
      apiFetch("/v1/usage", { signal: controller.signal }),
    ).rejects.toMatchObject({
      status: 0,
      message: "The gateway did not respond in time.",
    })
  })

  it("reports an unreachable gateway differently from a timeout", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new TypeError("Failed to fetch"),
    )

    await expect(apiFetch("/v1/models")).rejects.toBeInstanceOf(ApiError)
    await expect(apiFetch("/v1/models")).rejects.toMatchObject({
      message: expect.stringContaining("could not reach the gateway"),
    })
  })
})

describe("createSession", () => {
  it("bounds the request the same way apiFetch does", async () => {
    const seen: (AbortSignal | null | undefined)[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation((_input, init) => {
      seen.push(init?.signal)
      return Promise.resolve(new Response("{}", { status: 200 }))
    })

    await createSession("test-key")

    expect(seen[0]).toBeInstanceOf(AbortSignal)
  })

  it("maps a timeout to the same bounded-request message apiFetch uses", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new DOMException("timed out", "TimeoutError"),
    )

    await expect(createSession("test-key")).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining("did not respond within 30s"),
    })
  })
})

describe("deleteSession", () => {
  it("bounds the request the same way apiFetch does", async () => {
    const seen: (AbortSignal | null | undefined)[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation((_input, init) => {
      seen.push(init?.signal)
      return Promise.resolve(new Response(null, { status: 204 }))
    })

    await deleteSession()

    expect(seen[0]).toBeInstanceOf(AbortSignal)
  })

  it("still resolves, rather than throwing, when the request times out", async () => {
    // deleteSession is best-effort and swallows failures: AuthContext.logout
    // relies on this promise always settling, timeout or not, to clear
    // isSigningOut and unblock a subsequent sign-in (see #557).
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new DOMException("timed out", "TimeoutError"),
    )

    await expect(deleteSession()).resolves.toBeUndefined()
  })
})
