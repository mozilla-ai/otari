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
      // AbortSignal.timeout's clock is the runtime's, not the one vi.useFakeTimers
      // installs, so advancing time does not fire it and the deadline would take a
      // real 30 seconds to arrive. Stand in a controller driven by setTimeout,
      // which the fake clock does own, and assert the delay the client asked for:
      // that is the claim this test makes about the deadline.
      const timeout = vi
        .spyOn(AbortSignal, "timeout")
        .mockImplementation((delay) => {
          const controller = new AbortController()
          setTimeout(
            () =>
              controller.abort(new DOMException("timed out", "TimeoutError")),
            delay,
          )
          return controller.signal
        })
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
      expect(timeout).toHaveBeenCalledWith(30_000)
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

    await createSession({ masterKey: "test-key" })

    expect(seen[0]).toBeInstanceOf(AbortSignal)
  })

  it("maps a timeout to the same bounded-request message apiFetch uses", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new DOMException("timed out", "TimeoutError"),
    )

    await expect(
      createSession({ masterKey: "test-key" }),
    ).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining("did not respond within 30s"),
    })
  })

  it("throws a 503 the gateway did not write rather than blaming the credential", async () => {
    // The redeploy maintenance mode exists for is exactly when a proxy with no
    // healthy upstream answers 503 itself. That body carries no `detail`, and
    // rendering its status text on a credential's label row would say the
    // credential was rejected by a gateway that never saw it.
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("<html>502 Bad Gateway</html>", {
        status: 503,
        statusText: "Service Unavailable",
        headers: { "Content-Type": "text/html" },
      }),
    )

    await expect(
      createSession({ masterKey: "test-key" }),
    ).rejects.toMatchObject({
      status: 503,
      message: "Service Unavailable",
    })
  })

  it.each([401, 403, 503])(
    "returns %i as a refusal carrying the gateway's own wording",
    async (status) => {
      // 503 is maintenance mode. It belongs with the other two rather than on
      // the throw path: the gateway is deliberately refusing this sign-in, in
      // wording written for the person reading it. The body carrying `detail`
      // is what says the gateway wrote it.
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(JSON.stringify({ detail: "refused, and here is why" }), {
          status,
          headers: { "Content-Type": "application/json" },
        }),
      )

      // The status rides along with the wording so a caller can tell the three
      // refusals apart without re-reading the message, which is the one part of
      // a refusal that must not be recorded anywhere. A deployment frozen for
      // maintenance and a wrong credential are not the same funnel step.
      await expect(createSession({ masterKey: "test-key" })).resolves.toEqual({
        ok: false,
        message: "refused, and here is why",
        status,
      })
    },
  )
})

describe("createSession credentials", () => {
  it("posts a password credential as email and password, not as a master key", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response("{}", { status: 200 }))

    const result = await createSession({
      email: "operator@example.com",
      password: "a-real-password",
    })

    expect(result).toEqual({ ok: true })
    expect(fetchMock.mock.calls[0][1]?.body).toBe(
      JSON.stringify({
        email: "operator@example.com",
        password: "a-real-password",
      }),
    )
  })

  it("returns the gateway's message on a refusal rather than a bare false", async () => {
    // 401 and 403 both mean "not signed in" but not the same thing, and the
    // sign-in screen can only say which if the message survives the call.
    for (const [status, detail] of [
      [401, "Incorrect email or password"],
      [403, "Master-key sign-in is retired on this deployment"],
    ] as const) {
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(JSON.stringify({ detail }), {
          status,
          headers: { "Content-Type": "application/json" },
        }),
      )

      // The status rides along with the message so a caller can tell the two
      // refusals apart without re-reading the wording, which is the one part of
      // a refusal that must not be recorded anywhere.
      await expect(createSession({ masterKey: "k" })).resolves.toEqual({
        ok: false,
        message: detail,
        status,
      })
    }
  })

  it("throws rather than reporting a refusal when the gateway errors", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Database error" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await expect(createSession({ masterKey: "k" })).rejects.toMatchObject({
      status: 500,
      message: "Database error",
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
