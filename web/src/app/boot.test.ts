import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ApiError, RedirectedError } from "@/shared/api/client"
import type { WireBootstrap } from "@/shared/helpers/bootstrap"
import {
  BOOTSTRAP_ATTEMPT_TIMEOUT_MS,
  type BootLoader,
  createBootLoader,
  describeBootFailure,
  type FailedState,
  firstSettled,
  isTransientBootstrapFailure,
  readBootstrap,
  retryDelayMs,
} from "./boot"

const BOOTSTRAP: WireBootstrap = { surfaces: [] }
const DOWN = new ApiError(0, "Network error: could not reach the gateway.")

// No jitter, so a test can name the wait: the first retry waits 500ms, the
// second 1s, and so on.
const NO_JITTER = () => 0

let loader: BootLoader | undefined

function start(
  load: (signal: AbortSignal) => Promise<WireBootstrap>,
  prepare: () => Promise<void> = async () => {},
) {
  loader = createBootLoader({ prepare, load, random: NO_JITTER })
  return loader
}

function setVisibility(value: DocumentVisibilityState) {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => value,
  })
}

/** Drops the override, so `document` reads the environment's own value again. */
function restoreVisibility() {
  Reflect.deleteProperty(document, "visibilityState")
}

describe("createBootLoader", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    loader?.dispose()
    loader = undefined
    vi.useRealTimers()
    vi.restoreAllMocks()
    restoreVisibility()
  })

  it("reads the bootstrap once the request policy is settled", async () => {
    const order: string[] = []
    const boot = start(
      async () => {
        order.push("load")
        return BOOTSTRAP
      },
      async () => {
        order.push("prepare")
      },
    )

    await expect(firstSettled(boot)).resolves.toEqual({
      status: "ready",
      bootstrap: BOOTSTRAP,
    })
    expect(order).toEqual(["prepare", "load"])
  })

  it("fails without asking for a bootstrap, or retrying, when the deployment could not be chosen", async () => {
    const load = vi.fn(async () => BOOTSTRAP)
    const boot = start(load, async () => {
      throw new Error("no directory")
    })

    await expect(firstSettled(boot)).resolves.toMatchObject({
      status: "failed",
      stage: "prepare",
      willRetry: false,
    })
    await vi.advanceTimersByTimeAsync(60_000)
    expect(load).not.toHaveBeenCalled()
  })

  it("retries a failed read with backoff until it answers", async () => {
    const load = vi
      .fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
      .mockRejectedValueOnce(DOWN)
      .mockRejectedValueOnce(new ApiError(503, "unavailable"))
      .mockResolvedValue(BOOTSTRAP)
    const boot = start(load)

    // A failure is a state, never a guessed deployment.
    expect(await firstSettled(boot)).toEqual({
      status: "failed",
      error: DOWN,
      stage: "load",
      isAttempting: false,
      willRetry: true,
      timedOut: false,
      offline: false,
      failedAt: Date.now(),
    })

    await vi.advanceTimersByTimeAsync(499)
    expect(load).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(load).toHaveBeenCalledTimes(2)
    expect(boot.getState().status).toBe("failed")

    await vi.advanceTimersByTimeAsync(1_000)
    expect(load).toHaveBeenCalledTimes(3)
    expect(boot.getState()).toEqual({ status: "ready", bootstrap: BOOTSTRAP })
  })

  it("abandons an attempt the gateway does not answer in time and tries again", async () => {
    const signals: AbortSignal[] = []
    const load = vi.fn(
      (signal: AbortSignal) =>
        new Promise<WireBootstrap>((_resolve, reject) => {
          signals.push(signal)
          signal.addEventListener("abort", () => reject(signal.reason))
        }),
    )
    const boot = start(load)

    await vi.advanceTimersByTimeAsync(BOOTSTRAP_ATTEMPT_TIMEOUT_MS)
    expect(signals[0]?.aborted).toBe(true)
    expect(boot.getState()).toMatchObject({
      status: "failed",
      willRetry: true,
      timedOut: true,
    })

    await vi.advanceTimersByTimeAsync(500)
    expect(load).toHaveBeenCalledTimes(2)
  })

  it("retries at once when the browser comes back online", async () => {
    const load = vi
      .fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
      .mockRejectedValueOnce(DOWN)
      .mockRejectedValueOnce(DOWN)
      .mockResolvedValue(BOOTSTRAP)
    const boot = start(load)
    await firstSettled(boot)
    // Into the second wait (1s), well short of it.
    await vi.advanceTimersByTimeAsync(500)
    expect(load).toHaveBeenCalledTimes(2)

    window.dispatchEvent(new Event("online"))
    await vi.advanceTimersByTimeAsync(0)

    expect(load).toHaveBeenCalledTimes(3)
    expect(boot.getState().status).toBe("ready")
  })

  it("retries at once when the page becomes visible, and not when it is hidden", async () => {
    const load = vi
      .fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
      .mockRejectedValueOnce(DOWN)
      .mockResolvedValue(BOOTSTRAP)
    const boot = start(load)
    await firstSettled(boot)

    setVisibility("hidden")
    document.dispatchEvent(new Event("visibilitychange"))
    await vi.advanceTimersByTimeAsync(0)
    expect(load).toHaveBeenCalledTimes(1)

    setVisibility("visible")
    document.dispatchEvent(new Event("visibilitychange"))
    await vi.advanceTimersByTimeAsync(0)
    expect(load).toHaveBeenCalledTimes(2)
    expect(boot.getState().status).toBe("ready")
  })

  it("waits for the operator after a definitive refusal, then asks again on Retry", async () => {
    const load = vi
      .fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
      .mockRejectedValueOnce(new ApiError(404, "Not Found"))
      .mockResolvedValue(BOOTSTRAP)
    const boot = start(load)

    expect(await firstSettled(boot)).toMatchObject({
      status: "failed",
      willRetry: false,
    })
    window.dispatchEvent(new Event("online"))
    await vi.advanceTimersByTimeAsync(60_000)
    expect(load).toHaveBeenCalledTimes(1)

    boot.retryNow()
    await vi.advanceTimersByTimeAsync(0)
    expect(load).toHaveBeenCalledTimes(2)
    expect(boot.getState().status).toBe("ready")
  })

  it("leaves no timer or listener behind once it has answered", async () => {
    const load = vi
      .fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
      .mockRejectedValueOnce(DOWN)
      .mockResolvedValue(BOOTSTRAP)
    const boot = start(load)
    await firstSettled(boot)
    const offWindow = vi.spyOn(window, "removeEventListener")
    const offDocument = vi.spyOn(document, "removeEventListener")
    boot.retryNow()
    await vi.advanceTimersByTimeAsync(0)
    expect(boot.getState().status).toBe("ready")

    expect(vi.getTimerCount()).toBe(0)
    expect(offWindow).toHaveBeenCalledWith("online", expect.any(Function))
    expect(offDocument).toHaveBeenCalledWith(
      "visibilitychange",
      expect.any(Function),
    )
    window.dispatchEvent(new Event("online"))
    document.dispatchEvent(new Event("visibilitychange"))
    boot.retryNow()
    await vi.advanceTimersByTimeAsync(60_000)
    expect(load).toHaveBeenCalledTimes(2)
  })

  it("hands a press the attempt already running instead of starting a second", async () => {
    let answer: ((value: WireBootstrap) => void) | undefined
    const load = vi
      .fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
      .mockRejectedValueOnce(DOWN)
      .mockImplementationOnce(
        () =>
          new Promise<WireBootstrap>((resolve) => {
            answer = resolve
          }),
      )
    const boot = start(load)
    await firstSettled(boot)
    // The background retry starts, and the press lands while it runs.
    await vi.advanceTimersByTimeAsync(500)
    expect(boot.getState()).toMatchObject({ isAttempting: true })
    let settled = false
    void boot.retryNow().then(() => {
      settled = true
    })
    await vi.advanceTimersByTimeAsync(0)
    expect(load).toHaveBeenCalledTimes(2)
    expect(settled).toBe(false)

    answer?.(BOOTSTRAP)
    await vi.advanceTimersByTimeAsync(0)
    expect(settled).toBe(true)
    expect(boot.getState().status).toBe("ready")
  })

  it("starts a new attempt for a press that lands as the last one is reported", async () => {
    const load = vi
      .fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
      .mockRejectedValueOnce(DOWN)
      .mockResolvedValue(BOOTSTRAP)
    const boot = start(load)
    // Pressed from inside the notification that the first attempt failed.
    const unsubscribe = boot.subscribe(() => {
      if (boot.getState().status === "failed") {
        unsubscribe()
        void boot.retryNow()
      }
    })
    await vi.advanceTimersByTimeAsync(0)
    expect(load).toHaveBeenCalledTimes(2)
    expect(boot.getState().status).toBe("ready")
  })

  it("hands a press made as an attempt starts that attempt, not a second one", async () => {
    const answers: Array<(value: WireBootstrap) => void> = []
    const load = vi
      .fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
      .mockRejectedValueOnce(DOWN)
      .mockImplementation(
        () =>
          new Promise<WireBootstrap>((resolve) => {
            answers.push(resolve)
          }),
      )
    const boot = start(load)
    await firstSettled(boot)
    // Pressed from inside the notification that the background retry began.
    let joined: Promise<void> | undefined
    const unsubscribe = boot.subscribe(() => {
      const state = boot.getState()
      if (state.status === "failed" && state.isAttempting) {
        unsubscribe()
        joined = boot.retryNow()
      }
    })
    await vi.advanceTimersByTimeAsync(500)
    expect(load).toHaveBeenCalledTimes(2)

    let settled = false
    void joined?.then(() => {
      settled = true
    })
    await vi.advanceTimersByTimeAsync(0)
    // Still waiting on the attempt it joined, not the one that already failed.
    expect(settled).toBe(false)
    answers[0]?.(BOOTSTRAP)
    await vi.advanceTimersByTimeAsync(0)
    expect(settled).toBe(true)
    expect(load).toHaveBeenCalledTimes(2)
  })

  it("asks again at once after a failure when the browser came back online during the attempt", async () => {
    let fail: ((error: unknown) => void) | undefined
    const load = vi
      .fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
      .mockRejectedValueOnce(DOWN)
      .mockRejectedValueOnce(DOWN)
      .mockRejectedValueOnce(DOWN)
      .mockImplementationOnce(
        () =>
          new Promise<WireBootstrap>((_resolve, reject) => {
            fail = reject
          }),
      )
      .mockResolvedValue(BOOTSTRAP)
    const boot = start(load)
    await firstSettled(boot)
    // Three failures in, the next wait is 4s; the fourth attempt is running.
    await vi.advanceTimersByTimeAsync(500 + 1_000 + 2_000)
    expect(load).toHaveBeenCalledTimes(4)

    window.dispatchEvent(new Event("online"))
    fail?.(DOWN)
    await vi.advanceTimersByTimeAsync(0)

    expect(load).toHaveBeenCalledTimes(5)
    expect(boot.getState().status).toBe("ready")
  })

  it("does not ask for the bootstrap once disposed during preparation", async () => {
    let prepared: (() => void) | undefined
    const load = vi.fn(async () => BOOTSTRAP)
    const boot = start(
      load,
      () =>
        new Promise<void>((resolve) => {
          prepared = resolve
        }),
    )
    boot.dispose()
    prepared?.()
    await vi.advanceTimersByTimeAsync(BOOTSTRAP_ATTEMPT_TIMEOUT_MS * 2)

    expect(load).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it("gives up on a preparation that does not settle in time, as no gateway chosen", async () => {
    const load = vi.fn(async () => BOOTSTRAP)
    const boot = start(load, () => new Promise<void>(() => {}))

    await vi.advanceTimersByTimeAsync(BOOTSTRAP_ATTEMPT_TIMEOUT_MS - 1)
    expect(boot.getState().status).toBe("loading")
    await vi.advanceTimersByTimeAsync(1)
    expect(boot.getState()).toMatchObject({
      status: "failed",
      stage: "prepare",
      willRetry: false,
    })
    expect(load).not.toHaveBeenCalled()
  })

  it("waits on a preparation still running from a timed-out attempt instead of starting another", async () => {
    let prepared: (() => void) | undefined
    const prepare = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          prepared = resolve
        }),
    )
    const load = vi.fn(async () => BOOTSTRAP)
    const boot = start(load, prepare)
    await vi.advanceTimersByTimeAsync(BOOTSTRAP_ATTEMPT_TIMEOUT_MS)
    expect(boot.getState()).toMatchObject({
      status: "failed",
      stage: "prepare",
    })

    void boot.retryNow()
    await vi.advanceTimersByTimeAsync(0)
    expect(prepare).toHaveBeenCalledTimes(1)

    prepared?.()
    await vi.advanceTimersByTimeAsync(0)
    expect(load).toHaveBeenCalledTimes(1)
    expect(boot.getState().status).toBe("ready")
  })

  it("records that the browser was offline when the attempt failed", async () => {
    vi.spyOn(navigator, "onLine", "get").mockReturnValue(false)
    const boot = start(async () => {
      throw DOWN
    })
    expect(await firstSettled(boot)).toMatchObject({
      offline: true,
      willRetry: true,
    })
  })

  it("keeps asking quietly behind a sign-in redirect, and opens once an attempt answers", async () => {
    // Which side of a redirect answered cannot always be told from here (a
    // canonical host's 503 without CORS reads like a sign-in page), so the
    // sign-in screen is not a dead end.
    const load = vi
      .fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
      .mockRejectedValueOnce(new RedirectedError())
      .mockRejectedValueOnce(new RedirectedError())
      .mockResolvedValue(BOOTSTRAP)
    const boot = start(load)
    expect(await firstSettled(boot)).toMatchObject({ willRetry: true })

    await vi.advanceTimersByTimeAsync(500 + 1_000)
    expect(load).toHaveBeenCalledTimes(3)
    expect(boot.getState().status).toBe("ready")
  })

  it("stops retrying once disposed", async () => {
    const load = vi.fn(async (): Promise<WireBootstrap> => {
      throw DOWN
    })
    const boot = start(load)
    await firstSettled(boot)

    boot.dispose()
    window.dispatchEvent(new Event("online"))
    await vi.advanceTimersByTimeAsync(60_000)

    expect(load).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(0)
  })
})

describe("isTransientBootstrapFailure", () => {
  it.each([
    ["a network fault or timeout", new ApiError(0, "down"), true],
    ["a 5xx", new ApiError(502, "bad gateway"), true],
    ["a 408", new ApiError(408, "timeout"), true],
    ["a 429", new ApiError(429, "slow down"), true],
    ["an intermediary's non-JSON 200", new ApiError(200, "not JSON"), true],
    ["something that is not an ApiError", new TypeError("boom"), true],
    ["a 404", new ApiError(404, "not found"), false],
    ["a 403", new ApiError(403, "forbidden"), false],
    ["a redirect", new RedirectedError(), true],
  ])("treats %s as %s", (_name, error, transient) => {
    expect(isTransientBootstrapFailure(error)).toBe(transient)
  })
})

describe("retryDelayMs", () => {
  it("doubles from one second and stops at thirty", () => {
    const ceilings = [1, 2, 3, 4, 5, 6, 7, 10].map((retry) =>
      retryDelayMs(retry, () => 1),
    )
    expect(ceilings).toEqual([
      1_000, 2_000, 4_000, 8_000, 16_000, 30_000, 30_000, 30_000,
    ])
  })

  it("jitters within the upper half of the ceiling", () => {
    expect(retryDelayMs(3, () => 0)).toBe(2_000)
    expect(retryDelayMs(3, () => 0.5)).toBe(3_000)
  })
})

describe("describeBootFailure", () => {
  function failed(
    error: unknown,
    overrides: Partial<FailedState> = {},
  ): FailedState {
    return {
      status: "failed",
      error,
      stage: "load",
      isAttempting: false,
      willRetry: isTransientBootstrapFailure(error),
      timedOut: false,
      offline: false,
      failedAt: 0,
      ...overrides,
    }
  }

  it.each<[string, FailedState, string, string, boolean]>([
    ["no response", failed(DOWN), "unreachable", "no response", true],
    [
      "something that is not an ApiError",
      failed(new TypeError("Failed to fetch")),
      "unreachable",
      "no response",
      true,
    ],
    [
      "its own deadline",
      failed(new ApiError(0, "The gateway did not respond in time."), {
        timedOut: true,
      }),
      "timedOut",
      "no answer in 8 s",
      true,
    ],
    [
      "a 502",
      failed(new ApiError(502, "Bad Gateway")),
      "serverError",
      "HTTP 502",
      true,
    ],
    ["a 503", failed(new ApiError(503, "x")), "serverError", "HTTP 503", true],
    ["a 408", failed(new ApiError(408, "x")), "serverError", "HTTP 408", true],
    ["a 429", failed(new ApiError(429, "x")), "serverError", "HTTP 429", true],
    [
      "an intermediary's non-JSON 200",
      failed(new ApiError(200, "not JSON")),
      "serverError",
      "HTTP 200",
      true,
    ],
    [
      "the device being offline",
      failed(DOWN, { offline: true }),
      "offline",
      "offline",
      true,
    ],
    [
      "offline, even past the deadline",
      failed(DOWN, { offline: true, timedOut: true }),
      "offline",
      "offline",
      true,
    ],
    [
      "a 401",
      failed(new ApiError(401, "Unauthorized")),
      "signInRequired",
      "HTTP 401",
      false,
    ],
    [
      "a redirect",
      failed(new RedirectedError()),
      "signInRequired",
      "sent to sign-in",
      true,
    ],
    [
      "a 403",
      failed(new ApiError(403, "Forbidden")),
      "accessRefused",
      "HTTP 403",
      false,
    ],
    [
      "a 404",
      failed(new ApiError(404, "Not Found")),
      "unexpectedAnswer",
      "HTTP 404",
      false,
    ],
    [
      "a 400",
      failed(new ApiError(400, "x")),
      "unexpectedAnswer",
      "HTTP 400",
      false,
    ],
    [
      "a deployment that could not be chosen",
      failed(new Error("no directory"), { stage: "prepare", willRetry: false }),
      "unchosen",
      "",
      false,
    ],
  ])("reports %s", (_name, state, kind, reason, willRetry) => {
    expect(describeBootFailure(state)).toEqual({ kind, reason, willRetry })
  })

  it("never says more than the browser can know", () => {
    // A failed fetch reports neither a refused connection nor who wrote a
    // status, so no reason may claim either.
    const reasons = [
      DOWN,
      new ApiError(502, "Bad Gateway"),
      new ApiError(401, "Unauthorized"),
      new RedirectedError(),
    ].map((error) => describeBootFailure(failed(error)).reason)
    for (const reason of reasons) {
      expect(reason).not.toMatch(/refused|proxy/i)
    }
  })
})

describe("readBootstrap", () => {
  afterEach(() => vi.restoreAllMocks())

  const OPAQUE = {
    type: "opaqueredirect",
    status: 0,
    ok: false,
    statusText: "",
    headers: new Headers(),
  } as Response

  it("follows a redirect that lands on the bootstrap, such as a canonical host or HTTPS", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(OPAQUE)
      .mockResolvedValueOnce(Response.json(BOOTSTRAP))

    await expect(readBootstrap(new AbortController().signal)).resolves.toEqual(
      BOOTSTRAP,
    )
    expect(String(fetchMock.mock.calls[0]?.[0])).toMatch(
      /\/api\/v1\/bootstrap$/,
    )
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({ redirect: "manual" })
    expect(fetchMock.mock.calls[1]?.[1]).toMatchObject({ redirect: "follow" })
  })

  it("reports a redirect that cannot be followed from here, as to a sign-in page on another origin", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(OPAQUE)
      // What a cross-origin redirect without CORS looks like when followed.
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      // Followed without CORS, the redirect's destination answers: an
      // opaque response, which a page that is down cannot give.
      .mockResolvedValueOnce({
        type: "opaque",
        status: 0,
        ok: false,
      } as Response)

    await expect(
      readBootstrap(new AbortController().signal),
    ).rejects.toBeInstanceOf(RedirectedError)
    expect(String(fetchMock.mock.calls[2]?.[0])).toMatch(
      /\/api\/v1\/bootstrap$/,
    )
    expect(fetchMock.mock.calls[2]?.[1]).toMatchObject({
      mode: "no-cors",
      redirect: "follow",
    })
  })

  it("calls a redirect it cannot follow unreachable when its destination does not answer either", async () => {
    // A canonical host that is down, or a cached 301 replayed in front of a
    // gateway that is down, fails the same way a cross-origin sign-in does,
    // until something checks whether the destination answers at all.
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(OPAQUE)
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))

    const error = await readBootstrap(new AbortController().signal).catch(
      (caught: unknown) => caught,
    )
    expect(error).not.toBeInstanceOf(RedirectedError)
    expect(error).toMatchObject({ status: 0 })
  })

  it("reports a redirect that lands on a page rather than the bootstrap", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(OPAQUE)
      .mockResolvedValueOnce(
        new Response("<html>Sign in</html>", {
          status: 200,
          headers: { "Content-Type": "text/html" },
        }),
      )

    await expect(
      readBootstrap(new AbortController().signal),
    ).rejects.toBeInstanceOf(RedirectedError)
  })

  it("passes on a refusal the redirect led to, rather than calling it a sign-in", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(OPAQUE)
      .mockResolvedValueOnce(Response.json({ detail: "x" }, { status: 503 }))

    await expect(
      readBootstrap(new AbortController().signal),
    ).rejects.toMatchObject({ status: 503 })
  })
})
