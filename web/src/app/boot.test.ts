import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ApiError } from "@/shared/api/client"
import type { WireBootstrap } from "@/shared/helpers/bootstrap"
import {
  BOOTSTRAP_ATTEMPT_TIMEOUT_MS,
  type BootLoader,
  createBootLoader,
  firstSettled,
  isTransientBootstrapFailure,
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

describe("createBootLoader", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    loader?.dispose()
    loader = undefined
    vi.useRealTimers()
    vi.restoreAllMocks()
    setVisibility("visible")
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
    expect(boot.getState()).toMatchObject({ status: "failed", willRetry: true })

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
