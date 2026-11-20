/**
 * The order the dashboard starts in: settle where the API is, then read the
 * deployment's bootstrap from there, and keep asking until it answers.
 *
 * A preparation that rejects is a deployment that could not be chosen. Loading
 * the bootstrap anyway would read it from wherever the request policy happened
 * to point, which is a deployment nobody selected (web/AGENTS.md: do not guess a
 * deployment when that request fails), so it fails the boot instead. It is not
 * retried on a timer: the replacing module owns how its choice is made, and a
 * refusal from it is an answer rather than a stall. The Retry control asks it
 * again.
 *
 * The bootstrap read itself is retried. A gateway whose database stalls can
 * take longer than a per-attempt deadline to answer and then recover, and a
 * dashboard that read once would sit on its failure screen until someone
 * reloaded. Not a TanStack query: it is read once per page load and never
 * refetched once answered, the client that would own it is mounted beneath the
 * answer, and its retryer cannot cut a backoff short when the browser comes
 * back online, which is the moment an answer is most likely.
 *
 * Separate from `main.tsx` so the sequence is something a test can drive; that
 * file renders on import.
 */

import { ApiError } from "@/shared/api/client"
import type { WireBootstrap } from "@/shared/helpers/bootstrap"

/**
 * Per attempt, and shorter than `apiFetch`'s 30s default, because nothing at
 * all is on screen until the first attempt settles. A healthy gateway answers
 * at once, since the route reads configuration plus two primary-key lookups.
 * A merely slow or pool-starved database can outlast it (`db_pool_timeout`
 * defaults to 30s), which is what the retries are for.
 */
export const BOOTSTRAP_ATTEMPT_TIMEOUT_MS = 8_000

const FIRST_RETRY_DELAY_MS = 1_000
const MAX_RETRY_DELAY_MS = 30_000

export type BootState =
  | { status: "loading" }
  | { status: "ready"; bootstrap: WireBootstrap }
  | {
      status: "failed"
      error: unknown
      /**
       * Where it failed: choosing which gateway to ask (`prepare`, so no gateway
       * was asked at all) or reading that gateway's bootstrap (`load`).
       */
      stage: "prepare" | "load"
      /** An attempt is in flight right now. */
      isAttempting: boolean
      /** Another attempt will start on its own (a transient failure). */
      willRetry: boolean
    }

export interface BootLoader {
  getState(): BootState
  subscribe(listener: () => void): () => void
  /** Ask again now, cutting any scheduled wait short. */
  retryNow(): void
  /** Stop retrying and drop every timer and listener. */
  dispose(): void
}

/**
 * Whether asking again could plausibly get a different answer. A refusal the
 * gateway (or something in front of it) chose to give is not, except the two
 * that say "later": 408 and 429. A network fault or a timeout (status 0), a 5xx,
 * a non-JSON 200 from an intermediary answering in the gateway's place, and
 * anything that is not an `ApiError` at all are. The non-JSON case is retried on
 * purpose even though a static host serving the app's own page for every path
 * never stops doing so: an edge answering for a gateway mid-redeploy looks the
 * same, and that one does recover. The backoff caps what the other costs.
 */
export function isTransientBootstrapFailure(error: unknown): boolean {
  if (!(error instanceof ApiError)) {
    return true
  }
  if (error.status === 408 || error.status === 429) {
    return true
  }
  return error.status < 400 || error.status >= 500
}

/**
 * The wait before retry number `retry` (1-based): doubling from 1s, capped at
 * 30s, with half of it jittered so a fleet of tabs that lost the same gateway
 * does not come back to it in lockstep.
 */
export function retryDelayMs(retry: number, random: () => number): number {
  const ceiling = Math.min(
    MAX_RETRY_DELAY_MS,
    FIRST_RETRY_DELAY_MS * 2 ** Math.max(0, retry - 1),
  )
  return Math.round(ceiling / 2 + (random() * ceiling) / 2)
}

export function createBootLoader({
  prepare,
  load,
  random = Math.random,
}: {
  prepare: () => Promise<void>
  /** One read of the bootstrap. Rejects on any failure. */
  load: (signal: AbortSignal) => Promise<WireBootstrap>
  random?: () => number
}): BootLoader {
  let state: BootState = { status: "loading" }
  const listeners = new Set<() => void>()
  let isPrepared = false
  let isAttempting = false
  let isDisposed = false
  let retries = 0
  let retryTimer: ReturnType<typeof setTimeout> | undefined
  let inFlight: AbortController | undefined

  function setState(next: BootState) {
    state = next
    for (const listener of listeners) {
      listener()
    }
  }

  function clearRetryTimer() {
    if (retryTimer !== undefined) {
      clearTimeout(retryTimer)
      retryTimer = undefined
    }
  }

  function onWake() {
    if (document.visibilityState === "visible") {
      retryOnWake()
    }
  }

  // Only a failure that would be retried anyway is cut short: a definitive one
  // waits for the operator's Retry, and an attempt already running is left to
  // finish rather than stacked on.
  function retryOnWake() {
    if (state.status === "failed" && state.willRetry && !isAttempting) {
      attempt()
    }
  }

  function stopListening() {
    window.removeEventListener("online", retryOnWake)
    document.removeEventListener("visibilitychange", onWake)
  }

  function fail(error: unknown, stage: "prepare" | "load", willRetry: boolean) {
    if (willRetry) {
      retries += 1
      retryTimer = setTimeout(attempt, retryDelayMs(retries, random))
    }
    setState({ status: "failed", error, stage, isAttempting: false, willRetry })
  }

  async function attempt() {
    if (isDisposed || isAttempting || state.status === "ready") {
      return
    }
    clearRetryTimer()
    isAttempting = true
    if (state.status === "failed") {
      setState({ ...state, isAttempting: true })
    }

    if (!isPrepared) {
      try {
        await prepare()
        isPrepared = true
      } catch (error) {
        isAttempting = false
        if (!isDisposed) {
          fail(error, "prepare", false)
        }
        return
      }
    }

    const controller = new AbortController()
    inFlight = controller
    const deadline = setTimeout(
      () =>
        controller.abort(
          new DOMException("The bootstrap read timed out.", "TimeoutError"),
        ),
      BOOTSTRAP_ATTEMPT_TIMEOUT_MS,
    )
    try {
      const bootstrap = await load(controller.signal)
      if (isDisposed) {
        return
      }
      stopListening()
      setState({ status: "ready", bootstrap })
    } catch (error) {
      if (!isDisposed) {
        fail(error, "load", isTransientBootstrapFailure(error))
      }
    } finally {
      clearTimeout(deadline)
      inFlight = undefined
      isAttempting = false
    }
  }

  window.addEventListener("online", retryOnWake)
  document.addEventListener("visibilitychange", onWake)
  void attempt()

  return {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    retryNow() {
      void attempt()
    },
    dispose() {
      isDisposed = true
      clearRetryTimer()
      inFlight?.abort()
      stopListening()
      listeners.clear()
    },
  }
}

/** Resolves once the first attempt has settled, either way. */
export function firstSettled(loader: BootLoader): Promise<BootState> {
  return new Promise((resolve) => {
    if (loader.getState().status !== "loading") {
      resolve(loader.getState())
      return
    }
    const unsubscribe = loader.subscribe(() => {
      if (loader.getState().status !== "loading") {
        unsubscribe()
        resolve(loader.getState())
      }
    })
  })
}
