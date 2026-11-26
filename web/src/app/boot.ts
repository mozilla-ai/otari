/**
 * The order the dashboard starts in: settle where the API is, then read the
 * deployment's bootstrap from there, and keep asking until it answers.
 *
 * A preparation that rejects, or does not settle within the attempt deadline,
 * is a deployment that could not be chosen. Loading the bootstrap anyway would
 * read it from wherever the request policy happened to point, which is a
 * deployment nobody selected (web/AGENTS.md: do not guess a deployment when
 * that request fails), so it fails the boot instead. It is not
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

import {
  ApiError,
  apiFetch,
  RedirectedError,
  reachesAfterRedirects,
} from "@/shared/api/client"
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
      /** The attempt was cut off by its own deadline rather than answered. */
      timedOut: boolean
      /** The browser said it was offline when the attempt failed. */
      offline: boolean
      /** When the attempt failed, in epoch milliseconds. */
      failedAt: number
    }

export type FailedState = Extract<BootState, { status: "failed" }>

export interface BootLoader {
  getState(): BootState
  subscribe(listener: () => void): () => void
  /**
   * Ask again now, cutting any scheduled wait short. Resolves when that attempt
   * settles, either way; an attempt already running is adopted rather than
   * stacked on, so the promise is that attempt's.
   */
  retryNow(): Promise<void>
  /** Stop retrying and drop every timer and listener. */
  dispose(): void
}

/**
 * One read of the bootstrap, telling a redirect to a sign-in page from one
 * that only moves the request.
 *
 * The gateway's own bootstrap route does not redirect, but what is in front
 * of it may, for two different reasons. A canonical host, an HTTPS upgrade or
 * a path rewrite lands on the bootstrap and should just be followed. An access
 * proxy sending this browser to its sign-in page should be named: followed
 * blindly, one to another origin fails on CORS and reads as a gateway that
 * never answered, retried forever. So the first read does not follow, and a
 * redirect is then followed once: an answer is the bootstrap, and a page that
 * is not JSON is a sign-in, reported as a `RedirectedError`. Any other refusal
 * the redirect led to passes through as itself, and so does the deadline.
 *
 * A followed redirect whose fetch fails is the ambiguous case. A cross-origin
 * sign-in page fails that way, refused by CORS, but so does a canonical host
 * that is down, or a gateway that is down behind a redirect the browser cached
 * and replays without asking. Following the same redirects without CORS tells
 * them apart: if anything answers at the end, it is up and only refused to
 * share, which is a sign-in; if nothing answers, the failure stands as the
 * gateway being unreachable, to be retried.
 */
export async function readBootstrap(
  signal: AbortSignal,
): Promise<WireBootstrap> {
  try {
    return await apiFetch<WireBootstrap>("/bootstrap", {
      signal,
      redirect: "manual",
    })
  } catch (error) {
    if (!(error instanceof RedirectedError)) {
      throw error
    }
  }
  try {
    return await apiFetch<WireBootstrap>("/bootstrap", {
      signal,
      redirect: "follow",
    })
  } catch (error) {
    if (signal.aborted || !(error instanceof ApiError)) {
      throw error
    }
    // A 2xx here can only be a body that was not JSON: a page in the
    // bootstrap's place.
    if (error.status >= 200 && error.status < 300) {
      throw new RedirectedError()
    }
    if (error.status !== 0) {
      throw error
    }
    try {
      await reachesAfterRedirects("/bootstrap", signal)
    } catch {
      throw error
    }
    throw new RedirectedError()
  }
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
  // Something in front of the gateway wants this browser somewhere else. The
  // screen asks for a reload, but the loader keeps asking quietly: from here a
  // sign-in page and a host that answered without CORS can look the same, and
  // only the second one recovers on its own.
  if (error instanceof RedirectedError) {
    return true
  }
  if (!(error instanceof ApiError)) {
    return true
  }
  if (error.status === 408 || error.status === 429) {
    return true
  }
  return error.status < 400 || error.status >= 500
}

/**
 * Which failure the boot screen is showing. Each one picks its own wording and
 * action; the four that keep retrying share theirs and differ only in `reason`.
 */
export type BootFailureKind =
  | "unreachable"
  | "timedOut"
  | "serverError"
  | "offline"
  | "signInRequired"
  | "accessRefused"
  | "unexpectedAnswer"
  | "unchosen"

export interface BootFailure {
  kind: BootFailureKind
  /**
   * What the last attempt got, in the only terms the browser can know it: no
   * response, its own deadline, a status, the device being offline, or a
   * redirect. Empty where no gateway was asked. Never a cause the browser
   * cannot see, such as a refused connection or who wrote a status: a fetch
   * that fails reports neither.
   */
  reason: string
  /** Whether the loader keeps trying on its own. */
  willRetry: boolean
}

/**
 * The failure the screen reports for a failed state.
 *
 * The bootstrap takes no credential, so a 401, a 403 or a redirect comes from
 * something in front of the gateway rather than the gateway itself, and the
 * way past it is that thing's own sign-in, which a reload reaches and a retry
 * from here does not.
 */
export function describeBootFailure(state: FailedState): BootFailure {
  const { error, stage, willRetry } = state
  if (stage === "prepare") {
    return { kind: "unchosen", reason: "", willRetry }
  }
  if (error instanceof RedirectedError) {
    return {
      kind: "signInRequired",
      reason: "sent to sign-in",
      willRetry,
    }
  }
  const status = error instanceof ApiError ? error.status : 0
  if (status === 401) {
    return { kind: "signInRequired", reason: "HTTP 401", willRetry }
  }
  if (status === 403) {
    return { kind: "accessRefused", reason: "HTTP 403", willRetry }
  }
  if (status !== 0 && !willRetry) {
    return { kind: "unexpectedAnswer", reason: `HTTP ${status}`, willRetry }
  }
  if (status !== 0) {
    // 5xx, 408, 429, and a non-JSON answer from an intermediary: a status
    // arrived, so the gateway's address is answering something.
    return { kind: "serverError", reason: `HTTP ${status}`, willRetry }
  }
  if (state.offline) {
    return { kind: "offline", reason: "offline", willRetry }
  }
  if (state.timedOut) {
    return {
      kind: "timedOut",
      reason: `no answer in ${BOOTSTRAP_ATTEMPT_TIMEOUT_MS / 1000} s`,
      willRetry,
    }
  }
  return { kind: "unreachable", reason: "no response", willRetry }
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

/**
 * `promise`, or a `TimeoutError` once the attempt deadline passes. For the
 * preparation, which has no signal to abort: a choice of gateway that never
 * settles would otherwise hold the loading screen up indefinitely.
 */
function withDeadline<T>(promise: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(
      () =>
        reject(
          new DOMException("Choosing a gateway timed out.", "TimeoutError"),
        ),
      BOOTSTRAP_ATTEMPT_TIMEOUT_MS,
    )
  })
  return Promise.race([promise, deadline]).finally(() => clearTimeout(timer))
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
  let current: Promise<void> | undefined
  // The preparation in flight, kept across attempts: one that outlasted an
  // attempt's deadline is waited on again by the next attempt rather than
  // started a second time, since the replacing module's contract is that it
  // runs once before the bootstrap is read.
  let preparing: Promise<void> | undefined
  // The browser came back online, or the tab was shown again, while an
  // attempt was running: if that attempt fails, the next one starts at once.
  let wokeDuringAttempt = false

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
  // waits for the operator's Retry. An attempt already running is left to
  // finish rather than stacked on, and the wake is kept for when it does.
  function retryOnWake() {
    if (isAttempting) {
      wokeDuringAttempt = true
      return
    }
    if (state.status === "failed" && state.willRetry) {
      void attempt()
    }
  }

  function stopListening() {
    window.removeEventListener("online", retryOnWake)
    document.removeEventListener("visibilitychange", onWake)
  }

  function fail(
    error: unknown,
    stage: "prepare" | "load",
    willRetry: boolean,
    timedOut = false,
  ) {
    if (willRetry) {
      retries += 1
      retryTimer = setTimeout(
        attempt,
        wokeDuringAttempt ? 0 : retryDelayMs(retries, random),
      )
    }
    wokeDuringAttempt = false
    setState({
      status: "failed",
      error,
      stage,
      isAttempting: false,
      willRetry,
      timedOut,
      offline: navigator.onLine === false,
      failedAt: Date.now(),
    })
  }

  // One attempt at a time: a second call while one runs gets the running one.
  // Keyed on `isAttempting` rather than on the promise, which settles a
  // microtask after the attempt's state is published: a press that lands in
  // between belongs to a new attempt, not to the one that just failed. The
  // new attempt's promise is in place before anyone is told it started, so a
  // press made from that notification joins it.
  function attempt(): Promise<void> {
    if (isDisposed || state.status === "ready") {
      return Promise.resolve()
    }
    if (isAttempting && current) {
      return current
    }
    clearRetryTimer()
    isAttempting = true
    wokeDuringAttempt = false
    current = run()
    if (state.status === "failed") {
      setState({ ...state, isAttempting: true })
    }
    return current
  }

  async function run() {
    // Lets `attempt` publish the promise before anything here runs.
    await Promise.resolve()
    if (!isPrepared) {
      try {
        preparing ??= prepare().then(
          () => {
            isPrepared = true
          },
          (error: unknown) => {
            preparing = undefined
            throw error
          },
        )
        await withDeadline(preparing)
      } catch (error) {
        isAttempting = false
        if (!isDisposed) {
          fail(error, "prepare", false)
        }
        return
      }
      if (isDisposed) {
        isAttempting = false
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
    let outcome:
      | { ok: true; bootstrap: WireBootstrap }
      | { ok: false; error: unknown }
    try {
      outcome = { ok: true, bootstrap: await load(controller.signal) }
    } catch (error) {
      outcome = { ok: false, error }
    }
    // Settled before anyone hears about it, so a listener that asks again
    // starts a new attempt rather than adopting this one.
    clearTimeout(deadline)
    inFlight = undefined
    isAttempting = false
    if (isDisposed) {
      return
    }
    if (outcome.ok) {
      stopListening()
      setState({ status: "ready", bootstrap: outcome.bootstrap })
      return
    }
    // Only the deadline aborts this controller on a loader that is still live
    // (`dispose` aborts it too, and is excluded above), so an aborted signal
    // here means the deadline fired.
    fail(
      outcome.error,
      "load",
      isTransientBootstrapFailure(outcome.error),
      controller.signal.aborted,
    )
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
    retryNow: attempt,
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
