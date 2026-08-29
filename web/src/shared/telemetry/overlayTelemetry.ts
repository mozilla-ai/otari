import {
  isLocalDashboard,
  readMixpanelToken,
} from "@/shared/telemetry/mixpanelToken"
import type { Telemetry } from "@/shared/telemetry/types"

/**
 * The tracker the dashboard ships.
 *
 * Gated by key presence, not by edition. `VITE_MIXPANEL_TOKEN` is inlined at
 * build time. Without a token this module records nothing, loads no SDK, and
 * (on a local dashboard only) logs `Mixpanel not initialized` once. With a
 * token it dynamically imports `mixpanelClient`, which is the only module
 * that may touch `mixpanel-browser`.
 *
 * **Reach it by its `@/…` specifier.** An overlay may still replace this file
 * at build time. A relative import of the seam would miss that alias. Same
 * pattern as `src/app/nav/overlaySections.ts`.
 *
 * The contract lives in `types.ts` and the names in `events.ts`. Neither is
 * swapped. This file imports the contract as `@/shared/telemetry/types` so a
 * replacement copied into an overlay tree still resolves it.
 */

const UNINITIALIZED_MESSAGE = "Mixpanel not initialized"

const NO_TELEMETRY: Telemetry = {
  // Not "denied": nothing has been asked and nothing decided. A missing key
  // is not a stored refusal; it is the absence of a tracker.
  consent: "unknown",
  recordEvent: () => undefined,
  identify: () => undefined,
}

export async function loadMixpanelClient(token: string): Promise<Telemetry> {
  const { createMixpanelTelemetry } = await import(
    "@/shared/telemetry/mixpanelClient"
  )
  return createMixpanelTelemetry(token)
}

function createQueuedTelemetry(
  token: string,
  loadClient: (token: string) => Promise<Telemetry>,
): Telemetry {
  let client: Telemetry | undefined
  let failed = false
  const pending: Array<(loaded: Telemetry) => void> = []

  void loadClient(token)
    .then((loaded) => {
      if (failed) {
        return
      }
      client = loaded
      for (const apply of pending) {
        apply(loaded)
      }
      pending.length = 0
    })
    .catch(() => {
      failed = true
      pending.length = 0
      client = NO_TELEMETRY
    })

  const enqueue = (apply: (loaded: Telemetry) => void): void => {
    if (failed) {
      return
    }
    if (client) {
      apply(client)
      return
    }
    pending.push(apply)
  }

  return {
    consent: "granted",
    recordEvent: (event, properties) => {
      enqueue((loaded) => {
        loaded.recordEvent(event, properties)
      })
    },
    identify: (identity) => {
      enqueue((loaded) => {
        loaded.identify(identity)
      })
    },
  }
}

export function createTelemetry(
  token: string | undefined = readMixpanelToken(),
  {
    isLocal = isLocalDashboard(),
    loadClient = loadMixpanelClient,
    log = (message: string): void => {
      console.info(message)
    },
  }: {
    isLocal?: boolean
    loadClient?: (token: string) => Promise<Telemetry>
    log?: (message: string) => void
  } = {},
): Telemetry {
  if (token === undefined) {
    if (isLocal) {
      log(UNINITIALIZED_MESSAGE)
    }
    return NO_TELEMETRY
  }
  return createQueuedTelemetry(token, loadClient)
}

let cached: Telemetry | undefined

/**
 * A hook rather than bare functions, for the two reasons the platform's
 * `useTracking` is one: `consent` can change while the app is running, so a
 * caller reading it has to re-render, and a replacement needs somewhere to hold
 * the client it initializes. The returned object is referentially stable here,
 * and a replacement owes the same: `TelemetryIdentity.tsx` has it in an effect's
 * dependencies, and a fresh object each render would re-identify forever.
 */
export function useTelemetry(): Telemetry {
  cached ??= createTelemetry()
  return cached
}
