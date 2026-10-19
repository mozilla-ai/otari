/**
 * The order the dashboard starts in: settle where the API is, then read the
 * deployment's bootstrap from there.
 *
 * A preparation that rejects is a deployment that could not be chosen, and the
 * answer for that is the same as for a bootstrap that never arrived: `null`,
 * which `App` renders as the gateway being unreachable rather than as a guess.
 * Loading the bootstrap anyway would read it from wherever the request policy
 * happened to point, which is a deployment nobody selected (web/AGENTS.md: do
 * not guess a deployment when that request fails).
 *
 * A bootstrap kept from an earlier load renders first, while the current one is
 * read, and the current one renders again when it differs. One that never
 * arrives leaves the kept answer on screen, where each page's own requests
 * report their failure, and forgets it, so the next load waits and says so.
 *
 * Separate from `main.tsx` so the order is a function a test can call; that
 * file renders on import.
 */

import type { WireBootstrap } from "@/shared/helpers/bootstrap"

export interface KeptBootstrap {
  read(): WireBootstrap | null
  write(bootstrap: WireBootstrap | null): void
}

export async function startDeployment(
  prepare: () => Promise<void>,
  load: () => Promise<WireBootstrap | null>,
  keep: () => KeptBootstrap,
  render: (bootstrap: WireBootstrap | null) => void,
): Promise<void> {
  try {
    await prepare()
  } catch {
    render(null)
    return
  }
  // Taken before the request goes out, so the answer is kept under the origin
  // it was read from even if the request policy moves while it is in flight.
  const kept = keep()
  const current = load()
  const earlier = kept.read()
  if (earlier) render(earlier)
  const bootstrap = await current
  kept.write(bootstrap)
  const isDifferentAnswer =
    bootstrap !== null && JSON.stringify(bootstrap) !== JSON.stringify(earlier)
  if (!earlier || isDifferentAnswer) render(bootstrap)
}

const KEPT_BOOTSTRAP_KEY = "otari.dashboard.bootstrap"

/**
 * The last bootstrap read from the API at `origin`, kept in this browser.
 *
 * Only a hosted deployment's: it is the one reached across the internet, where
 * the round trip is worth saving, and the one that never changes mode under the
 * same address. A local gateway restarted in another mode would otherwise
 * paint the previous mode's screen first.
 */
export function openKeptBootstrap(origin: string): KeptBootstrap {
  return {
    read() {
      try {
        const kept = JSON.parse(
          window.localStorage.getItem(KEPT_BOOTSTRAP_KEY) ?? "null",
        ) as { origin?: unknown; bootstrap?: WireBootstrap } | null
        return kept?.origin === origin &&
          kept.bootstrap?.deployment_type === "hosted"
          ? kept.bootstrap
          : null
      } catch {
        return null
      }
    },
    write(bootstrap) {
      try {
        if (bootstrap?.deployment_type === "hosted") {
          window.localStorage.setItem(
            KEPT_BOOTSTRAP_KEY,
            JSON.stringify({ origin, bootstrap }),
          )
        } else {
          window.localStorage.removeItem(KEPT_BOOTSTRAP_KEY)
        }
      } catch {
        // Private mode or a full store: the next load waits for the bootstrap.
      }
    },
  }
}
