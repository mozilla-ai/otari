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
 * Separate from `main.tsx` so the order is a function a test can call; that
 * file renders on import.
 */

import type { WireBootstrap } from "@/shared/helpers/bootstrap"

export async function loadDeployment(
  prepare: () => Promise<void>,
  load: () => Promise<WireBootstrap | null>,
): Promise<WireBootstrap | null> {
  try {
    await prepare()
  } catch {
    return null
  }
  return load()
}
