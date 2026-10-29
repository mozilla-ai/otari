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
 * Where the request policy keeps the bootstrap (`keptAnswers.bootstrap`), the
 * one read on the previous load renders while the current one is read, and the
 * current one renders again only when it differs. A current one that never
 * arrives does not take the screen from a kept one, which may already be in
 * use; each page's own requests report the failure, and forgetting the kept
 * one makes the next load wait and say so.
 *
 * Separate from `main.tsx` so the order is a function a test can call; that
 * file renders on import.
 */

import {
  oauthCallbackProvider,
  publicAuthPath,
} from "@/features/auth/publicAuthPaths"
import {
  forgetKeptAnswer,
  type KeptAnswer,
  keptAnswer,
  markKeptAnswer,
} from "@/shared/api/keptAnswers"
import type { RequestPolicy } from "@/shared/api/requestPolicy"
import type { WireBootstrap } from "@/shared/helpers/bootstrap"

export type KeptBootstrap = KeptAnswer<WireBootstrap>

export async function startDeployment(
  prepare: () => Promise<void>,
  load: () => Promise<WireBootstrap | null>,
  keep: () => KeptBootstrap | null,
  render: (bootstrap: WireBootstrap | null) => void,
): Promise<void> {
  try {
    await prepare()
  } catch {
    render(null)
    return
  }
  // Opened before the request goes out, so the answer is kept under the origin
  // it was read from even if the request policy moves while it is in flight.
  const kept = keep()
  const current = load()
  const earlier = kept?.read()
  markKeptAnswer(
    `boot:kept-bootstrap:${kept ? (earlier ? "hit" : "miss") : "off"}`,
  )
  if (earlier) render(earlier)
  const bootstrap = await current
  kept?.write(bootstrap ?? undefined)
  if (!earlier) {
    render(bootstrap)
    return
  }
  if (bootstrap !== null && !isSameBootstrap(bootstrap, earlier)) {
    markKeptAnswer("boot:remount")
    render(bootstrap)
  }
}

const KEPT_BOOTSTRAP_KEY = "otari.dashboard.bootstrap"

/**
 * The bootstrap this browser keeps for the API `policy` names, or `null` when
 * the policy keeps none, in which case an entry an earlier build kept is
 * forgotten so turning the switch off cleans up after it.
 *
 * Landing on an OAuth callback reads none: that page spends its code on mount,
 * and the current bootstrap replacing a kept one remounts it, which would
 * spend the code twice. It still keeps the current one for the next load.
 */
export function openKeptBootstrap(
  policy: RequestPolicy,
  hash: string,
): KeptBootstrap | null {
  if (policy.keptAnswers?.bootstrap !== true) {
    forgetKeptAnswer(KEPT_BOOTSTRAP_KEY)
    return null
  }
  const kept = keptAnswer<WireBootstrap>(KEPT_BOOTSTRAP_KEY, policy.origin)
  const landing = publicAuthPath(hash)
  if (landing !== null && oauthCallbackProvider(landing) !== null) {
    return { read: () => undefined, write: kept.write }
  }
  return kept
}

/** Equal field for field, whatever order either arrived in. */
function isSameBootstrap(a: WireBootstrap, b: WireBootstrap): boolean {
  return canonical(a) === canonical(b)
}

function canonical(value: unknown): string {
  return JSON.stringify(value, (_key, field: unknown) =>
    typeof field === "object" && field !== null && !Array.isArray(field)
      ? Object.fromEntries(
          Object.entries(field).sort(([x], [y]) => (x < y ? -1 : 1)),
        )
      : field,
  )
}
