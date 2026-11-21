import type { McpCredentialMode, McpServerFoldCandidate } from "@/client"

// Folding a workspace's own copy of a server into its organization's: the
// copy's access rules become that workspace's settings on the organization's
// server, and its id keeps working there. A sign-in made through the
// organization server's own client is kept, every other one ends, and a token
// or shared account of the copy's own gives way to the organization's.

/** The most copies one add or promotion folds: the gateway refuses a longer list. */
export const MAX_FOLD_COPIES = 50

/**
 * Why a copy cannot be folded into a server that is on or off as
 * `targetEnabled` says, or "" where it can: no workspace can turn an
 * organization server back on, so one that is off cannot replace a copy that
 * is on.
 */
export function foldBlocker(
  candidate: McpServerFoldCandidate,
  targetEnabled: boolean,
): string {
  return candidate.enabled && !targetEnabled
    ? "It is on, and an organization server that is off cannot take its place."
    : ""
}

/** How many copies of each workspace a server that is on or off as `targetEnabled` says can fold. */
export function foldableByWorkspace(
  candidates: readonly McpServerFoldCandidate[],
  targetEnabled: boolean,
): Map<string, number> {
  const counts = new Map<string, number>()
  for (const candidate of candidates) {
    if (foldBlocker(candidate, targetEnabled) !== "") continue
    counts.set(
      candidate.workspace_id,
      (counts.get(candidate.workspace_id) ?? 0) + 1,
    )
  }
  return counts
}

/**
 * The copies a fold step ticks before anyone touches it: those it can fold
 * whose calls would use the same account, with nobody signing in again or
 * switching to another sign-in. A workspace folds one copy, so one that has
 * several it could fold gets none ticked, for the admin to pick.
 */
export function defaultFoldIds(
  candidates: readonly McpServerFoldCandidate[],
  targetEnabled: boolean,
): string[] {
  const foldable = foldableByWorkspace(candidates, targetEnabled)
  return candidates
    .filter(
      (candidate) =>
        foldable.get(candidate.workspace_id) === 1 &&
        foldBlocker(candidate, targetEnabled) === "" &&
        candidate.same_credential &&
        candidate.switch_account === 0 &&
        candidate.sign_in_again === 0,
    )
    .map((candidate) => candidate.server_id)
}

/** Of the ticked ids, those a fold can send: still listed, and foldable into the server as it is. */
export function foldableIds(
  candidates: readonly McpServerFoldCandidate[],
  selected: readonly string[],
  targetEnabled: boolean,
): string[] {
  return candidates
    .filter(
      (candidate) =>
        selected.includes(candidate.server_id) &&
        foldBlocker(candidate, targetEnabled) === "",
    )
    .map((candidate) => candidate.server_id)
}

function people(count: number): string {
  return count === 1 ? "1 person" : `${count} people`
}

/**
 * What folding one copy into a server of `targetMode` would do to the people
 * using it, one sentence each.
 */
export function foldEffects(
  candidate: McpServerFoldCandidate,
  targetMode: McpCredentialMode,
): string[] {
  const lines: string[] = []
  if (candidate.keep_sign_in > 0) {
    lines.push(`${people(candidate.keep_sign_in)} will keep their sign-in.`)
  }
  if (candidate.switch_account > 0) {
    lines.push(
      `${people(candidate.switch_account)} will use the account they already signed in with on the organization's server.`,
    )
  }
  if (candidate.sign_in_again > 0) {
    // A shared server signs nobody in: their own sign-ins end, and its credential serves them.
    lines.push(
      targetMode === "per_user"
        ? `${people(candidate.sign_in_again)} will need to sign in again.`
        : `${people(candidate.sign_in_again)} will stop using their own sign-in, and use the organization's account instead.`,
    )
  }
  if (candidate.replaces_shared_credential) {
    lines.push(
      "Its shared account or token will be replaced by the organization's.",
    )
  } else if (!candidate.same_credential) {
    // No credential of its own, folding into a server that signs in another way or brings a token.
    lines.push("The account its calls use will change.")
  }
  if (lines.length === 0) lines.push("No one's sign-in changes.")
  return lines
}

/**
 * Each workspace copy offered under one pick only: two picks at one URL would
 * otherwise both fold it, and the second add would be refused for a copy the
 * first already removed. The first pick that can fold it gets it, by whether
 * each pick's server is on (`enabled`), else the first that found it, where
 * it is shown with why it cannot be folded.
 */
export function offerEachCopyOnce(
  found: readonly (readonly McpServerFoldCandidate[])[],
  enabled: readonly boolean[],
): McpServerFoldCandidate[][] {
  const home = new Map<string, number>()
  found.forEach((candidates, index) => {
    for (const candidate of candidates) {
      const first = home.get(candidate.server_id)
      if (
        first === undefined ||
        (foldBlocker(candidate, enabled[first]) !== "" &&
          foldBlocker(candidate, enabled[index]) === "")
      ) {
        home.set(candidate.server_id, index)
      }
    }
  })
  return found.map((candidates, index) =>
    candidates.filter((candidate) => home.get(candidate.server_id) === index),
  )
}

/**
 * Runs `task` over `items` with at most `limit` in flight, and reports each
 * one's outcome in order, so one failure neither stops nor discards the rest.
 */
export async function settleAtMost<Item, Value>(
  items: readonly Item[],
  limit: number,
  task: (item: Item) => Promise<Value>,
): Promise<PromiseSettledResult<Value>[]> {
  const outcomes: PromiseSettledResult<Value>[] = new Array(items.length)
  let next = 0
  const worker = async () => {
    while (next < items.length) {
      const index = next++
      try {
        outcomes[index] = {
          status: "fulfilled",
          value: await task(items[index]),
        }
      } catch (reason) {
        outcomes[index] = { status: "rejected", reason }
      }
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, worker),
  )
  return outcomes
}
