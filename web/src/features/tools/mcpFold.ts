import type { McpServerFoldCandidate } from "@/client"

// Folding a workspace's own copy of a server into its organization's: the
// copy's access rules become that workspace's settings on the organization's
// server, and its id keeps working there. A sign-in made through the
// organization server's own client is kept, every other one ends, and a token
// or shared account of the copy's own gives way to the organization's.

/** The most copies one add or promotion folds: the gateway refuses a longer list. */
export const MAX_FOLD_COPIES = 50

/** The copies a fold step ticks before anyone touches it: those whose calls would use the same account. */
export function defaultFoldIds(
  candidates: readonly McpServerFoldCandidate[],
): string[] {
  return candidates
    .filter((candidate) => candidate.same_credential)
    .map((candidate) => candidate.server_id)
}

function people(count: number): string {
  return count === 1 ? "1 person" : `${count} people`
}

/** What folding one copy would do to the people using it, one sentence each. */
export function foldEffects(candidate: McpServerFoldCandidate): string[] {
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
    lines.push(`${people(candidate.sign_in_again)} will need to sign in again.`)
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
 * Each workspace copy offered under one pick only, the first that found it:
 * two picks at one URL would otherwise both fold it, and the second add would
 * be refused for a copy the first already removed.
 */
export function offerEachCopyOnce(
  found: readonly (readonly McpServerFoldCandidate[])[],
): McpServerFoldCandidate[][] {
  const offered = new Set<string>()
  return found.map((candidates) =>
    candidates.filter((candidate) => {
      if (offered.has(candidate.server_id)) return false
      offered.add(candidate.server_id)
      return true
    }),
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
