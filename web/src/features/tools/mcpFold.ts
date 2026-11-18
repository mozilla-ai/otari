import type { McpServerFoldCandidate } from "@/client"

// Folding a workspace's own copy of a server into its organization's: the
// copy's access rules become that workspace's settings on the organization's
// server, and its id keeps working there. A sign-in made through the
// organization server's own client is kept, every other one ends, and a token
// or shared account of the copy's own gives way to the organization's.

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
  if (candidate.sign_in_again > 0) {
    lines.push(`${people(candidate.sign_in_again)} will need to sign in again.`)
  }
  if (candidate.replaces_shared_credential) {
    lines.push(
      "Its shared account or token will be replaced by the organization's.",
    )
  }
  if (lines.length === 0) lines.push("No one's sign-in changes.")
  return lines
}
