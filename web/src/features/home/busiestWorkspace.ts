/**
 * The workspace a row spent the most in, from each candidate's cost, where
 * `undefined` is a read that failed. With none answered there is no answer,
 * so the caller stays in the workspace it is in.
 */
export function busiestWorkspace(
  candidates: readonly string[],
  costs: readonly (number | undefined)[],
): string | undefined {
  let best: { id: string; cost: number } | undefined
  candidates.forEach((id, index) => {
    const cost = costs[index]
    if (cost !== undefined && (!best || cost > best.cost)) best = { id, cost }
  })
  return best?.id
}
