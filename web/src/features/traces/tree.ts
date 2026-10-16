import type { Observation } from "./types"

export interface TreeNode {
  obs: Observation
  children: TreeNode[]
  depth: number
}

export interface FlatItem {
  node: TreeNode
  /** For each ancestor level, whether a vertical connector continues past this row. */
  treeLines: boolean[]
  isLast: boolean
}

export function buildTree(observations: Observation[]): TreeNode[] {
  const byId = new Map<string, TreeNode>(
    observations.map((obs) => [obs.id, { obs, children: [], depth: 0 }]),
  )
  const roots: TreeNode[] = []
  for (const node of byId.values()) {
    const parent = node.obs.parentObservationId
      ? byId.get(node.obs.parentObservationId)
      : undefined
    if (parent) parent.children.push(node)
    else roots.push(node)
  }
  const byStart = (a: TreeNode, b: TreeNode) => a.obs.startMs - b.obs.startMs
  const finish = (nodes: TreeNode[], depth: number) => {
    nodes.sort(byStart)
    for (const node of nodes) {
      node.depth = depth
      finish(node.children, depth + 1)
    }
  }
  finish(roots, 0)
  return roots
}

// Depth-first, honoring `collapsed`, with the connector bookkeeping the tree
// and the indented log both draw from.
export function flattenTreeOrder(
  roots: TreeNode[],
  collapsed: ReadonlySet<string> = new Set(),
): FlatItem[] {
  const out: FlatItem[] = []
  const walk = (nodes: TreeNode[], treeLines: boolean[]) => {
    nodes.forEach((node, index) => {
      const isLast = index === nodes.length - 1
      out.push({ node, treeLines, isLast })
      if (!collapsed.has(node.obs.id)) {
        walk(node.children, [...treeLines, !isLast])
      }
    })
  }
  walk(roots, [])
  return out
}

// Own cost plus every descendant's, bottom-up, so a parent's bar label says
// what everything under it cost.
export function rolledUpCosts(roots: TreeNode[]): Map<string, number> {
  const totals = new Map<string, number>()
  const walk = (node: TreeNode): number => {
    const own = node.obs.cost ?? 0
    const total = node.children.reduce((sum, child) => sum + walk(child), own)
    totals.set(node.obs.id, total)
    return total
  }
  for (const root of roots) walk(root)
  return totals
}
