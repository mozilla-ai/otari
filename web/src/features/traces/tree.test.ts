import { describe, expect, it } from "vitest"

import {
  buildTree,
  flattenTreeOrder,
  rolledUpCosts,
  type TreeNode,
} from "@/features/traces/tree"
import type { Observation } from "@/features/traces/types"

function observation(overrides: Partial<Observation> = {}): Observation {
  return {
    id: "obs",
    parentObservationId: null,
    type: "SPAN",
    name: "span",
    startMs: 0,
    endMs: 10,
    level: "DEFAULT",
    ...overrides,
  }
}

// Two roots, listed out of order:
//   a ─┬─ a1 ── a1x
//      └─ a2
//   b ─── b1
const forest = () => [
  observation({ id: "b1", parentObservationId: "b", startMs: 60, cost: 0.25 }),
  observation({ id: "a2", parentObservationId: "a", startMs: 30 }),
  observation({ id: "b", startMs: 50, cost: 0.5 }),
  observation({ id: "a1x", parentObservationId: "a1", startMs: 20, cost: 4 }),
  observation({ id: "a1", parentObservationId: "a", startMs: 10, cost: 2 }),
  observation({ id: "a", startMs: 0, cost: 1 }),
]

const ids = (nodes: TreeNode[]) => nodes.map((node) => node.obs.id)

describe("buildTree", () => {
  it("nests each span under its parent, in start order at every level", () => {
    const roots = buildTree(forest())

    expect(ids(roots)).toEqual(["a", "b"])
    const [a, b] = roots
    expect(ids(a.children)).toEqual(["a1", "a2"])
    expect(ids(a.children[0].children)).toEqual(["a1x"])
    expect(ids(b.children)).toEqual(["b1"])
  })

  it("records each span's depth", () => {
    const [a] = buildTree(forest())

    expect(a.depth).toBe(0)
    expect(a.children[0].depth).toBe(1)
    expect(a.children[0].children[0].depth).toBe(2)
  })

  it("keeps a span whose parent is missing as a root", () => {
    const roots = buildTree([
      observation({ id: "orphan", parentObservationId: "gone", startMs: 5 }),
      observation({ id: "root", startMs: 0 }),
    ])

    expect(ids(roots)).toEqual(["root", "orphan"])
  })
})

describe("flattenTreeOrder", () => {
  it("walks the tree depth first", () => {
    const flat = flattenTreeOrder(buildTree(forest()))

    expect(flat.map((item) => item.node.obs.id)).toEqual([
      "a",
      "a1",
      "a1x",
      "a2",
      "b",
      "b1",
    ])
  })

  it("draws a connector past a row for every ancestor with a sibling below it", () => {
    const flat = flattenTreeOrder(buildTree(forest()))
    const row = (id: string) => flat.find((item) => item.node.obs.id === id)

    expect(row("a")).toMatchObject({ treeLines: [], isLast: false })
    expect(row("a1")).toMatchObject({ treeLines: [true], isLast: false })
    expect(row("a1x")).toMatchObject({ treeLines: [true, true], isLast: true })
    expect(row("a2")).toMatchObject({ treeLines: [true], isLast: true })
    expect(row("b")).toMatchObject({ treeLines: [], isLast: true })
    expect(row("b1")).toMatchObject({ treeLines: [false], isLast: true })
  })

  it("leaves out everything under a collapsed span but keeps the span", () => {
    const flat = flattenTreeOrder(buildTree(forest()), new Set(["a"]))

    expect(flat.map((item) => item.node.obs.id)).toEqual(["a", "b", "b1"])
  })

  it("collapses only the span named, not its siblings", () => {
    const flat = flattenTreeOrder(buildTree(forest()), new Set(["a1"]))

    expect(flat.map((item) => item.node.obs.id)).toEqual([
      "a",
      "a1",
      "a2",
      "b",
      "b1",
    ])
  })
})

describe("rolledUpCosts", () => {
  it("gives every span its own cost plus everything under it", () => {
    const costs = rolledUpCosts(buildTree(forest()))

    expect(Object.fromEntries(costs)).toEqual({
      a: 7,
      a1: 6,
      a1x: 4,
      a2: 0,
      b: 0.75,
      b1: 0.25,
    })
  })
})
