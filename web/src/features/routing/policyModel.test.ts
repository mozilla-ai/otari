import { describe, expect, it } from "vitest"

import type { PolicySpec } from "@/client"
import {
  buildInitialPool,
  computeShares,
  describePartialScopeSave,
  findBudgetConditions,
  findCandidates,
  findFallthroughIndex,
  findFallthroughTarget,
  findRouterBackend,
  findWeights,
  normalizeBackend,
} from "./policyModel"

/** A spec in the shape the form writes: router entry first, fallthrough last. */
function weighted(
  candidates: string[],
  weights: Record<string, number>,
  fallthrough = candidates[0],
): PolicySpec {
  return {
    select: [
      { router: "weighted", candidates, weights },
      { default: fallthrough },
    ],
  } as PolicySpec
}

describe("defaultTargetOf", () => {
  it("finds the fallthrough entry", () => {
    expect(findFallthroughTarget(weighted(["a", "b"], {}, "b"))).toBe("b")
  })

  it("answers with an empty string for a spec that has none", () => {
    // Not a valid policy, and the form still has to render something rather
    // than passing undefined into a combo box.
    expect(findFallthroughTarget({ select: [] } as unknown as PolicySpec)).toBe(
      "",
    )
  })
})

describe("candidatesOf", () => {
  it("reads the router's pool", () => {
    expect(findCandidates(weighted(["a", "b"], {}))).toEqual(["a", "b"])
  })

  it("is empty for a policy with no router", () => {
    expect(
      findCandidates({ select: [{ default: "a" }] } as unknown as PolicySpec),
    ).toEqual([])
  })
})

describe("initialPool", () => {
  it("leaves a pool that already lists its fallthrough alone, in order", () => {
    expect(buildInitialPool(weighted(["a", "b"], {}, "b"))).toEqual(["a", "b"])
  })

  it("appends a fallthrough the pool omits, which is what the gateway does", () => {
    // A spec written through the API can leave the default out of the pool; the
    // server appends it before dispatching, so the form has to show it or it
    // shows a plan that is not the one that runs.
    expect(buildInitialPool(weighted(["a", "b"], {}, "c"))).toEqual([
      "a",
      "b",
      "c",
    ])
  })

  it("stays empty for a policy with no router, rather than becoming the default alone", () => {
    expect(
      buildInitialPool({ select: [{ default: "a" }] } as unknown as PolicySpec),
    ).toEqual([])
  })
})

describe("normalizedBackend", () => {
  it("matches the server's own strip-and-lower", () => {
    expect(normalizeBackend(" KNN ")).toBe("knn")
  })

  it("passes undefined through", () => {
    expect(normalizeBackend(undefined)).toBeUndefined()
  })
})

describe("routerBackendOf", () => {
  it("normalizes what the spec declared", () => {
    const spec = {
      select: [{ router: " Weighted ", candidates: ["a"] }, { default: "a" }],
    } as unknown as PolicySpec
    expect(findRouterBackend(spec)).toBe("weighted")
  })

  it("is undefined for a policy with no router", () => {
    expect(
      findRouterBackend({
        select: [{ default: "a" }],
      } as unknown as PolicySpec),
    ).toBeUndefined()
  })
})

describe("weightsOf", () => {
  it("is empty unless the policy is weighted", () => {
    expect(
      findWeights({ select: [{ default: "a" }] } as unknown as PolicySpec),
    ).toEqual({})
  })
})

describe("sharesOf", () => {
  it("reads relative weights as percentages", () => {
    expect(computeShares([7, 3])).toEqual([70, 30])
  })

  it("gives every candidate nothing when the weights total zero", () => {
    // The drained case, and the one that would divide by zero: a policy whose
    // weights are all zero routes nothing through the router, and the form has
    // to say 0% rather than NaN.
    expect(computeShares([0, 0])).toEqual([0, 0])
  })

  it("treats a negative weight as none of the traffic", () => {
    expect(computeShares([-5, 5])).toEqual([0, 100])
  })

  it("keeps an unweighted candidate at zero, which is how a provider is drained", () => {
    expect(computeShares([10, 0])).toEqual([100, 0])
  })
})

describe("initialSafeIndex", () => {
  it("points at the fallthrough's place in the pool", () => {
    expect(findFallthroughIndex(weighted(["a", "b", "c"], {}, "c"))).toBe(2)
  })

  it("points at the first entry for a policy with no pool to point into", () => {
    // `initialPool` appends a fallthrough the spec omits, so the only way to
    // reach the fallback is a policy with no router at all: the pool is empty
    // and there is no place to point. The control still has to select
    // something, so it selects the first.
    const spec = { select: [{ default: "main" }] } as unknown as PolicySpec
    expect(buildInitialPool(spec)).toEqual([])
    expect(findFallthroughIndex(spec)).toBe(0)
  })
})

describe("conditionsOf", () => {
  it("reads the conditional entries and leaves the fallthrough out", () => {
    const spec = {
      select: [
        { when: { budget_used_pct: { gte: 80 } }, target: "cheap" },
        { when: { budget_used_pct: { gte: 95 } }, target: "cheapest" },
        { default: "main" },
      ],
    } as unknown as PolicySpec
    expect(findBudgetConditions(spec)).toEqual([
      { threshold: 80, target: "cheap" },
      { threshold: 95, target: "cheapest" },
    ])
  })

  it("skips an entry missing either half of the pair", () => {
    // A threshold with nothing to route to, and a target on no condition, are
    // both unfinished rather than conditional. Neither is a row the form can
    // draw, so neither becomes one.
    const spec = {
      select: [
        { when: { budget_used_pct: { gte: 80 } } },
        { target: "orphan" },
        { default: "main" },
      ],
    } as unknown as PolicySpec
    expect(findBudgetConditions(spec)).toEqual([])
  })
})

describe("partialScopeReport", () => {
  const labelFor = (userId: string) => `user-${userId}`

  it("names both halves, so the operator knows which rows exist", () => {
    expect(
      describePartialScopeSave(
        ["1"],
        [{ userId: "2", reason: "already exists" }],
        labelFor,
      ),
    ).toBe(
      "Created for user-1. Not created for user-2 (already exists). Submitting again retries only the ones still missing.",
    )
  })

  it("omits the created half when nothing landed", () => {
    expect(
      describePartialScopeSave([], [{ userId: "2", reason: "boom" }], labelFor),
    ).toBe(
      "Not created for user-2 (boom). Submitting again retries only the ones still missing.",
    )
  })

  it("lists every name on each side", () => {
    expect(
      describePartialScopeSave(
        ["1", "2"],
        [
          { userId: "3", reason: "a" },
          { userId: "4", reason: "b" },
        ],
        labelFor,
      ),
    ).toBe(
      "Created for user-1, user-2. Not created for user-3 (a), user-4 (b). Submitting again retries only the ones still missing.",
    )
  })
})
