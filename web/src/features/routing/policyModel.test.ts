import { describe, expect, it } from "vitest"

import type { PolicySpec } from "@/client"
import {
  candidatesOf,
  defaultTargetOf,
  initialPool,
  normalizedBackend,
  routerBackendOf,
  sharesOf,
  weightsOf,
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
    expect(defaultTargetOf(weighted(["a", "b"], {}, "b"))).toBe("b")
  })

  it("answers with an empty string for a spec that has none", () => {
    // Not a valid policy, and the form still has to render something rather
    // than passing undefined into a combo box.
    expect(defaultTargetOf({ select: [] } as unknown as PolicySpec)).toBe("")
  })
})

describe("candidatesOf", () => {
  it("reads the router's pool", () => {
    expect(candidatesOf(weighted(["a", "b"], {}))).toEqual(["a", "b"])
  })

  it("is empty for a policy with no router", () => {
    expect(
      candidatesOf({ select: [{ default: "a" }] } as unknown as PolicySpec),
    ).toEqual([])
  })
})

describe("initialPool", () => {
  it("leaves a pool that already lists its fallthrough alone, in order", () => {
    expect(initialPool(weighted(["a", "b"], {}, "b"))).toEqual(["a", "b"])
  })

  it("appends a fallthrough the pool omits, which is what the gateway does", () => {
    // A spec written through the API can leave the default out of the pool; the
    // server appends it before dispatching, so the form has to show it or it
    // shows a plan that is not the one that runs.
    expect(initialPool(weighted(["a", "b"], {}, "c"))).toEqual(["a", "b", "c"])
  })

  it("stays empty for a policy with no router, rather than becoming the default alone", () => {
    expect(
      initialPool({ select: [{ default: "a" }] } as unknown as PolicySpec),
    ).toEqual([])
  })
})

describe("normalizedBackend", () => {
  it("matches the server's own strip-and-lower", () => {
    expect(normalizedBackend(" KNN ")).toBe("knn")
  })

  it("passes undefined through", () => {
    expect(normalizedBackend(undefined)).toBeUndefined()
  })
})

describe("routerBackendOf", () => {
  it("normalizes what the spec declared", () => {
    const spec = {
      select: [{ router: " Weighted ", candidates: ["a"] }, { default: "a" }],
    } as unknown as PolicySpec
    expect(routerBackendOf(spec)).toBe("weighted")
  })

  it("is undefined for a policy with no router", () => {
    expect(
      routerBackendOf({ select: [{ default: "a" }] } as unknown as PolicySpec),
    ).toBeUndefined()
  })
})

describe("weightsOf", () => {
  it("is empty unless the policy is weighted", () => {
    expect(
      weightsOf({ select: [{ default: "a" }] } as unknown as PolicySpec),
    ).toEqual({})
  })
})

describe("sharesOf", () => {
  it("reads relative weights as percentages", () => {
    expect(sharesOf([7, 3])).toEqual([70, 30])
  })

  it("gives every candidate nothing when the weights total zero", () => {
    // The drained case, and the one that would divide by zero: a policy whose
    // weights are all zero routes nothing through the router, and the form has
    // to say 0% rather than NaN.
    expect(sharesOf([0, 0])).toEqual([0, 0])
  })

  it("treats a negative weight as none of the traffic", () => {
    expect(sharesOf([-5, 5])).toEqual([0, 100])
  })

  it("keeps an unweighted candidate at zero, which is how a provider is drained", () => {
    expect(sharesOf([10, 0])).toEqual([100, 0])
  })
})
