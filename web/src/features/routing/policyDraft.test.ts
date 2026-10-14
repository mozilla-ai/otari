import { describe, expect, it } from "vitest"

import type { PolicySpec } from "@/client"

import {
  buildSpec,
  draftFromSpec,
  fallthroughOf,
  isDraftReady,
  plannedModels,
  poolProblems,
  seedDraft,
  selectionOf,
  switchSelection,
} from "./policyDraft"

const COMBINED: PolicySpec = {
  select: [
    { when: { budget_used_pct: { gte: 80 } }, target: "openai:gpt-5-nano" },
    {
      router: "weighted",
      candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
      weights: { "openai:gpt-5": 7.5, "anthropic:claude-sonnet-4-5": 0 },
    },
    { default: "openai:gpt-5" },
  ],
  on_failure: ["anthropic:claude-haiku-4-5"],
  guardrails: [{ profile: "pii", mode: "block", on_unavailable: "monitor" }],
}

describe("draftFromSpec and buildSpec", () => {
  it("round-trip a policy that combines every section", () => {
    expect(buildSpec(draftFromSpec(COMBINED))).toEqual(COMBINED)
  })

  it("round-trip a plain fallback", () => {
    const spec: PolicySpec = {
      select: [{ default: "openai:gpt-5" }],
      on_failure: ["anthropic:claude-haiku-4-5", "openai:gpt-5-mini"],
    }
    expect(buildSpec(draftFromSpec(spec))).toEqual(spec)
  })

  it("normalize a loosely spelled backend and add a default the pool omits", () => {
    const draft = draftFromSpec({
      select: [
        { router: " KNN ", candidates: ["openai:gpt-5-nano"] },
        { default: "openai:gpt-5" },
      ],
    })
    expect(buildSpec(draft).select).toEqual([
      { router: "knn", candidates: ["openai:gpt-5-nano", "openai:gpt-5"] },
      { default: "openai:gpt-5" },
    ])
  })

  it("write weights only for a split", () => {
    const draft = draftFromSpec(COMBINED)
    const priority = {
      ...draft,
      pool: { ...draft.pool, backend: "priority" as const },
    }
    expect(buildSpec(priority).select[1]).toEqual({
      router: "priority",
      candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
    })
  })
})

describe("seedDraft", () => {
  it("starts a fallback from the deep-linked main model", () => {
    const draft = seedDraft("fallback", "openai:gpt-4o")
    expect(selectionOf(draft)).toBe("single")
    expect(fallthroughOf(draft)).toBe("openai:gpt-4o")
  })

  it("starts a split with that model first and an even share", () => {
    const draft = seedDraft("weighted", "openai:gpt-4o")
    expect(draft.pool.models).toEqual(["openai:gpt-4o", ""])
    expect(draft.pool.weights).toEqual(["50", "50"])
    expect(draft.target).toBe("")
  })

  it("starts a tier-down at 80% with its cheaper model still to pick", () => {
    expect(seedDraft("tierdown").tierDowns).toEqual([
      { threshold: 80, target: "" },
    ])
  })
})

describe("switchSelection", () => {
  it("moves the main model into a new pool, and the marked one back out", () => {
    const single = draftFromSpec({ select: [{ default: "openai:gpt-5" }] })
    const routed = switchSelection(single, "knn")
    expect(routed.pool.models).toEqual(["openai:gpt-5", ""])
    const marked = {
      ...routed,
      pool: { ...routed.pool, models: ["a:x", "b:y"], safeIndex: 1 },
    }
    expect(switchSelection(marked, "single")).toMatchObject({
      target: "b:y",
      pool: { models: [] },
    })
  })

  it("gives a split an even share when no model has one", () => {
    const learned = draftFromSpec({
      select: [
        { router: "knn", candidates: ["a:x", "b:y"] },
        { default: "b:y" },
      ],
    })
    expect(switchSelection(learned, "weighted").pool.weights).toEqual([
      "50",
      "50",
    ])
  })
})

describe("readiness", () => {
  it("refuses a pool of one, a repeated model and an empty split", () => {
    const pool = {
      backend: "weighted" as const,
      models: ["a:x", "a:x"],
      weights: ["0", "0"],
      safeIndex: 0,
    }
    expect(poolProblems(pool)).toMatchObject({
      hasDuplicate: true,
      isSplitEmpty: true,
    })
    expect(
      poolProblems({ ...pool, models: ["a:x"], weights: ["1"] }),
    ).toMatchObject({ isTooSmall: true })
  })

  it("counts the pool and the backups against the cap", () => {
    const draft = draftFromSpec(COMBINED)
    expect(plannedModels(draft)).toBe(3)
    expect(isDraftReady(draft)).toBe(true)
    expect(
      isDraftReady({ ...draft, backups: ["c:1", "c:2", "c:3", "c:4"] }),
    ).toBe(false)
  })

  it("refuses a tier-down threshold the budget gate makes unreachable", () => {
    const draft = draftFromSpec(COMBINED)
    expect(
      isDraftReady({
        ...draft,
        tierDowns: [{ threshold: 100, target: "openai:gpt-5-nano" }],
      }),
    ).toBe(false)
  })
})
