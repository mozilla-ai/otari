/**
 * A policy being edited, as the create dialog and the policy page hold it.
 *
 * Pure, so the round trip from a stored spec to a draft and back is unit
 * tested on its own: an edit that saves a section has to write every other
 * section back exactly as it found it.
 */

import type { PolicyGuardrail, PolicySpec } from "@/client"

import {
  buildInitialPool,
  computeShares,
  findBudgetConditions,
  findFallthroughIndex,
  findFallthroughTarget,
  findRouterBackend,
  findWeights,
  KNN_BACKEND,
  MAX_CANDIDATES,
  PRIORITY_BACKEND,
  WEIGHTED_BACKEND,
} from "./policyModel"

export type RouterBackend =
  | typeof KNN_BACKEND
  | typeof WEIGHTED_BACKEND
  | typeof PRIORITY_BACKEND

export interface TierDown {
  threshold: number
  target: string
}

/**
 * The routed pool: the models one of which serves each request.
 *
 * `weights` runs parallel to `models` whatever the backend, as the text the
 * operator typed, so a share follows its model and a half-typed decimal
 * survives a render. Only the weighted backend writes it.
 */
export interface PoolDraft {
  backend: RouterBackend
  models: string[]
  weights: string[]
  /** Which model serves when the router declines or a caller opts out. */
  safeIndex: number
}

export interface PolicyDraft {
  /** The main model, while no pool is set. */
  target: string
  /** Routed when it holds any model. */
  pool: PoolDraft
  backups: string[]
  tierDowns: TierDown[]
  guardrails: PolicyGuardrail[]
}

/** What a new policy is for, picked before its form opens. */
export type Intent = "fallback" | "weighted" | "knn" | "priority" | "tierdown"

/** How the main selection is made, on the policy page. */
export type Selection = "single" | RouterBackend

export const INTENTS: readonly {
  value: Intent
  label: string
  description: string
}[] = [
  {
    value: "fallback",
    label: "Fallback",
    description: "One main model, with backups tried in order when it fails.",
  },
  {
    value: "weighted",
    label: "Split traffic",
    description: "A weighted mix across models, such as 70% and 30%.",
  },
  {
    value: "knn",
    label: "Smart router",
    description:
      "Learns from scored examples to pick a cheap model or a strong one.",
  },
  {
    value: "priority",
    label: "Spill over at rate limits",
    description:
      "Models in priority order; the next takes over when one is full.",
  },
  {
    value: "tierdown",
    label: "Tier down on budget",
    description: "Switch to a cheaper model once the budget fills up.",
  },
]

export const SELECTIONS: readonly { value: Selection; label: string }[] = [
  { value: "single", label: "One model" },
  { value: WEIGHTED_BACKEND, label: "Split traffic" },
  { value: KNN_BACKEND, label: "Smart router" },
  { value: PRIORITY_BACKEND, label: "Spill over" },
]

export function emptyPool(backend: RouterBackend = KNN_BACKEND): PoolDraft {
  return { backend, models: [], weights: [], safeIndex: 0 }
}

/** A new pool of two, the first being the model the policy already names. */
function seedPool(backend: RouterBackend, first: string): PoolDraft {
  return {
    backend,
    models: [first, ""],
    // An even split, which the operator then skews; for the other backends the
    // weights are carried and never written.
    weights: backend === WEIGHTED_BACKEND ? ["50", "50"] : ["0", "0"],
    safeIndex: 0,
  }
}

/** A backend name the form can write, or undefined for one it cannot. */
function writableBackend(name: string | undefined): RouterBackend | undefined {
  return name === KNN_BACKEND ||
    name === WEIGHTED_BACKEND ||
    name === PRIORITY_BACKEND
    ? name
    : undefined
}

export function draftFromSpec(spec: PolicySpec): PolicyDraft {
  const models = buildInitialPool(spec)
  const declared = findWeights(spec)
  return {
    target: models.length > 0 ? "" : findFallthroughTarget(spec),
    pool:
      models.length === 0
        ? emptyPool()
        : {
            backend: writableBackend(findRouterBackend(spec)) ?? KNN_BACKEND,
            models,
            weights: models.map((selector) => String(declared[selector] ?? 0)),
            safeIndex: findFallthroughIndex(spec),
          },
    backups: spec.on_failure ?? [],
    tierDowns: findBudgetConditions(spec),
    guardrails: spec.guardrails ?? [],
  }
}

/** The draft a new policy of this kind starts from. */
export function seedDraft(intent: Intent, target = ""): PolicyDraft {
  const base: PolicyDraft = {
    target,
    pool: emptyPool(),
    backups: [],
    tierDowns: [],
    guardrails: [],
  }
  switch (intent) {
    case "fallback":
      // A fallback is its backups, so the form opens on the first one.
      return { ...base, backups: [""] }
    case "tierdown":
      return { ...base, tierDowns: [{ threshold: 80, target: "" }] }
    default:
      return { ...base, target: "", pool: seedPool(intent, target) }
  }
}

export function isRouted(draft: PolicyDraft): boolean {
  return draft.pool.models.length > 0
}

export function selectionOf(draft: PolicyDraft): Selection {
  return isRouted(draft) ? draft.pool.backend : "single"
}

/** The model a request falls through to: the marked one in a pool. */
export function fallthroughOf(draft: PolicyDraft): string {
  return isRouted(draft)
    ? (draft.pool.models[draft.pool.safeIndex] ?? "")
    : draft.target
}

/**
 * Change how the main model is chosen, keeping what carries over.
 *
 * Into a pool, the current main model is its first entry. Out of one, the
 * marked model becomes the main model. Between backends the pool stays, and a
 * split with no share above zero is given an even one.
 */
export function switchSelection(
  draft: PolicyDraft,
  next: Selection,
): PolicyDraft {
  if (next === selectionOf(draft)) return draft
  if (next === "single")
    return { ...draft, target: fallthroughOf(draft), pool: emptyPool() }
  if (!isRouted(draft))
    return { ...draft, target: "", pool: seedPool(next, draft.target.trim()) }
  const hasSplit = parseWeights(draft.pool.weights).some((value) => value > 0)
  return {
    ...draft,
    pool: {
      ...draft.pool,
      backend: next,
      weights:
        next === WEIGHTED_BACKEND && !hasSplit
          ? draft.pool.models.map(() => "50")
          : draft.pool.weights,
    },
  }
}

/** Each share as a number; a cleared field is NaN, so it reads as unfinished. */
export function parseWeights(weights: string[]): number[] {
  return weights.map((text) => (text.trim() === "" ? Number.NaN : Number(text)))
}

/** Each model's percentage of the split, for the line under its share. */
export function poolShares(pool: PoolDraft): number[] {
  return computeShares(
    parseWeights(pool.weights).map((value) =>
      Number.isFinite(value) ? Math.max(0, value) : 0,
    ),
  )
}

export interface PoolProblems {
  /** Fewer than two models, which is not a routing decision. */
  isTooSmall: boolean
  isMissingModel: boolean
  hasDuplicate: boolean
  /** A share that is not a finite number of zero or more. */
  hasBadShare: boolean
  /** Every share zero, so the split could never send traffic. */
  isSplitEmpty: boolean
}

export function poolProblems(pool: PoolDraft): PoolProblems {
  const named = pool.models
    .map((entry) => entry.trim())
    .filter((entry) => entry !== "")
  const values = parseWeights(pool.weights)
  const weighted = pool.backend === WEIGHTED_BACKEND
  const hasBadShare =
    weighted && !values.every((value) => Number.isFinite(value) && value >= 0)
  return {
    isTooSmall: pool.models.length < 2,
    isMissingModel: named.length !== pool.models.length,
    hasDuplicate: new Set(named).size !== named.length,
    hasBadShare,
    isSplitEmpty:
      weighted && !hasBadShare && !values.some((value) => value > 0),
  }
}

/** How many models the compiled plan dispatches: the pool, or one, plus the backups. */
export function plannedModels(draft: PolicyDraft): number {
  return (draft.pool.models.length || 1) + draft.backups.length
}

export function isAtModelCap(draft: PolicyDraft): boolean {
  return plannedModels(draft) >= MAX_CANDIDATES
}

/** Whether the draft can be saved as it stands. */
export function isDraftReady(draft: PolicyDraft): boolean {
  const problems = poolProblems(draft.pool)
  const poolReady = !isRouted(draft) || !Object.values(problems).some(Boolean)
  return (
    poolReady &&
    fallthroughOf(draft).trim() !== "" &&
    draft.backups.every((entry) => entry.trim() !== "") &&
    draft.tierDowns.every(
      (tier) =>
        tier.target.trim() !== "" && tier.threshold > 0 && tier.threshold < 100,
    ) &&
    draft.guardrails.every((guardrail) => guardrail.profile.trim() !== "") &&
    plannedModels(draft) <= MAX_CANDIDATES
  )
}

/**
 * What is wrong with each backup, or "" for one that is fine: a backup that
 * repeats the main model or an earlier backup is never tried, since the plan
 * holds each model once.
 */
export function backupProblems(backups: string[], main: string): string[] {
  const named = backups.map((entry) => entry.trim())
  return named.map((entry, index) => {
    if (entry === "") return ""
    if (entry === main.trim())
      return "Same as the main model. Pick another one."
    const first = named.indexOf(entry)
    return first < index ? `Same as backup ${first + 1}. Pick another one.` : ""
  })
}

/** Whether a new fallback has its backups: at least one, each named and distinct. */
export function isFallbackReady(draft: PolicyDraft): boolean {
  return (
    draft.backups.length > 0 &&
    draft.backups.every((entry) => entry.trim() !== "") &&
    backupProblems(draft.backups, draft.target).every((entry) => entry === "")
  )
}

/**
 * The spec a draft saves as, in plan order: the tier-downs, then the router,
 * then the fallthrough, which the schema requires to be last.
 */
export function buildSpec(draft: PolicyDraft): PolicySpec {
  const { pool } = draft
  const values = parseWeights(pool.weights)
  return {
    select: [
      ...draft.tierDowns.map((tier) => ({
        when: { budget_used_pct: { gte: tier.threshold } },
        target: tier.target.trim(),
      })),
      // After the tier-downs: an explicit tier-down overrides the router.
      ...(isRouted(draft)
        ? [
            {
              router: pool.backend,
              candidates: pool.models.map((entry) => entry.trim()),
              // Only for the weighted backend: the API refuses a weight map on
              // any other, where it would read as a split and do nothing.
              ...(pool.backend === WEIGHTED_BACKEND
                ? {
                    weights: Object.fromEntries(
                      pool.models.map((entry, index) => {
                        const value = values[index] ?? 0
                        return [
                          entry.trim(),
                          Number.isFinite(value) ? Math.max(0, value) : 0,
                        ]
                      }),
                    ),
                  }
                : {}),
            },
          ]
        : []),
      { default: fallthroughOf(draft).trim() },
    ],
    ...(draft.backups.length > 0
      ? { on_failure: draft.backups.map((entry) => entry.trim()) }
      : {}),
    ...(draft.guardrails.length > 0 ? { guardrails: draft.guardrails } : {}),
  }
}
