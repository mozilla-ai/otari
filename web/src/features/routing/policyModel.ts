/**
 * What a routing policy is, shared by the page and by the form.
 *
 * The page reads these to summarize a row and to seed an edit; the form reads
 * them to build a draft. Split out of `RoutingPage.tsx` so neither imports the
 * other: a component module reaching into a page module is the dependency this
 * exists to keep pointing one way.
 */

import type { AliasResponse, PolicySpec, RoutingPolicyResponse } from "@/client"

/** A row on this page: either a routing policy or a stored/config alias.
 *
 *  An alias is the one-target case of a policy, so the two are listed together
 *  and this page is the single place either is managed. They still live in
 *  different tables behind different endpoints, so `kind` decides which API a
 *  write goes to; it is not cosmetic.
 */
export type RoutingRow = RoutingPolicyResponse & { kind: "policy" | "alias" }

/** The router backends the form can write. Any other is shown read-only rather
 *  than rewritten as one of these on save. */
export const KNN_BACKEND = "knn"
export const WEIGHTED_BACKEND = "weighted"
export const PRIORITY_BACKEND = "priority"

/** Server-side cap on a compiled plan (`MAX_CANDIDATES` in models/routing.py). */
export const MAX_CANDIDATES = 5

/** The fallthrough target of a spec, which every valid spec has exactly one of. */
export function findFallthroughTarget(spec: PolicySpec): string {
  return spec.select.find((entry) => entry.default !== undefined)?.default ?? ""
}

/** The router's candidate pool, or an empty list for a policy with no router. */
export function findCandidates(spec: PolicySpec): string[] {
  return (
    spec.select.find((entry) => entry.router !== undefined)?.candidates ?? []
  )
}

/** The pool the form edits: the router's candidates, with the default target in it.
 *
 *  The gateway appends the default target to the pool when a policy omits it, so a
 *  spec written through the API can list it or not. Normalizing here means the form
 *  shows the models that will actually be dispatched, in the order they were
 *  written, rather than a pool that is missing its own fallback.
 */
export function buildInitialPool(spec: PolicySpec): string[] {
  const candidates = findCandidates(spec)
  if (candidates.length === 0) return []
  const fallthrough = findFallthroughTarget(spec)
  return candidates.includes(fallthrough)
    ? candidates
    : [...candidates, fallthrough]
}

/** A backend name as the server reads it.
 *
 *  The resolver matches on `name.strip().lower()`, so `" KNN "` selects the learned
 *  router. Comparing the raw string here would show a policy the gateway routes
 *  perfectly well as an unrecognized backend, read-only and mislabelled.
 */
export function normalizeBackend(name: string | undefined): string | undefined {
  return name?.trim().toLowerCase()
}

/** The router backend a policy names, or undefined for a policy with no router. */
export function findRouterBackend(spec: PolicySpec): string | undefined {
  return normalizeBackend(
    spec.select.find((entry) => entry.router !== undefined)?.router,
  )
}

/** The declared traffic split, empty unless the policy is weighted. */
export function findWeights(spec: PolicySpec): Record<string, number> {
  return spec.select.find((entry) => entry.router !== undefined)?.weights ?? {}
}

/** Each candidate's percentage of the traffic, normalized like the server does.
 *
 *  Weights are relative, so the form shows what the operator actually gets: 7 and 3
 *  read as 70% and 30%. A candidate with no weight takes none of the traffic and
 *  stays in the plan as a failover target, which is how a provider is drained.
 */
export function computeShares(weights: number[]): number[] {
  const total = weights.reduce((sum, weight) => sum + Math.max(0, weight), 0)
  if (total <= 0) return weights.map(() => 0)
  return weights.map((weight) => (Math.max(0, weight) * 100) / total)
}

/** Which entry of `initialPool` serves when the router declines. */
export function findFallthroughIndex(spec: PolicySpec): number {
  const index = buildInitialPool(spec).indexOf(findFallthroughTarget(spec))
  return index === -1 ? 0 : index
}

/** The conditional entries, i.e. everything that is not the fallthrough. */
export function findBudgetConditions(
  spec: PolicySpec,
): { threshold: number; target: string }[] {
  return spec.select
    .filter(
      (entry) =>
        entry.when?.budget_used_pct?.gte !== undefined &&
        entry.target !== undefined,
    )
    .map((entry) => ({
      threshold: entry.when!.budget_used_pct!.gte!,
      target: entry.target!,
    }))
}

/**
 * Why a policy name cannot be sent as a `model`, or "" when it can.
 *
 * The delimiters are the server's rule: a name with `:` or `/` would shadow a
 * real `provider:model` selector. Whitespace is refused here because a caller
 * has to send the exact string, and the server would store it.
 */
export function policyNameProblem(name: string): string {
  const trimmed = name.trim()
  if (/[:/]/.test(trimmed)) return "A policy name cannot contain “:” or “/”."
  if (/\s/.test(trimmed)) return "A policy name cannot contain spaces."
  return ""
}

/** The account of a save that wrote some of its scopes and not the others.
 *
 *  Both halves by name, because "it failed" over a part-written save leaves the
 *  operator to work out which rows exist by reading the table. The ones that
 *  landed are remembered, so the retry the last sentence promises is real
 *  rather than a second pass over everything.
 */
export function describePartialScopeSave(
  written: string[],
  failed: { userId: string; reason: string }[],
  labelFor: (userId: string) => string,
): string {
  const created =
    written.length > 0
      ? `Created for ${written.map(labelFor).join(", ")}. `
      : ""
  const missing = failed
    .map((entry) => `${labelFor(entry.userId)} (${entry.reason})`)
    .join(", ")
  return `${created}Not created for ${missing}. Submitting again retries only the ones still missing.`
}

/** Present an alias as the one-target policy it is. */
export function aliasAsRow(alias: AliasResponse): RoutingRow {
  return {
    kind: "alias",
    name: alias.name,
    spec: { select: [{ default: alias.target }] },
    source: alias.source,
    user_id: alias.user_id,
    // Carried, not dropped: an edit or a delete goes back to the workspace the
    // row lives in, not the selected one.
    workspace_id: alias.workspace_id,
    is_dynamic: false,
    created_at: alias.created_at,
    updated_at: alias.updated_at,
  }
}

/**
 * A row's identity. Scope is half of it: the same name can exist globally and
 * per user, and keying on the name alone would collapse those rows into one.
 */
export const rowKeyOf = (row: RoutingRow): string =>
  JSON.stringify([row.kind, row.user_id, row.name])

/** Whether the dashboard can edit a spec without losing part of it.
 *
 *  An edit rebuilds the spec from its sections, so anything they do not model
 *  (a `user_id`/`key_id` condition, a comparator other than `gte`, an unknown
 *  router) would be dropped on save. Those are shown read-only: refusing to
 *  edit is recoverable, a silent lossy save is not.
 */
export function isEditableInForm(spec: PolicySpec): boolean {
  // Sections re-emit `select` as tier-downs, then the router, then the
  // default. Selection is order-sensitive server-side, so a router ahead of a
  // condition would come back meaning something else.
  const routerIndex = spec.select.findIndex(
    (entry) => entry.router !== undefined,
  )
  const lastConditionIndex = spec.select.reduce(
    (last, entry, index) => (entry.when !== undefined ? index : last),
    -1,
  )
  if (
    routerIndex !== -1 &&
    lastConditionIndex !== -1 &&
    routerIndex < lastConditionIndex
  )
    return false
  return spec.select.every((entry) => {
    if (entry.default !== undefined) return entry.when === undefined
    if (entry.router !== undefined) {
      if ((entry.candidates?.length ?? 0) === 0) return false
      const backend = normalizeBackend(entry.router)
      if (backend === KNN_BACKEND || backend === PRIORITY_BACKEND) return true
      // A weighted entry without weights cannot be saved back (the API
      // refuses it), so the form would have to invent a split.
      return (
        backend === WEIGHTED_BACKEND &&
        Object.keys(entry.weights ?? {}).length > 0
      )
    }
    const when = entry.when
    if (when === undefined || entry.target === undefined) return false
    const keys = Object.keys(when)
    return (
      keys.length === 1 &&
      keys[0] === "budget_used_pct" &&
      when.budget_used_pct?.gte !== undefined
    )
  })
}
