/**
 * What a routing policy is, shared by the page and by the form.
 *
 * The page reads these to summarize a row and to seed an edit; the form reads
 * them to build a draft. Split out of `RoutingPage.tsx` so neither imports the
 * other: a component module reaching into a page module is the dependency this
 * exists to keep pointing one way.
 */

import type { PolicySpec, RoutingPolicyResponse } from "@/client"

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

/** Server-side cap on a compiled plan (`MAX_CANDIDATES` in models/routing.py). */
export const MAX_CANDIDATES = 5

/** The fallthrough target of a spec, which every valid spec has exactly one of. */
export function defaultTargetOf(spec: PolicySpec): string {
  return spec.select.find((entry) => entry.default !== undefined)?.default ?? ""
}

/** The router's candidate pool, or an empty list for a policy with no router. */
export function candidatesOf(spec: PolicySpec): string[] {
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
export function initialPool(spec: PolicySpec): string[] {
  const candidates = candidatesOf(spec)
  if (candidates.length === 0) return []
  const fallthrough = defaultTargetOf(spec)
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
export function normalizedBackend(
  name: string | undefined,
): string | undefined {
  return name?.trim().toLowerCase()
}

/** The router backend a policy names, or undefined for a policy with no router. */
export function routerBackendOf(spec: PolicySpec): string | undefined {
  return normalizedBackend(
    spec.select.find((entry) => entry.router !== undefined)?.router,
  )
}

/** The declared traffic split, empty unless the policy is weighted. */
export function weightsOf(spec: PolicySpec): Record<string, number> {
  return spec.select.find((entry) => entry.router !== undefined)?.weights ?? {}
}

/** Each candidate's percentage of the traffic, normalized like the server does.
 *
 *  Weights are relative, so the form shows what the operator actually gets: 7 and 3
 *  read as 70% and 30%. A candidate with no weight takes none of the traffic and
 *  stays in the plan as a failover target, which is how a provider is drained.
 */
export function sharesOf(weights: number[]): number[] {
  const total = weights.reduce((sum, weight) => sum + Math.max(0, weight), 0)
  if (total <= 0) return weights.map(() => 0)
  return weights.map((weight) => (Math.max(0, weight) * 100) / total)
}
