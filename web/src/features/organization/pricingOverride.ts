/**
 * The client half of the rate-override rules the gateway enforces.
 *
 * Every check here has a server-side counterpart
 * (`services/organization_pricing_service.py`); these exist so a write that
 * would be refused is disabled with a reason rather than sent and rejected.
 * None of them is the authority: the server refuses either way.
 */

import type { OrganizationPricingOverride } from "@/client"

/** The rate fields an override carries, in the order the form shows them. */
export const RATE_FIELDS = [
  "input_price_per_million",
  "output_price_per_million",
  "cache_read_price_per_million",
  "cache_write_price_per_million",
  "cache_write_1h_price_per_million",
] as const

export type RateField = (typeof RATE_FIELDS)[number]

/**
 * Parse a rate the operator typed.
 *
 * Three outcomes, deliberately distinct: `undefined` for a blank field (an
 * optional rate left unset, which is not the same as zero), `NaN` for something
 * that is not a non-negative number, and the number otherwise. A blank required
 * field is the caller's to reject.
 */
export function parseRate(value: string): number | undefined {
  const trimmed = value.trim()
  if (trimmed === "") return undefined
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : Number.NaN
}

/**
 * Whether a model key could ever be read back.
 *
 * A pricing row is only resolved under a `prefix:model` selector, so a key with
 * no provider or instance prefix would store a rate nothing bills against. Same
 * rule, and the same accepted legacy slash form, as the deployment price dialog
 * (`isValidModelKey` in `features/models/SetPriceDialog.tsx`); kept separate
 * rather than imported because `features/` may not reach across to another
 * feature's dialog for a regex.
 */
export function isValidModelKey(value: string): boolean {
  return /^[^\s:/]+[:/][^\s]+$/.test(value.trim())
}

/**
 * Why a period would be refused, or undefined if it would be accepted.
 *
 * The period is half-open, `[from, to)`, so an end equal to the start covers no
 * instant at all and is refused along with an inverted one. That is the server's
 * rule (`validate_period`), stated here as a sentence.
 *
 * Each field is judged on its own, and that matters more than it looks. Blank is
 * legitimate on both sides (a start defaults to now, an absent end is open
 * ended), but a value that is present and *unparseable* is not the same thing:
 * `fromLocalInput` turns it into `null`, which the server reads as "starts now"
 * or "no end". So an early return that only looked at blankness would let a
 * half-typed date enable Save, store a period the operator did not ask for, and
 * skip the overlap check on the way (`Date.parse` yields `NaN`, and every
 * comparison against `NaN` is false, so nothing looks like it overlaps).
 *
 * A blank start is compared against *now* rather than skipped, because that is
 * what the dialog submits for it: leaving the start blank and setting an end in
 * the past would otherwise keep Save enabled and earn a 400, which is the one
 * outcome this module exists to prevent. `now` is injectable so a test can pin it.
 */
export function periodBlockedReason(
  effectiveFrom: string,
  effectiveTo: string,
  now = Date.now(),
): string | undefined {
  const from = effectiveFrom.trim()
  const to = effectiveTo.trim()
  if (from !== "" && Number.isNaN(Date.parse(from))) {
    return "The start is not a date this browser recognizes."
  }
  if (to !== "" && Number.isNaN(Date.parse(to))) {
    return "The end is not a date this browser recognizes."
  }
  // No end is open ended, which nothing can conflict with.
  if (to === "") return undefined
  const start = from === "" ? now : Date.parse(from)
  if (Date.parse(to) <= start) {
    return from === ""
      ? "The end is in the past, and a blank start means now. Set a start, or an end in the future."
      : "The end must be after the start. An end equal to the start covers no time at all."
  }
  return undefined
}

/**
 * Whether two periods cover a common instant.
 *
 * The client-side twin of the server's overlap refusal, used to disable the save
 * control before a doomed request rather than to decide the outcome. Half-open
 * on both sides, so two periods that merely touch do not overlap. A null end is
 * open ended, which overlaps everything after its start.
 */
export function periodsOverlap(
  first: { from: number; to: number | undefined },
  second: { from: number; to: number | undefined },
): boolean {
  const firstEndsAfter = first.to === undefined || first.to > second.from
  const secondEndsAfter = second.to === undefined || second.to > first.from
  return firstEndsAfter && secondEndsAfter
}

/** An override's period as epoch millis, for comparison. */
export function periodOf(override: OrganizationPricingOverride): {
  from: number
  to: number | undefined
} {
  return {
    from: Date.parse(override.effective_from),
    to:
      override.effective_to === null || override.effective_to === undefined
        ? undefined
        : Date.parse(override.effective_to),
  }
}

/**
 * The override in `existing` that a candidate period would collide with.
 *
 * Scoped to one model key, because that is the scope of the server's rule: two
 * models may share a period freely.
 */
export function findOverlapping(
  existing: readonly OrganizationPricingOverride[],
  candidate: {
    modelKey: string
    from: number
    to: number | undefined
    excludeId?: string
  },
): OrganizationPricingOverride | undefined {
  return existing.find(
    (override) =>
      override.model_key === candidate.modelKey &&
      override.id !== candidate.excludeId &&
      periodsOverlap(periodOf(override), {
        from: candidate.from,
        to: candidate.to,
      }),
  )
}

/**
 * Whether an override applies right now.
 *
 * What the table's status column reads. A period that has not started is
 * "Scheduled" and one that has ended is "Expired"; both are kept rather than
 * hidden, because an operator setting next quarter's rate needs to see it, and
 * one debugging a bill needs to see what applied last month.
 */
export function overrideStatus(
  override: OrganizationPricingOverride,
  now = Date.now(),
): "active" | "scheduled" | "expired" {
  const { from, to } = periodOf(override)
  if (Number.isFinite(from) && from > now) return "scheduled"
  if (to !== undefined && Number.isFinite(to) && to <= now) return "expired"
  return "active"
}
