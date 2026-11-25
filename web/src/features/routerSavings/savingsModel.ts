import type {
  SavingsAmounts,
  SubagentSavingsPoint,
  SubagentSavingsTotals,
} from "@/client"
import type { StackedPoint } from "@/design-system/metrics/charts"
import { formatNumber, formatSpend } from "@/shared/helpers/format"

/** How the money in a group of decisions was arrived at. */
export type Basis =
  | "Measured"
  | "Estimated"
  | "Mixed"
  | "Not priced"
  | "Not reported"

export function basisOf(amounts: SavingsAmounts): Basis {
  if (amounts.measured > 0 && amounts.estimated > 0) return "Mixed"
  if (amounts.measured > 0) return "Measured"
  if (amounts.estimated > 0) return "Estimated"
  // No subagent said what it consumed, which is not the same as a report
  // that could not be priced.
  if (amounts.with_outcome === 0) return "Not reported"
  return "Not priced"
}

/** The decisions whose outcome is priced on both sides: the ones the money counts. */
export const pricedCount = (amounts: SavingsAmounts) =>
  amounts.measured + amounts.estimated

/** What was paid: the subagents' own tokens and the recommendations. */
export const spentOn = (amounts: SavingsAmounts) =>
  amounts.actual_cost + amounts.router_cost

/** A saving with its sign, since a loss must read as one. */
export function formatSaved(value: number): string {
  const amount = formatSpend(Math.abs(value))
  return value < 0 ? `-${amount}` : amount
}

/** Which decisions the money counts and which it leaves out, as one sentence. */
export function coverageLine(totals: SubagentSavingsTotals): string {
  return [
    totals.measured > 0 ? `${formatNumber(totals.measured)} measured` : "",
    totals.estimated > 0 ? `${formatNumber(totals.estimated)} estimated` : "",
    totals.no_outcome > 0
      ? `${formatNumber(totals.no_outcome)} not reported`
      : "",
    totals.unpriced > 0 ? `${formatNumber(totals.unpriced)} not priced` : "",
  ]
    .filter(Boolean)
    .join(", ")
}

export const SPENT_KEY = "spent"
export const SAVED_KEY = "saved"

/**
 * The series as stacked bars: what was paid, and the saving on top of it, so a
 * bar's height is what the subagents would have cost. A bucket that lost money
 * has no saving to stack and shows what was paid alone.
 */
export function chartPoints(series: SubagentSavingsPoint[]): StackedPoint[] {
  return series.map((point) => ({
    x: point.bucket_start,
    [SPENT_KEY]: spentOn(point),
    [SAVED_KEY]: Math.max(0, point.net_saved),
  }))
}
