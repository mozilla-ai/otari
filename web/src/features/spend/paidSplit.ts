import type { UsageTotals } from "@/client"
import { formatUsd } from "@/shared/helpers/format"

/**
 * The spend split by who paid the provider, or "" where hosted providers served
 * nothing, so a deployment without them reads as it always has.
 */
export function paidSplitLine(
  totals: Pick<UsageTotals, "cost" | "deployment_paid_cost"> | undefined,
): string {
  const hosted = totals?.deployment_paid_cost ?? 0
  if (!totals || hosted <= 0) return ""
  const own = Math.max(0, totals.cost - hosted)
  return `${formatUsd(hosted)} hosted · ${formatUsd(own)} own keys`
}
