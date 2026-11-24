import { Link } from "@tanstack/react-router"

import { formatSaved, pricedCount } from "@/features/routerSavings/savingsModel"
import {
  isPeriodKey,
  type SpendWindow,
  usageRange,
  windowWords,
} from "@/features/spend/period"
import { useSubagentSavings } from "@/shared/api/subagentSavings"
import { useSurfaces } from "@/shared/hooks/useDeployment"

/**
 * One line on Home saying what the subagent router saved over Home's own
 * window, with the way to the Sessions view that breaks it down over the same
 * period and for the same people. Draws nothing until a
 * routed subagent has a priced outcome, so a deployment that never routes one
 * never sees it.
 */
export function RouterSavingsTile({
  window,
  workspaceId,
  userId,
  isEnabled = true,
}: {
  window: SpendWindow
  /** "" reads every workspace the caller may. */
  workspaceId: string
  /** Narrows to one person's subagents. */
  userId?: string
  isEnabled?: boolean
}) {
  const hasSurface = useSurfaces()
  const savings = useSubagentSavings(
    {
      ...usageRange(window),
      bucket: window.bucket,
      workspace_id: workspaceId || undefined,
      user_id: userId,
    },
    isEnabled && hasSurface("subagent_savings"),
  )
  const totals = savings.data?.totals
  if (!totals || pricedCount(totals) === 0) return null
  const words = windowWords(window)
  const estimated = totals.estimated > 0 ? " (estimated)" : ""
  return (
    <p className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-sm text-muted">
      <span className="text-foreground">
        {totals.net_saved < 0
          ? `The subagent router cost ${formatSaved(-totals.net_saved)} more than it saved ${words}${estimated}.`
          : `The subagent router saved ${formatSaved(totals.net_saved)} ${words}${estimated}.`}
      </span>
      <Link
        to="/router-savings"
        search={{
          ...(isPeriodKey(window.kind) ? { period: window.kind } : {}),
          ...(userId ? { who: "me" } : {}),
        }}
        className="text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
      >
        See router savings
      </Link>
    </p>
  )
}
