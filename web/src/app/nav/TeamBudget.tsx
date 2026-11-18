import { Link } from "@tanstack/react-router"
import { SpendMeter } from "@/design-system/metrics/SpendMeter"
import { usedShare, wholePct } from "@/features/spend/limits"
import { useOrganizationLimit } from "@/features/spend/useSpend"
import { formatUsd, formatUsdHeadline } from "@/shared/helpers/format"

/**
 * The organization's limit at the rail's foot: how much of it is spent, on
 * every page, for the people who can change it. Read as Home reads it, so the
 * two agree. Nothing until a limit exists, and nothing for anyone the server
 * would refuse the read. It opens Home, where the limit is changed and the
 * spend behind it is broken down.
 */
export function TeamBudget({ onNavigate }: { onNavigate?: () => void }) {
  const team = useOrganizationLimit()
  if (!team) return null
  return (
    <Link
      to="/"
      onClick={onNavigate}
      className="-mx-3 -mb-4 flex shrink-0 flex-col gap-1.5 border-t border-border px-6 py-3 transition-colors hover:bg-surface-alt"
    >
      <span className="text-overline">Team budget</span>
      <span className="flex items-baseline justify-between gap-2 text-shell-label tabular-nums">
        <span>
          {formatUsd(team.spent)}{" "}
          <span className="text-muted">of {formatUsdHeadline(team.max)}</span>
        </span>
        <span className="text-caption">{wholePct(usedShare(team))}</span>
      </span>
      <SpendMeter
        spent={team.spent}
        allocated={team.max}
        ariaLabel="Team budget spent"
      />
    </Link>
  )
}
