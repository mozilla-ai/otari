import type { WorkspaceMcpServerHealth } from "@/client"
import { Dot } from "@/design-system/indicators/Dot"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import {
  HEALTH_STATE,
  healthState,
  healthStateDetail,
} from "@/features/tools/mcpServerHealth"
import { formatDateTime } from "@/shared/helpers/format"

/**
 * What the gateway last heard from a server, as one dot beside its name: green
 * healthy, amber worth a look, red failing, gray never reached. A manager's
 * check and every request that uses the server both update it. The dot carries the same words
 * as its tooltip for a screen reader, and the server's own page says them in
 * full, so the tooltip is never the only place they are said.
 */
export function McpServerStatusDot({
  name,
  health,
}: {
  name: string
  health: WorkspaceMcpServerHealth | null
}) {
  const state = healthState(health)
  const { label, dot } = HEALTH_STATE[state]
  const detail = healthStateDetail(health, state)
  const when = health
    ? `Last contact ${formatDateTime(health.checked_at)}.`
    : ""
  return (
    <Tooltip
      content={
        <span className="flex max-w-xs flex-col gap-1">
          <span className="font-medium">{label}</span>
          <span>{detail}</span>
          {when ? <span className="text-caption">{when}</span> : null}
        </span>
      }
    >
      <span
        role="img"
        aria-label={`${name}: ${label}. ${detail} ${when}`.trim()}
        className="flex size-4 shrink-0 items-center justify-center"
      >
        <Dot className={dot} />
      </span>
    </Tooltip>
  )
}
