import { Button } from "@/design-system/actions/Button"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { UNCONFIRMED_SETTINGS } from "@/features/tools/mcpWorkspaceView"

/**
 * Says a workspace's settings on its organization's server could not be read
 * again after a failed write, so what is shown may not be what is saved, with
 * the one way out: reading them again. Shown wherever a change could start.
 */
export function McpUnconfirmedNotice({
  onRefresh,
  isRefreshing,
}: {
  onRefresh: () => void
  isRefreshing: boolean
}) {
  return (
    <InfoBanner>
      <span className="flex flex-wrap items-center gap-3">
        {UNCONFIRMED_SETTINGS}
        <Button
          onPress={onRefresh}
          isPending={isRefreshing}
          className="min-h-11 md:min-h-0"
        >
          Refresh
        </Button>
      </span>
    </InfoBanner>
  )
}
