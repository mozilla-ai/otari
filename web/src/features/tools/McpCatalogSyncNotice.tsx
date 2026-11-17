import type { McpCatalogSyncStatus } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import {
  isMcpCatalogSyncing,
  useRequestMcpCatalogSync,
} from "@/shared/api/mcpCatalog"
import { formatDateTimeMinutes } from "@/shared/helpers/format"

// What the Browse catalog dialog says about the catalog itself, and the one
// control that fills or refreshes it. Anyone who may add servers to the
// workspace may ask; the gateway holds them to one sync an hour, so no number
// of clicks reads the registry more often than that.

/** The sentence under "Refresh" or "Load catalog" saying when it may be asked again. */
function waitLine(status: McpCatalogSyncStatus): string {
  return status.next_request_at
    ? `It can be loaded again after ${formatDateTimeMinutes(status.next_request_at)}.`
    : ""
}

export function McpCatalogSyncNotice({
  status,
  workspaceId,
}: {
  status: McpCatalogSyncStatus
  /** The workspace the request is made for, which is what lets an admin ask. */
  workspaceId: string
}) {
  const request = useRequestMcpCatalogSync()
  const isEmpty = status.server_count === 0
  const canAsk = status.next_request_at === null

  if (status.mode === "off") {
    return isEmpty ? (
      <InfoBanner>
        The catalog is turned off on this deployment. You can still add a server
        by its URL.
      </InfoBanner>
    ) : null
  }

  if (isMcpCatalogSyncing(status)) {
    return (
      <InfoBanner>
        <span role="status">
          {isEmpty
            ? "Loading the catalog. This takes a couple of minutes, and the list fills in as it goes."
            : "Refreshing the catalog. This takes a couple of minutes."}
        </span>
      </InfoBanner>
    )
  }

  const ask = (
    <Button
      variant={isEmpty ? "primary" : "ghost"}
      isPending={request.isPending}
      isDisabled={!canAsk}
      onPress={() => request.mutate(workspaceId)}
    >
      {isEmpty ? "Load catalog" : "Refresh"}
    </Button>
  )

  if (isEmpty) {
    return (
      <div className="flex flex-col items-start gap-2">
        <p className="text-sm">
          The catalog has not been loaded yet. Loading copies the list of remote
          servers from the MCP Registry.
        </p>
        {status.outcome === "failed" && status.error ? (
          <InfoBanner tone="warning">
            The last attempt failed: {status.error}.
          </InfoBanner>
        ) : null}
        <ErrorBanner error={request.error} />
        {ask}
        {canAsk ? null : <p className="text-caption">{waitLine(status)}</p>}
      </div>
    )
  }

  if (!status.is_stale) return null
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-caption">
          {status.last_completed_at
            ? `Last loaded ${formatDateTimeMinutes(status.last_completed_at)}.`
            : "Not fully loaded yet."}
          {canAsk ? "" : ` ${waitLine(status)}`}
        </p>
        {ask}
      </div>
      <ErrorBanner error={request.error} />
    </div>
  )
}
