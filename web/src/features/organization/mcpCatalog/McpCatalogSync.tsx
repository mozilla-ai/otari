import type { McpCatalogSyncStatus } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Dot } from "@/design-system/indicators/Dot"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import {
  useMcpCatalogSync,
  useRequestMcpCatalogSync,
} from "@/shared/api/mcpCatalog"
import { docsSourceHref } from "@/shared/helpers/docs"
import { formatDateTimeMinutes, formatNumber } from "@/shared/helpers/format"

// The sync that fills the catalog from the MCP Registry. A worker runs it, so
// this only reads its status and asks for one.

const MODE_COPY: Record<McpCatalogSyncStatus["mode"], string> = {
  off: "Off. This deployment never contacts the MCP Registry (mcp_catalog_sync is off).",
  manual: "Manual. A sync runs only when an operator starts one here.",
  scheduled: "Scheduled. A sync also runs on its own.",
}

function describeSync(status: McpCatalogSyncStatus): {
  dot: string
  text: string
} {
  if (status.running) {
    return {
      dot: "bg-info",
      text: `Syncing. ${formatNumber(status.pages)} pages read, ${formatNumber(status.kept)} servers kept so far.`,
    }
  }
  if (status.requested) {
    return { dot: "bg-info", text: "A sync starts within a minute." }
  }
  if (status.outcome === "complete") {
    return {
      dot: "bg-success",
      text: `Complete on ${formatDateTimeMinutes(status.finished_at)}: ${formatNumber(status.kept)} servers kept of ${formatNumber(status.fetched)} read, ${formatNumber(status.removed)} removed.`,
    }
  }
  if (status.outcome === "partial") {
    return {
      dot: "bg-warning",
      text: `Partial on ${formatDateTimeMinutes(status.finished_at)}: ${status.error ?? "it stopped early"}. What it read was kept and nothing was removed.`,
    }
  }
  if (status.outcome === "failed") {
    return {
      dot: "bg-danger",
      text: `Failed on ${formatDateTimeMinutes(status.finished_at)}: ${status.error ?? "no page could be read"}. The catalog was left as it was.`,
    }
  }
  return { dot: "bg-text-subtle", text: "No sync has run yet." }
}

/** The registry sync's status and its Sync now, for an operator. The caller decides who that is. */
export function McpCatalogSync() {
  const sync = useMcpCatalogSync(true)
  const request = useRequestMcpCatalogSync()
  const status = sync.data
  const line = status ? describeSync(status) : undefined
  const isBusy = status ? status.running || status.requested : false

  return (
    <SettingsGroup
      title="Registry sync"
      description="The catalog is a copy of the official MCP Registry, kept only for remote servers Otari can connect to. Operators only."
      docsHref={docsSourceHref("mcp-catalog.md")}
      isBounded
      action={
        <Button
          variant="primary"
          className="min-h-11 md:min-h-9"
          isDisabled={!status || status.mode === "off" || isBusy}
          isPending={request.isPending}
          onPress={() => request.mutate()}
        >
          Sync now
        </Button>
      }
    >
      <div className="flex flex-col gap-2 py-3">
        <ErrorBanner error={sync.error ?? request.error} />
        {status && line ? (
          <>
            <p className="flex items-start gap-2 text-sm">
              <Dot className={`mt-1.5 ${line.dot}`} />
              <span>{line.text}</span>
            </p>
            <p className="text-caption">
              {`${formatNumber(status.server_count)} servers in the catalog. ${MODE_COPY[status.mode]}`}
              {status.next_sync_at
                ? ` Next sync ${formatDateTimeMinutes(status.next_sync_at)}.`
                : ""}
            </p>
          </>
        ) : null}
      </div>
    </SettingsGroup>
  )
}
