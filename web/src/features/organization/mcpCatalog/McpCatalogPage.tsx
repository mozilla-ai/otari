import { useState } from "react"
import { FiRefreshCw, FiTrash2 } from "react-icons/fi"

import type { McpCatalogFeatured, McpCatalogSyncStatus } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Chip } from "@/design-system/indicators/Chip"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { canManage } from "@/features/organization/roles"
import { McpCatalogBrowser } from "@/features/tools/McpCatalogBrowser"
import {
  useFeaturedMcpServers,
  useFeatureMcpServer,
  useMcpCatalogSync,
  useRequestMcpCatalogSync,
  useUnfeatureMcpServer,
} from "@/shared/api/mcpCatalog"
import {
  useDeploymentOperator,
  useOrganizationContext,
} from "@/shared/api/organizations"
import { docsSourceHref } from "@/shared/helpers/docs"
import { formatDateTimeMinutes, formatNumber } from "@/shared/helpers/format"

// The organization's featured shelf, and for an operator the sync that fills
// the catalog under it. Two audiences on one page because they are one
// subject: what the Browse catalog list offers a workspace, and in what order.

const MODE_COPY: Record<McpCatalogSyncStatus["mode"], string> = {
  off: "Off. This deployment never contacts the MCP Registry (mcp_catalog_sync is off).",
  manual: "Manual. A sync runs only when an operator starts one here.",
  scheduled: "Scheduled. A sync also runs on its own.",
}

function outcomeLine(status: McpCatalogSyncStatus): {
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
      text: `Partial on ${formatDateTimeMinutes(status.finished_at)}: ${status.error ?? "it stopped early"}. The ${formatNumber(status.pages)} pages it read were kept, nothing was removed, and the next sync resumes there.`,
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

function SyncGroup() {
  const sync = useMcpCatalogSync(true)
  const request = useRequestMcpCatalogSync()
  const status = sync.data
  const line = status ? outcomeLine(status) : undefined
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

const STATUS_CHIP: Record<
  McpCatalogFeatured["catalog_status"],
  { tone: "neutral" | "warning"; label: string; detail: string } | undefined
> = {
  current: undefined,
  changed: {
    tone: "warning",
    label: "URL changed in catalog",
    detail:
      "The catalog now lists another URL. This entry keeps the one you featured until you update it.",
  },
  removed: {
    tone: "neutral",
    label: "No longer in catalog",
    detail: "The catalog no longer lists this server. This entry is unchanged.",
  },
}

export function McpCatalogPage() {
  const context = useOrganizationContext()
  const { isOperator } = useDeploymentOperator()
  const canEdit = canManage(context.data)
  const featured = useFeaturedMcpServers()
  const feature = useFeatureMcpServer()
  const unfeature = useUnfeatureMcpServer()
  const [isAdding, setAdding] = useState(false)

  const shelf = featured.data?.data ?? []
  const onShelf = new Set(shelf.map((entry) => entry.name))

  const columns: DataTableColumn<McpCatalogFeatured>[] = [
    {
      id: "title",
      header: "Server",
      isRowHeader: true,
      cell: (row) => {
        const chip = STATUS_CHIP[row.catalog_status]
        return (
          <span className="flex flex-col gap-1">
            <span className="flex flex-wrap items-center gap-2">
              <span className="font-medium">{row.title}</span>
              {row.requires_token ? (
                <Chip tone="info">Needs a token</Chip>
              ) : null}
              {chip ? (
                <span title={chip.detail}>
                  <Chip tone={chip.tone}>{chip.label}</Chip>
                </span>
              ) : null}
            </span>
            <span className="font-mono text-caption">{row.name}</span>
          </span>
        )
      },
    },
    {
      id: "url",
      header: "URL",
      cell: (row) => <code className="break-all text-xs">{row.url}</code>,
    },
    {
      id: "created_at",
      header: "Featured",
      cell: (row) => (
        <span className="whitespace-nowrap">
          {formatDateTimeMinutes(row.created_at)}
        </span>
      ),
    },
  ]
  if (canEdit) {
    columns.push({
      id: "actions",
      header: "",
      cell: (row) => (
        <RowActionRow>
          {row.catalog_status === "changed" && row.catalog_server_id ? (
            <RowAction
              icon={FiRefreshCw}
              label="Update"
              ariaLabel={`Update ${row.title} to the catalog's URL`}
              isDisabled={feature.isPending}
              onPress={() => feature.mutate(row.catalog_server_id as string)}
            />
          ) : null}
          <RowAction
            icon={FiTrash2}
            label="Remove"
            ariaLabel={`Remove ${row.title}`}
            isDisabled={unfeature.isPending && unfeature.variables === row.id}
            onPress={() => unfeature.mutate(row.id)}
          />
        </RowActionRow>
      ),
    })
  }

  return (
    <div className="flex flex-col gap-10 pb-10">
      <PageIntro
        title="MCP catalog"
        docsHref={docsSourceHref("mcp-catalog.md")}
      >
        The MCP servers your organization recommends. They come first when
        someone adds a server from the catalog on Tools, MCP servers. Featuring
        a server permits nothing: a workspace uses a server only once it is
        added there.
      </PageIntro>

      <ErrorBanner error={context.error ?? featured.error} />

      <SettingsGroup
        title="Featured servers"
        count={featured.data ? shelf.length : undefined}
        description="Each entry keeps the URL it had when you featured it. A sync never changes or removes it."
        action={
          canEdit ? (
            <Button
              variant="primary"
              className="min-h-11 md:min-h-9"
              onPress={() => {
                feature.reset()
                setAdding(true)
              }}
            >
              Feature a server
            </Button>
          ) : null
        }
      >
        <ErrorBanner
          error={unfeature.error ?? (isAdding ? null : feature.error)}
        />
        <DataTable
          ariaLabel="Featured servers"
          columns={columns}
          rows={shelf}
          getRowKey={(row) => row.id}
          isLoading={featured.isPending && !featured.data}
          emptyContent="No server is featured yet."
          mobileLayout="cards"
        />
        {!canEdit && context.data ? (
          <p className="pt-3 text-caption">
            Only an owner or admin of the organization can change this list.
          </p>
        ) : null}
      </SettingsGroup>

      {isOperator ? <SyncGroup /> : null}

      <Dialog
        isOpen={isAdding}
        onOpenChange={setAdding}
        size="lg"
        title="Feature a server"
        description="Pick the servers to recommend. Each is copied as the catalog lists it now."
        actions={
          <Button variant="primary" onPress={() => setAdding(false)}>
            Done
          </Button>
        }
      >
        <DialogSection>
          <ErrorBanner error={feature.error} />
          <McpCatalogBrowser
            featured={[]}
            emptyCatalog={
              isOperator
                ? "The catalog is empty. Run Sync now on this page to fill it from the MCP Registry."
                : "The catalog is empty. This deployment's operator can fill it from the MCP Registry."
            }
            renderControl={(pick) => {
              const serverId = pick.serverId
              return onShelf.has(pick.name) || !serverId ? (
                <Chip tone="success">Featured</Chip>
              ) : (
                <Button
                  className="min-h-11 md:min-h-9"
                  aria-label={`Feature ${pick.title}`}
                  isDisabled={feature.isPending}
                  onPress={() => feature.mutate(serverId)}
                >
                  Feature
                </Button>
              )
            }}
          />
        </DialogSection>
      </Dialog>

      {!canEdit && !isOperator && context.data ? (
        <InfoBanner>
          The catalog itself is synced by this deployment&rsquo;s operator.
        </InfoBanner>
      ) : null}
    </div>
  )
}
