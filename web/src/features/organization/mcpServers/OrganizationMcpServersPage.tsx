import { useNavigate } from "@tanstack/react-router"
import { useState } from "react"
import { FiActivity, FiChevronRight } from "react-icons/fi"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { canManage } from "@/features/organization/roles"
import { McpConnectionNoticeBanner } from "@/features/tools/McpConnectionNoticeBanner"
import { McpModeLabel } from "@/features/tools/McpModeLabel"
import { McpNotice } from "@/features/tools/McpNotice"
import {
  McpServerDialog,
  type McpServerDraft,
} from "@/features/tools/McpServerDialog"
import { McpServerLogo } from "@/features/tools/McpServerLogo"
import { McpServerStatusDot } from "@/features/tools/McpServerStatusDot"
import { mcpServersPagePath } from "@/features/tools/mcpServerPage"
import { useMcpServerCheck } from "@/features/tools/useMcpServerCheck"
import {
  connectionsLabel,
  SharedAccountCell,
  toolsLabel,
} from "@/features/tools/WorkspaceMcpServersCard"
import { ORGANIZATION_MCP_SCOPE } from "@/shared/api/mcpServerScope"
import {
  useCreateOrganizationMcpServer,
  useOrganizationMcpServers,
} from "@/shared/api/organizationMcpServers"
import { useOrganizationContext } from "@/shared/api/organizations"

/** How many members connected their own account to a Personal server. */
function PersonalConnectionsCell({ server }: { server: WorkspaceMcpServer }) {
  const count = server.connected_users
  if (count === null || count === undefined) {
    return <span className="text-subtle">…</span>
  }
  const label = connectionsLabel(count, false)
  const isNobody = count === 0
  return (
    <span
      className={`flex items-center gap-2 whitespace-nowrap ${isNobody ? "text-muted" : ""}`}
    >
      <Dot className={isNobody ? "bg-text-subtle" : "bg-success"} />
      {label}
    </span>
  )
}

/**
 * The organization's own MCP servers, which every workspace of it reaches. An
 * owner or admin adds one here and sets, on its page, how far each workspace
 * may go with it; the workspace's own admins can only narrow that.
 */
export function OrganizationMcpServersPage() {
  const context = useOrganizationContext()
  // The server answers these to an owner or admin only, so a member's read is
  // not sent.
  const canEdit = canManage(context.data)
  const query = useOrganizationMcpServers(canEdit)
  const create = useCreateOrganizationMcpServer()
  const check = useMcpServerCheck({
    scope: ORGANIZATION_MCP_SCOPE,
    returnTo: mcpServersPagePath(ORGANIZATION_MCP_SCOPE),
  })
  const navigate = useNavigate()

  const [isDialogOpen, setDialogOpen] = useState(false)
  // Bumped on every open and used as the dialog's key, so each open starts
  // from a blank form and the fields survive the closing animation.
  const [openCount, setOpenCount] = useState(0)

  const rows = query.data?.data ?? []

  const openAdd = () => {
    create.reset()
    setOpenCount((count) => count + 1)
    setDialogOpen(true)
  }

  const openServer = (serverId: string) =>
    void navigate({
      to: "/organization/mcp-servers/$serverId",
      params: { serverId },
    })

  // A new server opens on its own page, where its check, its account and how
  // each workspace reaches it live.
  const submit = (draft: McpServerDraft) =>
    create.mutate(draft, { onSuccess: (server) => openServer(server.id) })

  const columns: DataTableColumn<WorkspaceMcpServer>[] = [
    {
      id: "name",
      header: "Name",
      isRowHeader: true,
      cell: (row) => (
        <span className="flex min-w-0 items-center gap-3">
          <McpServerLogo name={row.name} logoUrl={null} size="sm" />
          <span className="font-medium">{row.name}</span>
          <McpServerStatusDot name={row.name} health={row.health} />
        </span>
      ),
    },
    {
      id: "mode",
      header: "Sign-in",
      cell: (row) => <McpModeLabel mode={row.credential_mode} isLong />,
    },
    {
      id: "connections",
      header: "Connections",
      cell: (row) =>
        row.credential_mode === "per_user" ? (
          <PersonalConnectionsCell server={row} />
        ) : (
          <SharedAccountCell server={row} />
        ),
    },
    {
      id: "url",
      header: "URL",
      cell: (row) => <code className="break-all text-xs">{row.url}</code>,
    },
    {
      id: "allowed_tools",
      header: "Tools",
      cell: (row) => (
        <span className="whitespace-nowrap">{toolsLabel(row)}</span>
      ),
    },
    {
      id: "actions",
      header: "",
      cell: (row) => (
        <RowActionRow>
          <RowAction
            icon={FiActivity}
            label={
              row.credential_mode === "per_user"
                ? "Check this server (opens a sign-in, connects nothing)"
                : "Check this server"
            }
            ariaLabel={`Check ${row.name}`}
            isDisabled={check.isRunning(row.id)}
            onPress={() => check.run(row)}
          />
          <RowAction
            icon={FiChevronRight}
            label="Manage"
            ariaLabel={`Manage ${row.name}`}
            onPress={() => openServer(row.id)}
          />
        </RowActionRow>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-10 pb-10">
      <PageIntro title="MCP servers" />

      <McpConnectionNoticeBanner />
      <ErrorBanner error={context.error ?? query.error} />
      {check.error ? (
        <McpNotice message={check.error} onDismiss={check.dismissError} />
      ) : null}

      {/* Held back until the context answers: `canEdit` is false while the role
          resolves, and the banner would flash for an admin on every load. */}
      {context.data && !canEdit ? (
        <InfoBanner>
          Only organization owners and admins can see this organization&rsquo;s
          MCP servers.
        </InfoBanner>
      ) : null}

      {canEdit || context.isPending ? (
        <SettingsGroup
          title="Organization servers"
          count={query.data ? rows.length : undefined}
          description="Servers every workspace in your organization can use. Each workspace's admins can narrow access or turn one off, never widen it."
          action={
            canEdit ? (
              <Button
                variant="primary"
                className="min-h-11 md:min-h-9"
                onPress={openAdd}
              >
                Add server
              </Button>
            ) : null
          }
        >
          <TableScrollFrame className="otari-mcp-table">
            <DataTable
              ariaLabel="Organization MCP servers"
              mobileLayout="cards"
              columns={columns}
              rows={rows}
              getRowKey={(row) => row.id}
              onRowAction={openServer}
              isLoading={context.isPending || (query.isPending && !query.data)}
              emptyContent="No organization MCP servers yet."
            />
          </TableScrollFrame>
        </SettingsGroup>
      ) : null}

      {/* Keyed on the open count, so each open starts from a blank form. */}
      <McpServerDialog
        key={openCount}
        isOpen={isDialogOpen}
        onOpenChange={setDialogOpen}
        scope={ORGANIZATION_MCP_SCOPE}
        isPending={create.isPending}
        error={create.error}
        onSubmit={submit}
      />
    </div>
  )
}
