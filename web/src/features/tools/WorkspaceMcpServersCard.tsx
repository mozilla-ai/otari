import { useNavigate } from "@tanstack/react-router"
import { useState } from "react"
import { FiActivity, FiChevronRight, FiTrash2 } from "react-icons/fi"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { canManageWorkspace } from "@/features/organization/roles"
import { McpCatalogDialog } from "@/features/tools/McpCatalogDialog"
import { McpConnectionNoticeBanner } from "@/features/tools/McpConnectionNoticeBanner"
import { McpMemberServers } from "@/features/tools/McpMemberServers"
import { McpModeLabel } from "@/features/tools/McpModeLabel"
import { McpNotice } from "@/features/tools/McpNotice"
import {
  McpServerDialog,
  type McpServerDraft,
} from "@/features/tools/McpServerDialog"
import { McpServerLogo } from "@/features/tools/McpServerLogo"
import { McpServerRequestsBlock } from "@/features/tools/McpServerRequestsBlock"
import { McpServerStatusDot } from "@/features/tools/McpServerStatusDot"
import { sharedAccountState } from "@/features/tools/McpSharedAccount"
import { AVAILABILITY, availabilityOf } from "@/features/tools/mcpAvailability"
import type { CatalogPick } from "@/features/tools/mcpCatalog"
import { hasToolListing } from "@/features/tools/mcpServerHealth"
import { useMcpServerCheck } from "@/features/tools/useMcpServerCheck"
import { WorkspaceRequestMcpServers } from "@/features/tools/WorkspaceRequestMcpServers"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useCreateWorkspaceMcpServer,
  useDeleteWorkspaceMcpServer,
  useMyMcpConnections,
  useWorkspaceMcpServerLogos,
  useWorkspaceMcpServers,
} from "@/shared/api/tools"
import { useWorkspace } from "@/shared/api/workspaces"
import { docsSourceHref } from "@/shared/helpers/docs"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useSurfaces } from "@/shared/hooks/useDeployment"

// The MCP servers this workspace has registered, which a request reaches by
// naming their ids in `mcp_server_ids` (see docs/mcp.md). A manager gets the
// table and its controls; a member gets `McpMemberServers`, their own
// connections first.
//
// A list, where the two workspace config planes beside it on this page hold one
// row each, and that difference is the whole shape of this card: nothing here
// can say "deployment default", and there is no third stance. A server is
// registered or it is not, and `enabled` decides whether requests reach it.
// `web/AGENTS.md` says why this plane differs from the two beside it.
//
// **No client-side copy of the server's cap.** A workspace may hold only so
// many servers, and that number lives in the service
// (`MAX_MCP_SERVERS_PER_WORKSPACE`) where the create path enforces it under a
// workspace lock. Restating it here would be a second copy to go stale, so the
// 51st Add is left to earn the server's own refusal, which names the real
// number and lands in the dialog's error banner.

/** The page's own opening line, which the `/tools` card does not carry. */
const PAGE_DESCRIPTION =
  "Give this workspace's requests the tools of remote MCP servers."

const DOCS_HREF = docsSourceHref("mcp-servers.md")

/**
 * How many tools a server exposes, never their names: the allow-list's length
 * where it has one, else what its last tool listing counted, else "All".
 */
function toolsLabel(server: WorkspaceMcpServer): string {
  const count =
    server.allowed_tools?.length ??
    (hasToolListing(server.health) ? server.health.tool_count : null)
  if (count === null || count === undefined) return "All"
  return `${count} ${count === 1 ? "tool" : "tools"}`
}

/** Whether a Shared server's one account is set up, in the list's Connections column. */
function SharedAccountCell({ server }: { server: WorkspaceMcpServer }) {
  const state = sharedAccountState(server)
  if (state === "not_set_up") {
    return (
      <span className="flex items-center gap-2 whitespace-nowrap">
        <Dot className="bg-warning" />
        Not set up
      </span>
    )
  }
  if (state === "open") {
    return (
      <span className="whitespace-nowrap text-muted">No account needed</span>
    )
  }
  return (
    <span className="flex items-center gap-2 whitespace-nowrap">
      <Dot className="bg-success" />
      Set up
    </span>
  )
}

/** "No one yet", "1 member", "3 members", with "including you" when the admin looking is one of them. */
export function connectionsLabel(count: number, includesYou: boolean): string {
  // The admin's own connection is known before the list's count catches up with it.
  const total = Math.max(count, includesYou ? 1 : 0)
  if (total === 0) return "No one yet"
  const members = `${total} ${total === 1 ? "member" : "members"}`
  if (!includesYou) return members
  return total === 1 ? "You" : `${members}, including you`
}

/** How many members connected their own account to a Personal server, in the list's Connections column. */
function PersonalConnectionsCell({
  server,
  includesYou,
}: {
  server: WorkspaceMcpServer
  /** Whether the admin looking has connected; undefined until that is read. */
  includesYou: boolean | undefined
}) {
  const count = server.connected_users
  if (count === null || count === undefined) {
    // A gateway that does not count yet still knows the admin's own connection.
    if (includesYou === undefined) return <span className="text-subtle">…</span>
    return (
      <span className="whitespace-nowrap text-muted">
        {includesYou ? "You" : "Not you"}
      </span>
    )
  }
  const label = connectionsLabel(count, includesYou ?? false)
  return (
    <span
      className={`flex items-center gap-2 whitespace-nowrap ${label === "No one yet" ? "text-muted" : ""}`}
    >
      <Dot
        className={label === "No one yet" ? "bg-text-subtle" : "bg-success"}
      />
      {label}
    </span>
  )
}

export function WorkspaceMcpServersCard({
  variant = "card",
}: {
  /**
   * Where this is rendering. As a `card` it is one section at the foot of
   * `/tools`, with its own `h2` and the register control beside it. As a
   * `page` it *is* `/tools/mcp-servers`, so it renders that page's opening
   * instead and the control sits in the heading row there, which is where
   * every other page in the dashboard keeps the one thing it creates.
   */
  variant?: "card" | "page"
}) {
  const { selected, isLoading: workspaceLoading } = useSelectedWorkspace()
  const context = useOrganizationContext()
  // The client half of the *write* gate the service enforces. The list is a
  // member read now (otari-ai#1942): these servers act on every request the
  // member sends through the workspace, and the token was never in the rows,
  // so any member who can see the workspace reads the table and only an owner
  // or admin gets the affordances that change it.
  const manages = canManageWorkspace(context.data, selected?.role)
  const workspaceId = selected ? selected.workspace_id : null
  const query = useWorkspaceMcpServers(workspaceId)
  const mine = useMyMcpConnections(manages ? workspaceId : null)
  const create = useCreateWorkspaceMcpServer()
  const remove = useDeleteWorkspaceMcpServer()
  const check = useMcpServerCheck({
    workspaceId: selected?.workspace_id ?? "",
    returnTo: "/tools/mcp-servers",
  })
  const navigate = useNavigate()
  // The workspace row carries the request-named servers switch; only a manager
  // is offered it, so only a manager pays for the read.
  const workspace = useWorkspace(workspaceId, manages)

  const [isDialogOpen, setDialogOpen] = useState(false)
  // Bumped on every open and used as the dialog's key, so the draft is cleared
  // on the way in. Clearing it on close would blank the fields while the dialog
  // is still animating away.
  const [openCount, setOpenCount] = useState(0)
  const [pendingDelete, setPendingDelete] = useState<WorkspaceMcpServer>()
  // The catalog is a feature a deployment can switch off, and its dialog is
  // keyed like the form's so each open starts from nothing picked.
  const hasCatalog = useSurfaces()("mcp_catalog")
  const logos = useWorkspaceMcpServerLogos(workspaceId)
  const [isCatalogOpen, setCatalogOpen] = useState(false)
  const [catalogOpenCount, setCatalogOpenCount] = useState(0)
  // The server a member's request names, which the catalog opens its confirm step on.
  const [requestedPick, setRequestedPick] = useState<CatalogPick>()
  const openCatalog = (pick?: CatalogPick) => {
    setRequestedPick(pick)
    setCatalogOpenCount((count) => count + 1)
    setCatalogOpen(true)
  }
  const openRequested = (pick: CatalogPick) => openCatalog(pick)

  const isPage = variant === "page"

  if (!selected) {
    return (
      <section className="flex flex-col gap-2">
        {isPage ? (
          <PageIntro title="MCP servers" docsHref={DOCS_HREF}>
            {PAGE_DESCRIPTION}
          </PageIntro>
        ) : (
          <h2 className="text-title">MCP servers</h2>
        )}
        <InfoBanner>
          {workspaceLoading
            ? "Reading the workspaces you belong to."
            : "MCP servers belong to a workspace. Ask an owner or admin to add you to one."}
        </InfoBanner>
      </section>
    )
  }

  const rows = query.data?.data ?? []
  const connectedIds = new Set(
    (mine.data?.data ?? []).map((entry) => entry.server_id),
  )

  // Each opener clears the mutation it will render the error of. Without this a
  // refused write leaves its banner up, so reopening the form shows the last
  // 409 over a blank one. Same reset `PasskeysCard` does for the same reason.
  const openAdd = () => {
    create.reset()
    setOpenCount((count) => count + 1)
    setDialogOpen(true)
  }

  const openDelete = (server: WorkspaceMcpServer) => {
    remove.reset()
    setPendingDelete(server)
  }

  const openServer = (serverId: string) =>
    void navigate({
      to: "/tools/mcp-servers/$serverId",
      params: { serverId },
    })

  // A new server opens on its own page, where its connection check, its
  // access list and every later change live.
  const submit = (draft: McpServerDraft) =>
    create.mutate(
      { workspaceId: selected.workspace_id, body: draft },
      { onSuccess: (server) => openServer(server.id) },
    )

  const columns: DataTableColumn<WorkspaceMcpServer>[] = [
    {
      id: "name",
      header: "Name",
      isRowHeader: true,
      cell: (row) => (
        <span className="flex min-w-0 items-center gap-3">
          <McpServerLogo
            name={row.name}
            logoUrl={logos.get(row.id)?.logo_url}
            tone={logos.get(row.id)?.logo_tone}
            size="sm"
          />
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
      // What stands behind the sign-in: whether a shared server's one account
      // is set up, or how many members connected their own to a personal one.
      cell: (row) =>
        row.credential_mode !== "per_user" ? (
          <SharedAccountCell server={row} />
        ) : (
          <PersonalConnectionsCell
            server={row}
            includesYou={
              mine.data === undefined ? undefined : connectedIds.has(row.id)
            }
          />
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
      // A count, never the names: an allow-list can run to hundreds. The
      // server's own page lists them.
      cell: (row) => (
        <span className="whitespace-nowrap">{toolsLabel(row)}</span>
      ),
    },
    {
      id: "availability",
      header: "Availability",
      // One indicator for the one choice the server's page makes.
      cell: (row) => {
        const availability = AVAILABILITY[availabilityOf(row)]
        return (
          <span className="flex items-center gap-2 text-mono-caption whitespace-nowrap text-muted">
            <Dot className={availability.dot} />
            {availability.badge}
          </span>
        )
      },
    },
  ]
  if (manages) {
    columns.push({
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
          <RowAction
            icon={FiTrash2}
            label="Delete"
            onPress={() => openDelete(row)}
          />
        </RowActionRow>
      ),
    })
  }

  const addButton = manages ? (
    <div className="flex flex-wrap gap-2">
      {hasCatalog ? (
        <Button onPress={() => openCatalog()}>Browse catalog</Button>
      ) : null}
      <Button variant="primary" onPress={openAdd}>
        Add MCP server
      </Button>
    </div>
  ) : null

  return (
    <section className="flex flex-col gap-2">
      {isPage ? (
        <PageIntro title="MCP servers" action={addButton} docsHref={DOCS_HREF}>
          {PAGE_DESCRIPTION}
        </PageIntro>
      ) : (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-title">MCP servers</h2>
          {addButton}
        </div>
      )}
      {manages && workspace.data ? (
        // Pulled up under the page's opening line, which ends in its own padding.
        <div className={isPage ? "-mt-4 pb-2" : ""}>
          <WorkspaceRequestMcpServers workspace={workspace.data} />
        </div>
      ) : null}

      <McpConnectionNoticeBanner />
      <ErrorBanner error={query.error} />
      {check.error ? (
        <McpNotice message={check.error} onDismiss={check.dismissError} />
      ) : null}
      <ErrorBanner error={workspace.error} />

      {manages ? null : (
        <McpMemberServers
          workspaceId={selected.workspace_id}
          workspaceName={selected.name}
          servers={rows}
          logos={logos}
          isLoading={query.isPending && !query.data}
        />
      )}

      {manages && hasCatalog ? (
        <McpServerRequestsBlock
          workspaceId={selected.workspace_id}
          onAdd={openRequested}
        />
      ) : null}

      {manages ? (
        <TableScrollFrame className="otari-mcp-table">
          <DataTable
            ariaLabel={`MCP servers for ${selected.name}`}
            mobileLayout="cards"
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.id}
            onRowAction={openServer}
            isLoading={query.isPending && !query.data}
            // Deliberately asserts nothing about the workspace: an empty table
            // is also what a failed request leaves behind, and the banner above
            // is the only thing that knows which of the two happened.
            emptyContent="No MCP servers yet."
          />
        </TableScrollFrame>
      ) : null}

      {/* Keyed on the open count, so each open starts from a blank form. */}
      <McpServerDialog
        key={openCount}
        isOpen={isDialogOpen}
        onOpenChange={setDialogOpen}
        workspaceId={selected.workspace_id}
        isPending={create.isPending}
        error={create.error}
        onSubmit={submit}
      />

      {hasCatalog ? (
        <McpCatalogDialog
          key={`catalog-${catalogOpenCount}`}
          isOpen={isCatalogOpen}
          onOpenChange={setCatalogOpen}
          workspaceId={selected.workspace_id}
          workspaceName={selected.name}
          existing={rows}
          initialPicks={requestedPick ? [requestedPick] : undefined}
          onAddByUrl={() => {
            setCatalogOpen(false)
            openAdd()
          }}
        />
      ) : null}

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(undefined)
        }}
        heading="Delete MCP server"
        body={
          pendingDelete
            ? `${pendingDelete.name} is removed, with its token and its access rules. Requests that still ask for it will fail.`
            : null
        }
        confirmLabel="Delete server"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (!pendingDelete) return
          remove.mutate(
            {
              workspaceId: selected.workspace_id,
              serverId: pendingDelete.id,
            },
            { onSuccess: () => setPendingDelete(undefined) },
          )
        }}
      />
    </section>
  )
}
