import { Button, Card, Chip } from "@heroui/react"
import { useState } from "react"

import type { WorkspaceMcpServer } from "@/client"
import { canManageWorkspace } from "@/features/organization/roles"
import {
  McpServerDialog,
  type McpServerDraft,
} from "@/features/tools/McpServerDialog"
import {
  useCreateWorkspaceMcpServer,
  useDeleteWorkspaceMcpServer,
  useOrganizationContext,
  useUpdateWorkspaceMcpServer,
  useWorkspaceMcpServers,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { ErrorBanner, InfoBanner } from "@/shared/components/ui"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

// The MCP servers this workspace has registered, which a request reaches by
// naming their ids in `mcp_server_ids` (see docs/mcp.md).
//
// A list, where the two workspace config planes beside it on this page hold one
// row each, and that difference is the whole shape of this card: there is no
// deployment-wide MCP server for a workspace row to narrow, so a row here is
// the configuration rather than a narrowing of one. Nothing on this card can
// therefore say "deployment default", and there is no third stance: a server is
// registered or it is not, and `enabled` decides whether requests reach it.
//
// **No client-side copy of the server's cap.** A workspace may hold only so
// many servers, and that number lives in the service
// (`MAX_MCP_SERVERS_PER_WORKSPACE`) where the create path enforces it under a
// workspace lock. Restating it here would be a second copy to go stale, so the
// 51st Add is left to earn the server's own refusal, which names the real
// number and lands in the dialog's error banner.

function tokenChip(server: WorkspaceMcpServer) {
  // The token itself is write-only, so this is the only thing the API can say
  // about it, and the only thing worth a column: whether a request to this
  // server will carry a credential.
  return server.has_token ? (
    <Chip size="sm" variant="secondary">
      Stored
    </Chip>
  ) : (
    <span className="text-muted">None</span>
  )
}

export function WorkspaceMcpServersCard({
  showHeading = true,
}: {
  /** Suppressed on the page whose own title already says "MCP servers". */
  showHeading?: boolean
}) {
  const { selected, isLoading: workspaceLoading } = useSelectedWorkspace()
  const context = useOrganizationContext()
  // The client half of the gate the service enforces, and it gates the *read*
  // too: these rows name the external endpoints the gateway connects to on the
  // workspace's behalf, so the service admin-gates listing them as well as
  // changing them, and asking as a member would earn a 403 banner over a table
  // that could never fill.
  const manages = canManageWorkspace(context.data, selected?.role)
  const workspaceId = selected && manages ? selected.workspace_id : null
  const query = useWorkspaceMcpServers(workspaceId)
  const create = useCreateWorkspaceMcpServer()
  const update = useUpdateWorkspaceMcpServer()
  const remove = useDeleteWorkspaceMcpServer()

  const [isDialogOpen, setDialogOpen] = useState(false)
  const [editing, setEditing] = useState<WorkspaceMcpServer>()
  const [pendingDelete, setPendingDelete] = useState<WorkspaceMcpServer>()

  const heading = showHeading ? (
    <h2 className="text-sm font-semibold text-foreground">MCP servers</h2>
  ) : null

  if (!selected) {
    return (
      <section className="flex flex-col gap-2">
        {heading}
        <InfoBanner>
          {workspaceLoading
            ? "Reading the workspaces you belong to."
            : "MCP servers are registered on a workspace you belong to. An owner or admin can add you to one on the Workspaces page."}
        </InfoBanner>
      </section>
    )
  }

  if (!manages) {
    return (
      <section className="flex flex-col gap-2">
        {heading}
        <InfoBanner>
          The MCP servers for {selected.name} are managed by an owner or admin
          of the workspace, or of the organization. They name endpoints the
          gateway connects to, so only they can list them.
        </InfoBanner>
      </section>
    )
  }

  const rows = query.data?.data ?? []

  // Each opener clears the mutation it will render the error of. Without this a
  // refused write leaves its banner up, so reopening the form shows the last
  // 409 over a blank one. Same reset `PasskeysCard` does for the same reason.
  const openAdd = () => {
    setEditing(undefined)
    create.reset()
    setDialogOpen(true)
  }

  const openEdit = (server: WorkspaceMcpServer) => {
    setEditing(server)
    update.reset()
    setDialogOpen(true)
  }

  const openDelete = (server: WorkspaceMcpServer) => {
    remove.reset()
    setPendingDelete(server)
  }

  const submit = (draft: McpServerDraft) => {
    const onDone = { onSuccess: () => setDialogOpen(false) }
    if (editing) {
      update.mutate(
        {
          workspaceId: selected.workspace_id,
          serverId: editing.id,
          // `authorization_token` is `undefined` unless the operator typed one
          // or asked to remove it, and `JSON.stringify` drops an undefined
          // value, so the request omits the field and the stored token is left
          // alone. That is the endpoint's own "not sent" state, and the reason
          // an edit can restate every other field safely.
          body: draft,
        },
        onDone,
      )
      return
    }
    create.mutate({ workspaceId: selected.workspace_id, body: draft }, onDone)
  }

  const columns: DataTableColumn<WorkspaceMcpServer>[] = [
    {
      id: "name",
      header: "Name",
      isRowHeader: true,
      cell: (row) => <span className="font-medium">{row.name}</span>,
    },
    {
      id: "url",
      header: "URL",
      cell: (row) => <code className="break-all text-xs">{row.url}</code>,
    },
    {
      id: "allowed_tools",
      header: "Tools",
      cell: (row) =>
        row.allowed_tools === null ? (
          <span className="text-muted">All</span>
        ) : (
          // The names themselves would be an unbounded column (the list runs to
          // hundreds), so the count is the cell and the list is in the dialog.
          <span>{row.allowed_tools.length} allowed</span>
        ),
    },
    { id: "token", header: "Token", cell: tokenChip },
    {
      id: "enabled",
      header: "Status",
      cell: (row) => (
        <Chip size="sm" variant="secondary">
          <span className={row.enabled ? "text-success" : "text-muted"}>
            {row.enabled ? "Enabled" : "Disabled"}
          </span>
        </Chip>
      ),
    },
    {
      id: "actions",
      header: "",
      cell: (row) => (
        <div className="flex justify-end gap-2">
          <Button size="sm" variant="ghost" onPress={() => openEdit(row)}>
            Edit
          </Button>
          <Button size="sm" variant="ghost" onPress={() => openDelete(row)}>
            Delete
          </Button>
        </div>
      ),
    },
  ]

  return (
    <section className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        {heading}
        <Button size="sm" variant="primary" onPress={openAdd}>
          Add MCP server
        </Button>
      </div>

      <p className="text-sm text-muted">
        Endpoints requests billed to {selected.name} can use by naming their ids
        in <code className="font-mono">mcp_server_ids</code>. A request may
        still pass its own servers inline; these are the ones it does not have
        to carry a URL or a token for. Tokens are stored encrypted and are never
        shown again.
      </p>

      <ErrorBanner error={query.error} />

      <Card>
        <Card.Content className="p-0">
          <DataTable
            ariaLabel={`MCP servers for ${selected.name}`}
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.id}
            isLoading={query.isPending && !query.data}
            // Deliberately asserts nothing about the workspace: an empty table
            // is also what a failed request leaves behind, and the banner above
            // is the only thing that knows which of the two happened.
            emptyContent="No MCP server registered. Add one to let this workspace's requests name it by id."
          />
        </Card.Content>
      </Card>

      <McpServerDialog
        isOpen={isDialogOpen}
        onOpenChange={setDialogOpen}
        editing={editing}
        isPending={create.isPending || update.isPending}
        error={editing ? update.error : create.error}
        onSubmit={submit}
      />

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(undefined)
        }}
        heading="Delete MCP server"
        body={
          pendingDelete
            ? `${pendingDelete.name} and the token stored with it are removed. A request that still names its id is refused, so update the callers that use it.`
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
