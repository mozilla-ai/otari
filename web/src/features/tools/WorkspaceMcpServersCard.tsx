import { useState } from "react"
import { FiEdit2, FiRefreshCw, FiTrash2 } from "react-icons/fi"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { CopyableValue } from "@/design-system/actions/CopyField"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { Popover } from "@/design-system/overlays/Popover"
import { canManageWorkspace } from "@/features/organization/roles"
import {
  McpServerDialog,
  type McpServerDraft,
} from "@/features/tools/McpServerDialog"
import {
  healthDetail,
  healthLabel,
  healthNeedsAttention,
  toolCountLabel,
} from "@/features/tools/mcpServerHealth"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useCheckWorkspaceMcpServer,
  useCreateWorkspaceMcpServer,
  useDeleteWorkspaceMcpServer,
  useUpdateWorkspaceMcpServer,
  useWorkspaceMcpServers,
} from "@/shared/api/tools"
import { formatDateTime } from "@/shared/helpers/format"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

// The MCP servers this workspace has registered, which a request reaches by
// naming their ids in `mcp_server_ids` (see docs/mcp.md).
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

function tokenChip(server: WorkspaceMcpServer) {
  // The token itself is write-only, so this is the only thing the API can say
  // about it, and the only thing worth a column: whether a request to this
  // server will carry a credential.
  // Affirmative: a server that carries a credential is marked, and one that
  // does not is the unmarked state rather than a second badge saying so.
  return server.has_token ? (
    <span className="flex items-center gap-2 text-mono-caption text-muted">
      <Dot className="bg-accent" />
      STORED
    </span>
  ) : (
    <span className="text-subtle">None</span>
  )
}

// The last connection check, as of when it ran. The label is the cell; the
// sentence saying what to fix and when it was checked open from it, as a
// pressable rather than a title, so a keyboard or a finger reaches them too.
function healthCell(server: WorkspaceMcpServer) {
  const health = server.health
  if (health === null) {
    return <span className="text-subtle">Not checked</span>
  }
  const dot =
    health.status === "failing"
      ? "bg-danger"
      : healthNeedsAttention(health)
        ? "bg-warning"
        : "bg-success"
  return (
    <Popover
      label={`Last check of ${server.name}`}
      trigger={
        // Pulled left by the ghost button's 16px inset (1px edge, 15px
        // padding), so the dot lines up with the column header and the cells
        // beside it rather than sitting a padding's width in.
        <Button className="-ms-4 gap-2">
          <Dot className={dot} />
          {/* The cell's own type, so the label reads like the Token and
              Status cells beside it rather than like a button. */}
          <span className="text-mono-caption text-muted">
            {healthLabel(health)}
          </span>
        </Button>
      }
    >
      <div className="flex max-w-xs flex-col gap-1">
        <p className="text-sm">{healthDetail(health)}</p>
        {(health.tools ?? []).length > 0 ? (
          <p className="max-h-32 overflow-y-auto font-mono text-xs break-words">
            {(health.tools ?? []).join(", ")}
          </p>
        ) : null}
        <p className="text-caption">
          Checked {formatDateTime(health.checked_at)}
        </p>
      </div>
    </Popover>
  )
}

/** The page's own opening line, which the `/tools` card does not carry. */
const PAGE_DESCRIPTION =
  "MCP servers this workspace's requests can use by id. Each is checked for SSRF safety and tested when saved."

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
  const create = useCreateWorkspaceMcpServer()
  const update = useUpdateWorkspaceMcpServer()
  const remove = useDeleteWorkspaceMcpServer()
  const check = useCheckWorkspaceMcpServer()

  const [isDialogOpen, setDialogOpen] = useState(false)
  // Bumped on every open and used as the dialog's key, so the draft is cleared
  // on the way in. Clearing it on close would blank the fields while the dialog
  // is still animating away.
  const [openCount, setOpenCount] = useState(0)
  const [editing, setEditing] = useState<WorkspaceMcpServer>()
  const [saved, setSaved] = useState<WorkspaceMcpServer>()
  const [pendingDelete, setPendingDelete] = useState<WorkspaceMcpServer>()

  const isPage = variant === "page"

  if (!selected) {
    return (
      <section className="flex flex-col gap-2">
        {isPage ? (
          <PageIntro title="MCP servers">{PAGE_DESCRIPTION}</PageIntro>
        ) : (
          <h2 className="text-title">MCP servers</h2>
        )}
        <InfoBanner>
          {workspaceLoading
            ? "Reading the workspaces you belong to."
            : "MCP servers are registered on a workspace you belong to. An owner or admin can add you to one on the Workspaces page."}
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
    setSaved(undefined)
    create.reset()
    setOpenCount((count) => count + 1)
    setDialogOpen(true)
  }

  const openEdit = (server: WorkspaceMcpServer) => {
    setEditing(server)
    setSaved(undefined)
    update.reset()
    setOpenCount((count) => count + 1)
    setDialogOpen(true)
  }

  const openDelete = (server: WorkspaceMcpServer) => {
    remove.reset()
    setPendingDelete(server)
  }

  const submit = (draft: McpServerDraft) => {
    // A save checks the connection when it adds a server or changes its URL,
    // token or allow-list, the same rule the service applies. When that check
    // found a problem, or could not run, the dialog stays open on the stored
    // row and says so, so the fix is an edit rather than a second add.
    const checked =
      editing === undefined ||
      draft.url !== editing.url ||
      draft.authorization_token !== undefined ||
      JSON.stringify(draft.allowed_tools) !==
        JSON.stringify(editing.allowed_tools)
    const settle = (server: WorkspaceMcpServer) => {
      if (
        checked &&
        (server.health === null || healthNeedsAttention(server.health))
      ) {
        setEditing(server)
        setSaved(server)
        return
      }
      setDialogOpen(false)
    }
    const onDone = { onSuccess: settle }
    if (editing) {
      update.mutate(
        {
          workspaceId: selected.workspace_id,
          serverId: editing.id,
          // The token's states ride on `authorization_token` being absent,
          // which `JSON.stringify` does for `undefined`. That is what lets an
          // edit restate every other field safely; see `McpServerDialog`.
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
      id: "server_id",
      header: "ID",
      // What a request names in `mcp_server_ids`. Truncated because a UUID
      // would crowd the URL column; the copy and the tooltip carry all of it.
      cell: (row) => (
        <CopyableValue value={row.id} label={`id for ${row.name}`}>
          <code className="font-mono text-xs" title={row.id}>
            {row.id.slice(0, 8)}…
          </code>
        </CopyableValue>
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
      // A passing check gives the real count; without one, the allow-list is
      // all there is to say. `[]` is a list that admits nothing, which is how
      // the tool loop reads it: this form never sends one, but a row created
      // over the API can hold it. The names would be an unbounded column (the
      // list runs to hundreds), so they live in the Health popover and dialog.
      cell: (row) => {
        const counted = toolCountLabel(row.health, row.allowed_tools !== null)
        if (counted) return <span className="whitespace-nowrap">{counted}</span>
        return row.allowed_tools === null ? (
          <span className="text-muted">All</span>
        ) : (
          <span>{row.allowed_tools.length} allowed</span>
        )
      },
    },
    { id: "token", header: "Token", cell: tokenChip },
    {
      id: "enabled",
      header: "Status",
      // `color` rather than a span that re-skins the chip from inside: it is
      // the component's own prop, it carries the right foreground for its fill
      // (`--chip-fg` per `.chip--success`), and it is what the two cards next
      // door already use.
      cell: (row) => (
        <span className="flex items-center gap-2 text-mono-caption text-muted">
          <Dot className={row.enabled ? "bg-success" : "bg-text-subtle"} />
          {row.enabled ? "ENABLED" : "DISABLED"}
        </span>
      ),
    },
    { id: "health", header: "Health", cell: healthCell },
  ]
  if (manages) {
    columns.push({
      id: "actions",
      header: "",
      cell: (row) => (
        <RowActionRow>
          <RowAction
            icon={FiRefreshCw}
            label="Recheck"
            isDisabled={check.isPending && check.variables?.serverId === row.id}
            onPress={() =>
              check.mutate({
                workspaceId: selected.workspace_id,
                serverId: row.id,
              })
            }
          />
          <RowAction
            icon={FiEdit2}
            label="Edit"
            onPress={() => openEdit(row)}
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
    <Button variant="primary" onPress={openAdd}>
      Add MCP server
    </Button>
  ) : null

  return (
    <section className="flex flex-col gap-2">
      {isPage ? (
        <PageIntro title="MCP servers" action={addButton}>
          {PAGE_DESCRIPTION}
        </PageIntro>
      ) : (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-title">MCP servers</h2>
          {addButton}
        </div>
      )}

      <p className="max-w-prose text-sm text-muted">
        Requests billed to {selected.name} use these by listing their ids in{" "}
        <code className="font-mono">mcp_server_ids</code>, with no URL or token
        of their own. Tokens are encrypted and never shown again.
        {manages
          ? null
          : " Only an owner or admin of the workspace or organization can change them."}
      </p>

      <ErrorBanner error={query.error} />
      <ErrorBanner error={check.error} />

      <TableScrollFrame className="otari-mcp-table">
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
      </TableScrollFrame>

      {/* Keyed on the open count, so each open remounts the form: it seeds its
          fields on mount, and one row's draft must not survive into the next
          row's dialog. */}
      <McpServerDialog
        key={openCount}
        isOpen={isDialogOpen}
        onOpenChange={setDialogOpen}
        editing={editing}
        saved={saved}
        workspaceId={selected.workspace_id}
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
