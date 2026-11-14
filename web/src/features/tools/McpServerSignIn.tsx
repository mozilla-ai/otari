import { useState } from "react"
import { FiTrash2 } from "react-icons/fi"

import type {
  McpConnection,
  McpServerConnection,
  WorkspaceMcpServer,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import {
  PAGE_SIZE_OPTIONS,
  TablePagination,
} from "@/design-system/data/TablePagination"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Dot } from "@/design-system/indicators/Dot"
import {
  SettingsBlock,
  SettingsGroup,
} from "@/design-system/layout/SettingsGroup"
import type { OwnMcpAccount } from "@/features/tools/McpOwnAccount"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useMcpServerConnections,
  useRemoveMcpServerConnection,
} from "@/shared/api/tools"
import { formatDate } from "@/shared/helpers/format"

const MEANING =
  "Personal: each member signs in with their own account, and their keys and the Playground then use it as them."

/**
 * A personal server's sign-ins. A member gets their own connection, with
 * Connect and Disconnect. A manager gets the plain list of everyone who is
 * connected, their own row first; their own account is connected from the
 * page's heading (`McpOwnAccountControl`), which `own` comes from.
 *
 * Connect opens the server's own authorization page in a popup and updates
 * this page when it finishes (`useMcpSignIn`); where the popup is blocked the
 * page itself goes, and `McpConnectionCallbackPage` brings it back here. This
 * is the person's own connection, apart from the check a manager runs under
 * Exposed tools, which connects nobody.
 */
export function McpServerSignIn({
  workspaceId,
  server,
  canManage,
  own,
  onOpenUserRule,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
  canManage: boolean
  /** The viewer's own account on this server, shared with the page's heading. */
  own: OwnMcpAccount
  /** Opens a connected user's access rule for this server. */
  onOpenUserRule?: (userId: string) => void
}) {
  const { connection, isKnown, isStarting, hasExpired } = own

  if (canManage) {
    return (
      <ConnectedUsers
        workspaceId={workspaceId}
        server={server}
        own={connection}
        onOpenUserRule={onOpenUserRule}
        onDisconnect={own.openDisconnect}
      />
    )
  }

  return (
    <SettingsGroup
      title="Your account"
      description="This server uses your own account, not a shared one. Once you connect, your keys and the Playground use it as you."
      isBounded
    >
      <SettingsBlock>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="flex items-center gap-2 text-body">
            <Dot
              className={
                hasExpired
                  ? "bg-warning"
                  : connection
                    ? "bg-success"
                    : "bg-text-subtle"
              }
            />
            {hasExpired
              ? "Your sign-in has expired"
              : connection
                ? `Connected since ${formatDate(connection.connected_at)}`
                : isKnown
                  ? "Not connected"
                  : own.readError
                    ? "Status unavailable"
                    : "Checking…"}
          </span>
          {connection && !hasExpired ? (
            <div className="flex flex-wrap gap-2">
              <Button
                onPress={own.connect}
                isPending={isStarting}
                className="min-h-11 md:min-h-0"
              >
                Reconnect
              </Button>
              <Button
                onPress={own.openDisconnect}
                className="min-h-11 md:min-h-0"
              >
                Disconnect
              </Button>
            </div>
          ) : (
            <Button
              variant="primary"
              onPress={own.connect}
              isPending={isStarting}
              isDisabled={!isKnown}
              className="min-h-11 md:min-h-0"
            >
              {hasExpired ? "Sign in again" : "Connect your account"}
            </Button>
          )}
        </div>
      </SettingsBlock>
    </SettingsGroup>
  )
}

/**
 * Everyone connected to a personal server, for a manager: a plain list, the
 * manager's own row first and marked. Every row has the one remove action; on
 * the manager's own row it is Disconnect, with its confirmation.
 */
function ConnectedUsers({
  workspaceId,
  server,
  own,
  onOpenUserRule,
  onDisconnect,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
  /** The manager's own connection, when they have one. */
  own?: McpConnection
  onOpenUserRule?: (userId: string) => void
  onDisconnect: () => void
}) {
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState<number>(PAGE_SIZE_OPTIONS[0])
  const connections = useMcpServerConnections(
    workspaceId,
    server.id,
    { page, pageSize },
    true,
  )
  const remove = useRemoveMcpServerConnection()
  const [removing, setRemoving] = useState<McpServerConnection>()
  const total = connections.data?.count
  // A page emptied by a removal is left for the last one that still has rows.
  const lastPage =
    total === undefined ? page : Math.max(0, Math.ceil(total / pageSize) - 1)
  if (page > lastPage) setPage(lastPage)
  // A connection is held by the dashboard account's own id, so the admin
  // looking finds their row by it. Theirs goes first, marked.
  const caller = useOrganizationContext().data?.caller
  const callerId = caller?.user_id ?? ""
  const isOwn = (row: McpServerConnection) =>
    callerId !== "" && row.user_id === callerId
  const listed = connections.data?.data ?? []
  // The manager's own row heads the first page wherever the list has it, and
  // before the list has caught up with a sign-in that just finished.
  const ownRow: McpServerConnection[] =
    page > 0
      ? []
      : listed.some(isOwn) || !own || callerId === ""
        ? listed.filter(isOwn)
        : [
            {
              user_id: callerId,
              email: caller?.email ?? null,
              full_name: caller?.full_name ?? null,
              connected_at: own.connected_at,
              updated_at: own.updated_at,
            },
          ]
  const rows = [...ownRow, ...listed.filter((row) => !isOwn(row))]
  const describeUser = (row: McpServerConnection) =>
    row.full_name ?? row.email ?? row.user_id

  const columns: DataTableColumn<McpServerConnection>[] = [
    {
      id: "user",
      header: "User",
      isRowHeader: true,
      cell: (row) => (
        <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-medium break-all">
            {describeUser(row)}
            {isOwn(row) ? (
              <span className="font-normal text-muted"> (you)</span>
            ) : null}
          </span>
          {row.email && row.email !== describeUser(row) ? (
            <span className="text-caption break-all">{row.email}</span>
          ) : null}
        </span>
      ),
    },
    {
      id: "connected",
      header: "Connected",
      cell: (row) => formatDate(row.connected_at),
    },
    {
      id: "actions",
      header: "",
      cell: (row) =>
        isOwn(row) ? (
          <RowActionRow>
            <RowAction
              icon={FiTrash2}
              label="Disconnect"
              ariaLabel="Disconnect your account"
              onPress={onDisconnect}
            />
          </RowActionRow>
        ) : (
          <RowActionRow>
            <RowAction
              icon={FiTrash2}
              label="Remove connection"
              ariaLabel={`Remove the connection of ${describeUser(row)}`}
              onPress={() => {
                remove.reset()
                setRemoving(row)
              }}
            />
          </RowActionRow>
        ),
    },
  ]

  return (
    <SettingsGroup
      title="Connected users"
      description={`${MEANING}${onOpenUserRule ? " Click a member to set their access rule." : ""}`}
      isBounded
    >
      {connections.error ? (
        <SettingsBlock>
          <ErrorBanner error={connections.error} />
        </SettingsBlock>
      ) : rows.length === 0 ? (
        <SettingsBlock>
          <p className="text-caption">
            {connections.isPending && !connections.data
              ? "Loading…"
              : "Nobody has connected yet."}
          </p>
        </SettingsBlock>
      ) : (
        <DataTable
          ariaLabel={`Users connected to ${server.name}`}
          mobileLayout="cards"
          columns={columns}
          rows={rows}
          getRowKey={(row) => row.user_id}
          onRowAction={onOpenUserRule}
        />
      )}
      {total !== undefined && total > PAGE_SIZE_OPTIONS[0] ? (
        <SettingsBlock>
          <TablePagination
            page={page}
            pageSize={pageSize}
            total={total}
            rowsOnPage={listed.length}
            onPageChange={setPage}
            onPageSizeChange={(size) => {
              setPageSize(size)
              setPage(0)
            }}
            isFetching={connections.isFetching}
          />
        </SettingsBlock>
      ) : null}
      {removing ? (
        <ConfirmDialog
          isOpen
          onOpenChange={(open) => {
            if (!open) setRemoving(undefined)
          }}
          heading="Remove connection"
          body={`${describeUser(removing)} stops reaching ${server.name} until they connect again.`}
          confirmLabel="Remove connection"
          isPending={remove.isPending}
          error={remove.error}
          onConfirm={() =>
            remove.mutate(
              { workspaceId, serverId: server.id, userId: removing.user_id },
              { onSuccess: () => setRemoving(undefined) },
            )
          }
        />
      ) : null}
    </SettingsGroup>
  )
}
