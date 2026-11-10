import { useState } from "react"
import { FiTrash2 } from "react-icons/fi"

import type { McpServerConnection, WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Dot } from "@/design-system/indicators/Dot"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import {
  useDisconnectMcp,
  useMcpServerConnections,
  useMyMcpConnections,
  useRemoveMcpServerConnection,
  useStartMcpConnection,
} from "@/shared/api/tools"
import { formatDate } from "@/shared/helpers/format"

function leaveFor(url: string) {
  window.location.assign(url)
}

/**
 * A per-user server's sign-ins: the caller's own connection, with Connect and
 * Disconnect, and for a manager the list of everyone who is connected.
 *
 * Connect leaves the dashboard for the server's own authorization page. The
 * way back is `McpConnectionCallbackPage`, which finishes the sign-in and
 * returns here.
 */
export function McpServerSignIn({
  workspaceId,
  server,
  canManage,
  onLeave = leaveFor,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
  canManage: boolean
  /** Sends the browser to the authorization page; a test observes it here. */
  onLeave?: (url: string) => void
}) {
  const mine = useMyMcpConnections(workspaceId)
  const start = useStartMcpConnection()
  const disconnect = useDisconnectMcp()
  const [isDisconnecting, setDisconnecting] = useState(false)
  const connection = mine.data?.data.find(
    (entry) => entry.server_id === server.id,
  )
  const ids = { workspaceId, serverId: server.id }
  // Nothing is claimed either way until the read has answered, and not after it failed.
  const isKnown = mine.data !== undefined

  const connect = () =>
    start.mutate(
      { ...ids, returnTo: `/tools/mcp-servers/${server.id}` },
      { onSuccess: (started) => onLeave(started.authorization_url) },
    )

  return (
    <SettingsGroup
      title="Your account"
      description="This server uses each person's own account. Your keys and the Playground reach it as you once you connect."
      isBounded
    >
      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-3">
          <ErrorBanner error={mine.error ?? start.error} />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span className="flex items-center gap-2 text-body">
              <Dot className={connection ? "bg-success" : "bg-text-subtle"} />
              {connection
                ? `Connected since ${formatDate(connection.connected_at)}`
                : isKnown
                  ? "Not connected"
                  : mine.isError
                    ? "Status unavailable"
                    : "Checking…"}
            </span>
            {connection ? (
              <div className="flex flex-wrap gap-2">
                <Button onPress={connect} isPending={start.isPending}>
                  Reconnect
                </Button>
                <Button
                  onPress={() => {
                    disconnect.reset()
                    setDisconnecting(true)
                  }}
                >
                  Disconnect
                </Button>
              </div>
            ) : (
              <Button
                variant="primary"
                onPress={connect}
                isPending={start.isPending}
                isDisabled={!isKnown}
                className="min-h-11 md:min-h-0"
              >
                Connect your account
              </Button>
            )}
          </div>
          {connection && !connection.renewable ? (
            <p className="text-caption">
              This server issues no refresh token, so you will be asked to
              connect again when the sign-in expires.
            </p>
          ) : null}
        </div>
        {canManage ? (
          <ConnectedUsers workspaceId={workspaceId} server={server} />
        ) : null}
      </div>

      <ConfirmDialog
        isOpen={isDisconnecting}
        onOpenChange={setDisconnecting}
        heading="Disconnect your account"
        body={`Your keys and the Playground stop reaching ${server.name} until you connect again.`}
        confirmLabel="Disconnect"
        isPending={disconnect.isPending}
        error={disconnect.error}
        onConfirm={() =>
          disconnect.mutate(ids, { onSuccess: () => setDisconnecting(false) })
        }
      />
    </SettingsGroup>
  )
}

function ConnectedUsers({
  workspaceId,
  server,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
}) {
  const connections = useMcpServerConnections(workspaceId, server.id, true)
  const remove = useRemoveMcpServerConnection()
  const [removing, setRemoving] = useState<McpServerConnection>()
  const rows = connections.data?.data ?? []
  const describeUser = (row: McpServerConnection) =>
    row.full_name ?? row.email ?? row.user_id

  const columns: DataTableColumn<McpServerConnection>[] = [
    {
      id: "user",
      header: "User",
      isRowHeader: true,
      cell: (row) => (
        <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-medium break-all">{describeUser(row)}</span>
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
      cell: (row) => (
        <RowAction
          icon={FiTrash2}
          label="Remove connection"
          ariaLabel={`Remove the connection of ${describeUser(row)}`}
          onPress={() => {
            remove.reset()
            setRemoving(row)
          }}
        />
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-title">Connected users</h3>
      <ErrorBanner error={connections.error} />
      {connections.error ? null : rows.length === 0 ? (
        <p className="text-caption">
          {connections.isPending && !connections.data
            ? "Loading…"
            : "Nobody has connected yet."}
        </p>
      ) : (
        <DataTable
          ariaLabel={`Users connected to ${server.name}`}
          columns={columns}
          rows={rows}
          getRowKey={(row) => row.user_id}
        />
      )}
      {connections.data && connections.data.count > rows.length ? (
        <p className="text-caption">
          Showing the first {rows.length} of {connections.data.count}.
        </p>
      ) : null}
      <ConfirmDialog
        isOpen={removing !== undefined}
        onOpenChange={(open) => {
          if (!open) setRemoving(undefined)
        }}
        heading="Remove connection"
        body={
          removing
            ? `${describeUser(removing)} stops reaching ${server.name} until they connect again.`
            : ""
        }
        confirmLabel="Remove connection"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (!removing) return
          remove.mutate(
            { workspaceId, serverId: server.id, userId: removing.user_id },
            { onSuccess: () => setRemoving(undefined) },
          )
        }}
      />
    </div>
  )
}
