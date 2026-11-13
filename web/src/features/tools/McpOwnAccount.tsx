import { Dropdown, Label } from "@heroui/react"
import { useState } from "react"
import { FiChevronDown } from "react-icons/fi"

import type { McpConnection, WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Dot } from "@/design-system/indicators/Dot"
import { McpNotice } from "@/features/tools/McpNotice"
import { asksForOAuthClient } from "@/features/tools/mcpErrors"
import { useMcpSignIn } from "@/features/tools/useMcpSignIn"
import { useDisconnectMcp, useMyMcpConnections } from "@/shared/api/tools"

/**
 * The viewer's own account on a Personal server: whether it is connected, and
 * connecting, reconnecting and disconnecting it. One instance serves the page's
 * heading control, the notes under the heading and the viewer's row in
 * Connected users, so they agree and share one confirmation.
 */
export function useOwnMcpAccount({
  workspaceId,
  server,
  onLeave,
  openWindow,
}: {
  /** Null while there is nothing to read: no workspace yet, or a server that is not Personal. */
  workspaceId: string | null
  server: Pick<WorkspaceMcpServer, "id" | "name">
  /** Sends the whole page to the authorization page; a test observes it here. */
  onLeave?: (url: string) => void
  /** Opens the popup; a test stands in for the browser's window here. */
  openWindow?: () => Window | undefined
}) {
  const mine = useMyMcpConnections(workspaceId)
  const disconnect = useDisconnectMcp()
  const [isDisconnecting, setDisconnecting] = useState(false)
  const signIn = useMcpSignIn({
    workspaceId: workspaceId ?? "",
    returnTo: `/tools/mcp-servers/${server.id}`,
    onLeave,
    openWindow,
  })
  const connection: McpConnection | undefined = mine.data?.data.find(
    (entry) => entry.server_id === server.id,
  )
  return {
    workspaceId,
    server,
    connection,
    /** Nothing is claimed either way until the read has answered, and not after it failed. */
    isKnown: mine.data !== undefined,
    readError: mine.error,
    isStarting: signIn.running?.purpose === "connect",
    signInError: signIn.error,
    clearSignInError: signIn.clearError,
    connect: () => signIn.start(server.id, "connect"),
    openDisconnect: () => {
      disconnect.reset()
      setDisconnecting(true)
    },
    disconnect,
    isDisconnecting,
    setDisconnecting,
  }
}

export type OwnMcpAccount = ReturnType<typeof useOwnMcpAccount>

/**
 * The heading's control for the viewer's own account, worded so it cannot be
 * read as connecting someone else: "Connect your account" until they have, then
 * "Your account: connected", which opens Reconnect and Disconnect.
 */
export function McpOwnAccountControl({ own }: { own: OwnMcpAccount }) {
  if (!own.isKnown) return null
  if (!own.connection) {
    return (
      <Button
        variant="primary"
        size="sm"
        onPress={own.connect}
        isPending={own.isStarting}
        className="min-h-11 md:min-h-0"
      >
        Connect your account
      </Button>
    )
  }
  return (
    <Dropdown>
      <Button
        size="sm"
        aria-label="Your account: connected. Reconnect or disconnect"
        className="min-h-11 md:min-h-0"
      >
        <Dot className="bg-success" />
        Your account: connected
        <FiChevronDown aria-hidden className="size-3.5 shrink-0 text-muted" />
      </Button>
      <Dropdown.Popover placement="bottom end">
        <Dropdown.Menu
          aria-label="Your account"
          onAction={(id) => {
            if (id === "reconnect") own.connect()
            if (id === "disconnect") own.openDisconnect()
          }}
        >
          <Dropdown.Item
            id="reconnect"
            textValue="Reconnect"
            isDisabled={own.isStarting}
          >
            <Label className="text-body">Reconnect</Label>
          </Dropdown.Item>
          <Dropdown.Item id="disconnect" textValue="Disconnect">
            <Label className="text-body">Disconnect</Label>
          </Dropdown.Item>
        </Dropdown.Menu>
      </Dropdown.Popover>
    </Dropdown>
  )
}

/**
 * What the viewer's own sign-in has to say, under the page's heading: a
 * refusal, a wait, an expiry. Also holds the one Disconnect confirmation.
 */
export function McpOwnAccountNotes({
  own,
  onAddClient,
}: {
  own: OwnMcpAccount
  /** Opens Settings on the OAuth client ID, for an admin whose sign-in asked for one. */
  onAddClient?: () => void
}) {
  return (
    <>
      <ErrorBanner error={own.readError} />
      {own.signInError ? (
        <McpNotice
          message={own.signInError}
          action={
            onAddClient && asksForOAuthClient(own.signInError) ? (
              <Button size="sm" onPress={onAddClient}>
                Open Settings
              </Button>
            ) : undefined
          }
          onDismiss={own.clearSignInError}
        />
      ) : null}
      {own.isStarting ? (
        <p className="text-caption">
          Finish signing in in the window that opened.
        </p>
      ) : null}
      {own.connection && !own.connection.renewable ? (
        <p className="text-caption">
          You'll need to connect again when this sign-in expires.
        </p>
      ) : null}
      <ConfirmDialog
        isOpen={own.isDisconnecting}
        onOpenChange={own.setDisconnecting}
        heading="Disconnect your account"
        body={`Your keys and the Playground stop reaching ${own.server.name} until you connect again.`}
        confirmLabel="Disconnect"
        isPending={own.disconnect.isPending}
        error={own.disconnect.error}
        onConfirm={() =>
          own.disconnect.mutate(
            { workspaceId: own.workspaceId ?? "", serverId: own.server.id },
            { onSuccess: () => own.setDisconnecting(false) },
          )
        }
      />
    </>
  )
}
