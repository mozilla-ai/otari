import { Link } from "@tanstack/react-router"
import { useState } from "react"

import type {
  McpConnection,
  WorkspaceMcpServer,
  WorkspaceMcpServerLogo,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Chip, type ChipTone } from "@/design-system/indicators/Chip"
import { Dot } from "@/design-system/indicators/Dot"
import { McpConnectDialog } from "@/features/tools/McpConnectDialog"
import { McpNotice } from "@/features/tools/McpNotice"
import { McpRequestAnswers } from "@/features/tools/McpRequestAnswers"
import { McpRequestServerDialog } from "@/features/tools/McpRequestServerDialog"
import { McpServerLogo } from "@/features/tools/McpServerLogo"
import { availabilityOf } from "@/features/tools/mcpAvailability"
import { useMcpSignIn } from "@/features/tools/useMcpSignIn"
import { useDisconnectMcp, useMyMcpConnections } from "@/shared/api/tools"
import { formatDate } from "@/shared/helpers/format"
import { useSurfaces } from "@/shared/hooks/useDeployment"

const LIST = "/tools/mcp-servers"

/** What the access rules give the member on a server, in a chip's two words. */
const ACCESS: Record<
  NonNullable<WorkspaceMcpServer["caller_access"]>,
  { label: string; tone: ChipTone }
> = {
  all: { label: "All tools", tone: "success" },
  read: { label: "Read-only", tone: "info" },
  some: { label: "Restricted", tone: "neutral" },
  none: { label: "No access", tone: "warning" },
}

/**
 * A member's MCP servers: the ones they use, as a grid of cards. That is every
 * Shared server an admin has set up, which works as it is, and every Personal
 * server they connected their own account to. "Connect a server" opens the Personal servers they have
 * not connected yet, and a connected one is disconnected from its card. "Request
 * a server", there and on the empty page, asks an admin for one from the catalog,
 * and the admin's answer heads the page until the member dismisses it.
 *
 * A disabled server is left out, and so is a Shared one still waiting for its
 * account: no request reaches either, so it is nothing a member can use.
 */
export function McpMemberServers({
  workspaceId,
  workspaceName,
  servers,
  logos = new Map(),
  isLoading,
  onLeave,
  openWindow,
}: {
  workspaceId: string
  workspaceName: string
  servers: readonly WorkspaceMcpServer[]
  /** Each server's logo, by server id. */
  logos?: ReadonlyMap<string, WorkspaceMcpServerLogo>
  isLoading: boolean
  /** Sends the whole page to the authorization page; a test observes it here. */
  onLeave?: (url: string) => void
  /** Opens the sign-in popup; a test stands in for the browser's window here. */
  openWindow?: () => Window | undefined
}) {
  const mine = useMyMcpConnections(workspaceId)
  const disconnect = useDisconnectMcp()
  const signIn = useMcpSignIn({
    workspaceId,
    returnTo: LIST,
    onLeave,
    openWindow,
  })
  const [disconnecting, setDisconnecting] = useState<WorkspaceMcpServer>()
  const [isConnectOpen, setConnectOpen] = useState(false)
  const [connectOpenCount, setConnectOpenCount] = useState(0)
  // Asking for a server needs the catalog, which a deployment can switch off.
  const hasCatalog = useSurfaces()("mcp_catalog")
  const [isRequestOpen, setRequestOpen] = useState(false)
  const [requestOpenCount, setRequestOpenCount] = useState(0)
  const openRequest = () => {
    setConnectOpen(false)
    setRequestOpenCount((count) => count + 1)
    setRequestOpen(true)
  }
  const connections = new Map(
    (mine.data?.data ?? []).map((entry) => [entry.server_id, entry]),
  )
  const usable = servers.filter(
    (server) => server.enabled && !server.needs_setup,
  )
  const isPersonal = (server: WorkspaceMcpServer) =>
    server.credential_mode === "per_user"
  const used = usable.filter(
    (server) => !isPersonal(server) || connections.has(server.id),
  )
  const connectable = usable.filter(
    (server) => isPersonal(server) && !connections.has(server.id),
  )
  const isReading = isLoading || (mine.isPending && !mine.data)

  const connectButton = (
    <Button
      variant="primary"
      onPress={() => {
        setConnectOpenCount((count) => count + 1)
        setConnectOpen(true)
      }}
      // Nothing is claimed either way until the read has answered.
      isDisabled={mine.data === undefined}
    >
      Connect a server
    </Button>
  )

  return (
    <div className="flex flex-col gap-4">
      <ErrorBanner error={mine.error} />
      {signIn.error ? (
        <McpNotice message={signIn.error} onDismiss={signIn.clearError} />
      ) : null}
      {hasCatalog ? (
        <McpRequestAnswers
          workspaceId={workspaceId}
          servers={servers}
          connected={mine.data ? new Set(connections.keys()) : undefined}
          onConnect={(server) => signIn.start(server.id, "connect")}
          connectingId={
            signIn.running?.purpose === "connect" ? signIn.running.serverId : ""
          }
        />
      ) : null}

      {isReading ? (
        <p className="text-caption">Reading your servers…</p>
      ) : used.length === 0 ? (
        <div className="flex flex-col items-center gap-4 px-4 py-12 text-center">
          <div className="flex flex-col gap-1">
            <h2 className="text-title">No servers yet</h2>
            <p className="max-w-prose text-sm text-muted">
              {connectable.length > 0
                ? "Connect one with your own account to use its tools."
                : hasCatalog
                  ? "This workspace has no servers for you yet. Ask an admin to add one from the catalog."
                  : "This workspace has no servers for you yet."}
            </p>
          </div>
          <div className="flex flex-wrap justify-center gap-2">
            {connectable.length > 0 ? connectButton : null}
            {hasCatalog ? (
              <Button
                variant={connectable.length > 0 ? "ghost" : "primary"}
                onPress={openRequest}
              >
                Request a server
              </Button>
            ) : null}
          </div>
        </div>
      ) : (
        <section className="flex flex-col gap-3">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div className="flex flex-col gap-1">
              <h2 className="text-title">
                Your servers
                <span className="font-normal text-subtle">
                  {" "}
                  ({used.length})
                </span>
              </h2>
              <p className="max-w-prose text-sm text-muted">
                Shared servers work as they are. Personal ones use your own
                account once you connect it.
              </p>
            </div>
            <div className="shrink-0">{connectButton}</div>
          </div>
          <ul
            aria-label="Your MCP servers"
            // Rows stretch, so every card in a row is as tall as its tallest.
            className="grid auto-rows-fr grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3"
          >
            {used.map((server) => (
              <ServerCard
                key={server.id}
                server={server}
                logo={logos.get(server.id)}
                connection={connections.get(server.id)}
                onDisconnect={() => {
                  disconnect.reset()
                  setDisconnecting(server)
                }}
              />
            ))}
          </ul>
        </section>
      )}

      <McpConnectDialog
        key={connectOpenCount}
        isOpen={isConnectOpen}
        onOpenChange={setConnectOpen}
        servers={connectable}
        logos={logos}
        pendingId={
          signIn.running?.purpose === "connect" ? signIn.running.serverId : ""
        }
        onPick={(server) => {
          signIn.start(server.id, "connect")
          setConnectOpen(false)
        }}
        onRequest={hasCatalog ? openRequest : undefined}
      />

      {hasCatalog ? (
        <McpRequestServerDialog
          key={`request-${requestOpenCount}`}
          isOpen={isRequestOpen}
          onOpenChange={setRequestOpen}
          workspaceId={workspaceId}
          workspaceName={workspaceName}
          existing={servers}
        />
      ) : null}

      <ConfirmDialog
        isOpen={disconnecting !== undefined}
        onOpenChange={(open) => {
          if (!open) setDisconnecting(undefined)
        }}
        heading="Disconnect your account"
        body={
          disconnecting
            ? `Your keys and the Playground stop using ${disconnecting.name} until you connect again.`
            : ""
        }
        confirmLabel="Disconnect"
        isPending={disconnect.isPending}
        error={disconnect.error}
        onConfirm={() => {
          if (!disconnecting) return
          disconnect.mutate(
            { workspaceId, serverId: disconnecting.id },
            { onSuccess: () => setDisconnecting(undefined) },
          )
        }}
      />
    </div>
  )
}

/**
 * One server the member uses. The whole card opens the server's page: the name
 * is the link and it is stretched over the card, with Disconnect lifted above
 * it so the two never share a press.
 */
function ServerCard({
  server,
  logo,
  connection,
  onDisconnect,
}: {
  server: WorkspaceMcpServer
  logo?: WorkspaceMcpServerLogo
  /** The member's own connection to a Personal server; none on a Shared one. */
  connection?: McpConnection
  onDisconnect: () => void
}) {
  const access = server.caller_access ? ACCESS[server.caller_access] : undefined
  const isShared = server.credential_mode !== "per_user"
  return (
    <li className="relative flex h-full min-w-0 flex-col gap-3 border border-border p-4 transition-colors focus-within:bg-surface-alt hover:bg-surface-alt motion-reduce:transition-none">
      <div className="flex min-w-0 items-start gap-3">
        <McpServerLogo
          name={server.name}
          logoUrl={logo?.logo_url}
          tone={logo?.logo_tone}
        />
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <Link
            to="/tools/mcp-servers/$serverId"
            params={{ serverId: server.id }}
            className="line-clamp-2 font-medium break-words after:absolute after:inset-0 after:content-['']"
          >
            {server.name}
          </Link>
          {/* Two lines kept whether or not there is a hint, so cards line up. */}
          <p className="line-clamp-2 min-h-[2lh] text-caption break-words">
            {server.purpose_hint}
          </p>
        </div>
        <span className="flex shrink-0 flex-col items-end gap-1">
          {access ? <Chip tone={access.tone}>{access.label}</Chip> : null}
          {isShared ? <Chip>Shared</Chip> : null}
          {availabilityOf(server) === "on_request" ? (
            <Chip tone="accent">On request</Chip>
          ) : null}
        </span>
      </div>
      {/* Pinned to the foot of every card, whatever sits above it. */}
      <div className="mt-auto flex min-h-11 flex-wrap items-center justify-between gap-2 md:min-h-8">
        {connection ? (
          <>
            <span className="flex items-center gap-2 text-sm text-muted">
              <Dot className="bg-success" />
              Connected since {formatDate(connection.connected_at)}
            </span>
            <Button
              size="sm"
              onPress={onDisconnect}
              aria-label={`Disconnect ${server.name}`}
              className="relative z-10 min-h-11 shrink-0 md:min-h-0"
            >
              Disconnect
            </Button>
          </>
        ) : null}
      </div>
    </li>
  )
}
