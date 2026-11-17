import { useState } from "react"

import type { WorkspaceMcpServer, WorkspaceMcpServerLogo } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { SearchField } from "@/design-system/forms/SearchField"
import { McpServerLogo } from "@/features/tools/McpServerLogo"

/**
 * The Personal servers a member can still connect, as a grid of tiles with a
 * search above. Pressing a tile is the whole action: it starts that server's
 * sign-in, so there is no separate Connect button to find. A tile has no edge
 * of its own, only a fill under the pointer or the focus. The footer leads to
 * the catalog, for a server the workspace does not have yet.
 */
export function McpConnectDialog({
  isOpen,
  onOpenChange,
  servers,
  logos,
  pendingId,
  onPick,
  onRequest,
}: {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  /** The Personal servers the member has not connected yet. */
  servers: readonly WorkspaceMcpServer[]
  /** Each server's logo, by server id. */
  logos: ReadonlyMap<string, WorkspaceMcpServerLogo>
  /** The server whose sign-in is starting, if any. */
  pendingId: string
  onPick: (server: WorkspaceMcpServer) => void
  /** Opens the catalog to ask an admin for a server this workspace lacks; absent where there is no catalog. */
  onRequest?: () => void
}) {
  const [query, setQuery] = useState("")
  const needle = query.trim().toLowerCase()
  const shown = servers.filter(
    (server) =>
      needle === "" ||
      server.name.toLowerCase().includes(needle) ||
      (server.purpose_hint ?? "").toLowerCase().includes(needle),
  )

  return (
    <Dialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      title="Connect a server"
      description="Pick a server to sign in to it with your own account. Your keys and the Playground then use it as you."
      size="lg"
      footerStart={
        onRequest
          ? "Not here? Ask an admin to add it from the catalog."
          : undefined
      }
      actions={
        onRequest ? (
          <Button onPress={onRequest}>Request a server</Button>
        ) : undefined
      }
    >
      <DialogSection className="gap-4">
        {servers.length > 0 ? (
          <SearchField
            label="Search servers"
            placeholder="Search servers"
            value={query}
            onChange={setQuery}
          />
        ) : null}
        {servers.length === 0 ? (
          <EmptyMessage>
            There is nothing left to connect. Every server here either works as
            it is or is already connected.
          </EmptyMessage>
        ) : shown.length === 0 ? (
          <EmptyMessage>No server matches "{query.trim()}".</EmptyMessage>
        ) : (
          // `otari-actions` is the place that drops a ghost button's edge.
          <ul
            aria-label="Servers you can connect"
            className="otari-actions grid grid-cols-2 gap-2 sm:grid-cols-3"
          >
            {shown.map((server) => (
              <li key={server.id} className="min-w-0">
                <Button
                  onPress={() => onPick(server)}
                  isPending={pendingId === server.id}
                  // Starting another would drop this one's result.
                  isDisabled={pendingId !== "" && pendingId !== server.id}
                  aria-label={`Connect ${server.name}`}
                  fullWidth
                  className="h-full min-h-28 flex-col justify-start gap-2 px-3 py-4 whitespace-normal"
                >
                  <McpServerLogo
                    name={server.name}
                    logoUrl={logos.get(server.id)?.logo_url}
                    tone={logos.get(server.id)?.logo_tone}
                    size="lg"
                  />
                  <span className="max-w-full truncate text-sm font-medium">
                    {server.name}
                  </span>
                  {server.purpose_hint ? (
                    <span className="line-clamp-2 max-w-full text-caption break-words">
                      {server.purpose_hint}
                    </span>
                  ) : null}
                </Button>
              </li>
            ))}
          </ul>
        )}
      </DialogSection>
    </Dialog>
  )
}
