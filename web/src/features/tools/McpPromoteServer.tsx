import { useNavigate } from "@tanstack/react-router"
import { useState } from "react"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import {
  SettingsBlock,
  SettingsGroup,
} from "@/design-system/layout/SettingsGroup"
import { McpFoldChoices } from "@/features/tools/McpFoldChoices"
import { defaultFoldIds } from "@/features/tools/mcpFold"
import {
  useMcpServerCopies,
  usePromoteWorkspaceMcpServer,
} from "@/shared/api/organizationMcpServers"

const PROMOTE_LABEL = "Make it an organization server"

/**
 * The confirm step: the other workspaces' own copies of the server, offered
 * to fold into it, then the promote. Mounted only while open, so each opening
 * reads the copies afresh and starts from their default ticks.
 */
function PromoteDialog({
  workspaceId,
  server,
  onClose,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
  onClose: () => void
}) {
  const navigate = useNavigate()
  const copies = useMcpServerCopies(
    { url: server.url, serverId: server.id },
    true,
  )
  const promote = usePromoteWorkspaceMcpServer()
  // Undefined until a box is touched, so the default ticks follow the read.
  const [touched, setTouched] = useState<string[]>()
  const candidates = copies.data ?? []
  const selected = touched ?? defaultFoldIds(candidates)

  return (
    <Dialog
      isOpen
      onOpenChange={(isOpen) => {
        if (!isOpen) onClose()
      }}
      isDismissable={!promote.isPending}
      size="lg"
      title={`Make ${server.name} an organization server`}
      description="Every workspace in your organization can use it. It keeps its id, its settings and everyone's sign-in, and this workspace's access rules stay its own."
      actions={
        <>
          <Button isDisabled={promote.isPending} onPress={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            isDisabled={copies.data === undefined}
            isPending={promote.isPending}
            onPress={() =>
              promote.mutate(
                {
                  workspaceId,
                  serverId: server.id,
                  body: { fold_server_ids: selected },
                },
                {
                  onSuccess: () =>
                    void navigate({
                      to: "/organization/mcp-servers/$serverId",
                      params: { serverId: server.id },
                    }),
                },
              )
            }
          >
            {PROMOTE_LABEL}
          </Button>
        </>
      }
    >
      <DialogSection>
        <ErrorBanner error={promote.error ?? copies.error} />
        {copies.isPending ? (
          <EmptyMessage>
            Looking for copies of it in other workspaces.
          </EmptyMessage>
        ) : candidates.length > 0 ? (
          <McpFoldChoices
            candidates={candidates}
            selected={selected}
            isDisabled={promote.isPending}
            onChange={setTouched}
          />
        ) : copies.data ? (
          <p className="text-sm text-muted">
            No other workspace has its own server at this URL.
          </p>
        ) : null}
      </DialogSection>
    </Dialog>
  )
}

/**
 * Offers an organization owner or admin to make a workspace's own server its
 * organization's, in place.
 */
export function McpPromoteServer({
  workspaceId,
  server,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
}) {
  const [isOpen, setOpen] = useState(false)
  return (
    <>
      <SettingsGroup title="Organization" isBounded>
        <SettingsBlock>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-body">
              Share this server with every workspace in your organization. It
              keeps its id, its settings and everyone&rsquo;s sign-in.
            </p>
            <Button
              className="min-h-11 shrink-0 md:min-h-9"
              onPress={() => setOpen(true)}
            >
              {PROMOTE_LABEL}
            </Button>
          </div>
        </SettingsBlock>
      </SettingsGroup>
      {isOpen ? (
        <PromoteDialog
          workspaceId={workspaceId}
          server={server}
          onClose={() => setOpen(false)}
        />
      ) : null}
    </>
  )
}
