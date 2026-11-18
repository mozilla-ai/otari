import { useNavigate } from "@tanstack/react-router"
import { useState } from "react"

import type { McpServerFoldCandidate, WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import {
  SettingsBlock,
  SettingsGroup,
} from "@/design-system/layout/SettingsGroup"
import { McpFoldChoices } from "@/features/tools/McpFoldChoices"
import { defaultFoldIds, MAX_FOLD_COPIES } from "@/features/tools/mcpFold"
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
  // A copy read before this opening may have changed since, so only a read
  // that answered after it counts.
  const [openedAt] = useState(Date.now)
  const copies = useMcpServerCopies({ serverId: server.id }, true)
  const isFresh = copies.isSuccess && copies.dataUpdatedAt >= openedAt
  const promote = usePromoteWorkspaceMcpServer()
  // The copies ticked by hand, undefined until a box is touched so the default
  // ticks follow the read.
  const [touched, setTouched] = useState<McpServerFoldCandidate[]>()
  const candidates = isFresh ? copies.data : []
  const present = new Set(candidates.map((candidate) => candidate.server_id))
  // A ticked copy another admin removed since: no longer offered or sent.
  const gone = isFresh
    ? (touched ?? []).filter((copy) => !present.has(copy.server_id))
    : []
  const selected = touched
    ? touched
        .map((copy) => copy.server_id)
        .filter((serverId) => present.has(serverId))
    : defaultFoldIds(candidates)
  const isOverLimit = selected.length > MAX_FOLD_COPIES

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
          {copies.isError ? (
            <Button
              isPending={copies.isFetching}
              onPress={() => void copies.refetch()}
            >
              Retry
            </Button>
          ) : null}
          <Button
            variant="primary"
            isDisabled={!isFresh || isOverLimit}
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
        {gone.length > 0 ? (
          <InfoBanner tone="warning">
            {`No longer there, so no longer folded: ${gone
              .map((copy) => `${copy.workspace_name}: ${copy.name}`)
              .join(", ")}.`}
          </InfoBanner>
        ) : null}
        {!isFresh ? (
          copies.isError ? null : (
            <EmptyMessage>
              Looking for copies of it in other workspaces.
            </EmptyMessage>
          )
        ) : candidates.length > 0 ? (
          <McpFoldChoices
            candidates={candidates}
            selected={selected}
            isDisabled={promote.isPending}
            onChange={(ids) =>
              setTouched(
                candidates.filter((candidate) =>
                  ids.includes(candidate.server_id),
                ),
              )
            }
          />
        ) : (
          <p className="text-sm text-muted">
            No other workspace has its own server at this URL.
          </p>
        )}
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
