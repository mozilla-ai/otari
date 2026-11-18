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
import { isUnconfirmed } from "@/features/tools/mcpCatalog"
import {
  defaultFoldIds,
  foldableIds,
  MAX_FOLD_COPIES,
} from "@/features/tools/mcpFold"
import {
  useMcpServerCopies,
  useOrganizationMcpServer,
  usePromoteWorkspaceMcpServer,
  useRefreshAfterOrganizationServer,
} from "@/shared/api/organizationMcpServers"

const PROMOTE_LABEL = "Make it an organization server"

/**
 * The confirm step: the other workspaces' own copies of the server, offered
 * to fold into it, then the promote. Mounted only while open, so each opening
 * reads the copies afresh and starts from the default ticks of that read.
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
  const copies = useMcpServerCopies(server.id, true)
  // A copy read before this opening may have changed since, so only a read
  // that answered after it counts. Counted by the query rather than by the
  // clock, which ties with a read cached in the same millisecond.
  const isFresh = copies.isSuccess && copies.isFetchedAfterMount
  const promote = usePromoteWorkspaceMcpServer()
  const refreshAfterUnconfirmed = useRefreshAfterOrganizationServer()
  // No answer came back, so whether it was promoted is read from the
  // organization's servers, by the id it keeps either way.
  const isUnanswered = promote.isError && isUnconfirmed(promote.error)
  const promoted = useOrganizationMcpServer(server.id, isUnanswered)
  const wasPromoted = isUnanswered && promoted.isSuccess
  const candidates = isFresh ? copies.data : []
  // The ticked copies: the defaults of the first read this opening answered,
  // then the admin's. A copy a later read adds was never reviewed, so it starts
  // unticked.
  const [touched, setTouched] = useState<McpServerFoldCandidate[]>()
  if (isFresh && touched === undefined) {
    const defaults = defaultFoldIds(candidates, server.enabled)
    setTouched(
      candidates.filter((candidate) => defaults.includes(candidate.server_id)),
    )
  }
  const present = new Set(candidates.map((candidate) => candidate.server_id))
  // A ticked copy another admin removed since: no longer offered or sent.
  const gone = isFresh
    ? (touched ?? []).filter((copy) => !present.has(copy.server_id))
    : []
  const selected = foldableIds(
    candidates,
    (touched ?? []).map((copy) => copy.server_id),
    server.enabled,
  )
  const isOverLimit = selected.length > MAX_FOLD_COPIES
  const openPromoted = () =>
    void navigate({
      to: "/organization/mcp-servers/$serverId",
      params: { serverId: server.id },
    })

  return (
    <Dialog
      isOpen
      onOpenChange={(isOpen) => {
        if (!isOpen) onClose()
      }}
      isDismissable={!promote.isPending}
      size="lg"
      title={`Make ${server.name} an organization server`}
      description={`${
        server.enabled
          ? "Every workspace in your organization can use it."
          : "It stays off, so no workspace can use it until you turn it on."
      } It keeps its id, its settings and everyone's sign-in, and this workspace's access rules stay its own.`}
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
          {wasPromoted ? (
            <Button variant="primary" onPress={openPromoted}>
              Open it
            </Button>
          ) : (
            <Button
              variant="primary"
              // Held while a read is out too: a copy it drops is refused if sent.
              isDisabled={
                !isFresh ||
                copies.isFetching ||
                isOverLimit ||
                (isUnanswered && !promoted.isError)
              }
              isPending={promote.isPending}
              onPress={() =>
                promote.mutate(
                  {
                    workspaceId,
                    serverId: server.id,
                    body: { fold_server_ids: selected },
                  },
                  {
                    onSuccess: openPromoted,
                    onError: (error) => {
                      if (isUnconfirmed(error)) refreshAfterUnconfirmed()
                    },
                  },
                )
              }
            >
              {PROMOTE_LABEL}
            </Button>
          )}
        </>
      }
    >
      <DialogSection>
        {wasPromoted ? (
          <InfoBanner>
            It was made an organization server, though the answer was lost on
            the way.
          </InfoBanner>
        ) : (
          <ErrorBanner error={promote.error ?? copies.error} />
        )}
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
            targetEnabled={server.enabled}
            targetMode={server.credential_mode}
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
              keeps its id, its settings and everyone&rsquo;s sign-in
              {server.enabled ? "." : ", and stays off until you turn it on."}
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
