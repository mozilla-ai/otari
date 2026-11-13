import { useState } from "react"

import type { McpServerRequest, WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { TextArea } from "@/design-system/forms/TextArea"
import { Chip } from "@/design-system/indicators/Chip"
import { McpCatalogBrowser } from "@/features/tools/McpCatalogBrowser"
import { McpServerLogo } from "@/features/tools/McpServerLogo"
import {
  type CatalogPick,
  normalizeEndpoint,
} from "@/features/tools/mcpCatalog"
import { isMcpCatalogSyncing, useMcpCatalogSync } from "@/shared/api/mcpCatalog"
import {
  useAcknowledgeMcpServerRequest,
  useMyMcpServerRequests,
  useRequestMcpServer,
  useWithdrawMcpServerRequest,
} from "@/shared/api/mcpServerRequests"
import { formatDate } from "@/shared/helpers/format"

// The gateway's own bound on a request's note.
const MAX_NOTE_LENGTH = 500

/**
 * A member's way to a server the workspace does not have: browse the catalog
 * and ask the admins for one. Asking adds nothing; an admin adds the server or
 * declines. The member's own waiting and declined requests head the list: a
 * waiting one can be withdrawn there, and a declined one stays until dismissed.
 * A request that was approved is not listed, since its server is on the page.
 */
export function McpRequestServerDialog({
  isOpen,
  onOpenChange,
  workspaceId,
  workspaceName,
  existing,
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  workspaceId: string
  workspaceName: string
  /** The workspace's servers, which cannot be asked for again. */
  existing: readonly WorkspaceMcpServer[]
}) {
  const sync = useMcpCatalogSync(isOpen)
  const mine = useMyMcpServerRequests(workspaceId, isOpen)
  const request = useRequestMcpServer()
  const withdraw = useWithdrawMcpServerRequest()
  const acknowledge = useAcknowledgeMcpServerRequest()
  // The server a note is being written for; none while browsing.
  const [asking, setAsking] = useState<CatalogPick>()
  const [note, setNote] = useState("")

  const requests = (mine.data?.data ?? []).filter(
    (entry) => entry.status === "pending" || entry.status === "declined",
  )
  const waiting = new Map(
    requests
      .filter((entry) => entry.status === "pending")
      .map((entry) => [entry.catalog_name, entry]),
  )
  const added = new Set(existing.map((server) => normalizeEndpoint(server.url)))

  const startAsking = (pick: CatalogPick) => {
    request.reset()
    setNote("")
    setAsking(pick)
  }
  const send = () => {
    if (!asking || note.length > MAX_NOTE_LENGTH) return
    request.mutate(
      {
        workspaceId,
        body: {
          catalog_name: asking.name,
          ...(note.trim() ? { note: note.trim() } : {}),
        },
      },
      { onSuccess: () => setAsking(undefined) },
    )
  }
  const withdrawRequest = (entry: McpServerRequest) =>
    withdraw.mutate({ workspaceId, requestId: entry.id })
  const isWithdrawing = (entry: McpServerRequest) =>
    withdraw.isPending && withdraw.variables?.requestId === entry.id

  if (asking) {
    return (
      <Dialog
        isOpen={isOpen}
        onOpenChange={onOpenChange}
        isDismissable={!request.isPending}
        size="lg"
        title={`Request ${asking.title}`}
        description={`The admins of ${workspaceName} decide whether to add it.`}
        actions={
          <>
            <Button
              isDisabled={request.isPending}
              onPress={() => setAsking(undefined)}
            >
              Back
            </Button>
            <Button
              variant="primary"
              isPending={request.isPending}
              isDisabled={note.length > MAX_NOTE_LENGTH}
              onPress={send}
            >
              Send request
            </Button>
          </>
        }
      >
        <DialogSection className="gap-4">
          <ErrorBanner error={request.error} />
          <div className="flex items-center gap-3">
            <McpServerLogo
              name={asking.title}
              logoUrl={asking.logoUrl}
              tone={asking.logoTone}
            />
            <code className="min-w-0 font-mono text-caption break-all">
              {asking.url}
            </code>
          </div>
          <TextArea
            label="What do you need it for?"
            value={note}
            onChange={setNote}
            rows={3}
            placeholder="Optional"
            description="Shown to the admins with your name."
            isInvalid={note.length > MAX_NOTE_LENGTH}
            errorMessage={`Keep it under ${MAX_NOTE_LENGTH} characters.`}
          />
        </DialogSection>
      </Dialog>
    )
  }

  const status = sync.data
  const hasServers = (status?.server_count ?? 0) > 0
  return (
    <Dialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      size="lg"
      title="Request a server"
      description={`Ask the admins of ${workspaceName} to add a server from the catalog.`}
      footerStart="Listing a server here does not vouch for it."
      actions={<Button onPress={() => onOpenChange(false)}>Close</Button>}
    >
      {requests.length > 0 ? (
        <DialogSection>
          <h3 className="text-title">Your requests</h3>
          <ErrorBanner error={withdraw.error} />
          <ErrorBanner error={acknowledge.error} />
          <ul aria-label="Your requests" className="flex flex-col gap-3">
            {requests.map((entry) => (
              <li key={entry.id} className="flex items-start gap-3">
                <McpServerLogo
                  name={entry.title}
                  logoUrl={entry.catalog_server?.logo_url}
                  tone={entry.catalog_server?.logo_tone}
                  size="sm"
                />
                <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className="flex flex-wrap items-center gap-2">
                    <span className="font-medium break-words">
                      {entry.title}
                    </span>
                    {entry.status === "pending" ? (
                      <Chip tone="info">Waiting for an admin</Chip>
                    ) : (
                      <Chip tone="warning">Declined</Chip>
                    )}
                  </span>
                  <span className="text-caption break-words">
                    {entry.status === "pending"
                      ? `Asked on ${formatDate(entry.created_at)}`
                      : entry.decision_reason
                        ? `Declined on ${formatDate(entry.decided_at)}: ${entry.decision_reason}`
                        : `Declined on ${formatDate(entry.decided_at)}`}
                  </span>
                </div>
                {entry.status === "pending" ? (
                  <Button
                    size="sm"
                    onPress={() => withdrawRequest(entry)}
                    isPending={isWithdrawing(entry)}
                    aria-label={`Withdraw your request for ${entry.title}`}
                    className="min-h-11 shrink-0 md:min-h-0"
                  >
                    Withdraw
                  </Button>
                ) : (
                  <Button
                    size="sm"
                    onPress={() =>
                      acknowledge.mutate({ workspaceId, requestId: entry.id })
                    }
                    aria-label={`Dismiss your declined request for ${entry.title}`}
                    className="min-h-11 shrink-0 md:min-h-0"
                  >
                    Dismiss
                  </Button>
                )}
              </li>
            ))}
          </ul>
        </DialogSection>
      ) : null}
      <DialogSection>
        <ErrorBanner error={sync.error} />
        <ErrorBanner error={mine.error} />
        {status === undefined ? (
          sync.isPending ? (
            <EmptyMessage>Reading the catalog.</EmptyMessage>
          ) : null
        ) : hasServers ? (
          <McpCatalogBrowser
            workspaceId={workspaceId}
            serverCount={status.server_count}
            isSyncing={isMcpCatalogSyncing(status)}
            selected={[]}
            isAdded={(pick) => added.has(normalizeEndpoint(pick.url))}
            controlAt="end"
            emptyPopular="This workspace has every well-known server. Search for more."
            renderControl={(pick) => {
              const entry = waiting.get(pick.name)
              return entry ? (
                <Chip tone="info">Requested</Chip>
              ) : (
                <Button
                  size="sm"
                  onPress={() => startAsking(pick)}
                  // Nothing is offered until the member's own requests are read.
                  isDisabled={mine.data === undefined}
                  aria-label={`Request ${pick.title}`}
                  className="min-h-11 md:min-h-0"
                >
                  Request
                </Button>
              )
            }}
          />
        ) : (
          <EmptyMessage>
            The catalog is empty. Ask an admin to load it.
          </EmptyMessage>
        )}
      </DialogSection>
    </Dialog>
  )
}
