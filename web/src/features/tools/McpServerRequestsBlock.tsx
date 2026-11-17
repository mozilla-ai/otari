import { useState } from "react"

import type { McpServerRequester, McpServerRequestGroup } from "@/client"
import { Button } from "@/design-system/actions/Button"
import {
  PAGE_SIZE_OPTIONS,
  TablePagination,
} from "@/design-system/data/TablePagination"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { TextArea } from "@/design-system/forms/TextArea"
import {
  SettingsBlock,
  SettingsGroup,
} from "@/design-system/layout/SettingsGroup"
import { OfficialChip } from "@/features/tools/McpCatalogBrowser"
import { McpServerLogo } from "@/features/tools/McpServerLogo"
import { type CatalogPick, toCatalogPick } from "@/features/tools/mcpCatalog"
import {
  useDeclineMcpServerRequests,
  useMcpServerRequests,
} from "@/shared/api/mcpServerRequests"
import { formatDate } from "@/shared/helpers/format"

// The gateway's own bound on a decline reason.
const MAX_REASON_LENGTH = 500

/** A requester's name, or else their email, or else what is left to say. */
function nameOf(requester: McpServerRequester): string {
  return requester.full_name ?? requester.email ?? "A member"
}

/**
 * "Ada", "Ada and Grace", "Ada, Grace and 2 others". `total` counts everyone who
 * asked, the gateway listing only the first few of a server's requesters.
 */
export function whoAsked(
  requesters: readonly McpServerRequester[],
  total = requesters.length,
): string {
  const names = requesters.map(nameOf)
  if (total <= 2) return names.join(" and ")
  const others = total - 2
  return `${names[0]}, ${names[1]} and ${others} ${others === 1 ? "other" : "others"}`
}

/**
 * What the catalog flow needs to add the server a request names. From the
 * catalog's entry while it lists one; otherwise from the title and URL the
 * request kept, so a request still leads somewhere after a sync dropped it.
 */
export function pickFor(group: McpServerRequestGroup): CatalogPick {
  if (group.catalog_server) return toCatalogPick(group.catalog_server)
  return {
    name: group.catalog_name,
    title: group.title,
    description: "",
    url: group.url,
    requiresToken: false,
    logoUrl: "",
    logoTone: null,
    isOfficial: false,
  }
}

/**
 * The servers members asked for, above an admin's list, and absent while
 * nobody is waiting. Add opens the catalog's confirm step for that server;
 * storing it answers everyone who asked. Decline answers them with a reason.
 */
export function McpServerRequestsBlock({
  workspaceId,
  onAdd,
}: {
  workspaceId: string
  /** Opens the catalog's confirm step for the server a request names. */
  onAdd: (pick: CatalogPick) => void
}) {
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState<number>(PAGE_SIZE_OPTIONS[0])
  const requests = useMcpServerRequests(workspaceId, { page, pageSize }, true)
  const decline = useDeclineMcpServerRequests()
  const [declining, setDeclining] = useState<McpServerRequestGroup>()
  const [reason, setReason] = useState("")

  // A page counts servers, not requests. One emptied by an answer is left for
  // the last one that still has servers.
  const servers = requests.data?.server_count
  const lastPage =
    servers === undefined
      ? page
      : Math.max(0, Math.ceil(servers / pageSize) - 1)
  if (page > lastPage) setPage(lastPage)

  const groups = requests.data?.data ?? []
  if (groups.length === 0) {
    if (!requests.isError) return null
    // A page that failed with nothing kept for it has no pager to leave by.
    return (
      <div className="flex flex-col items-start gap-2">
        <ErrorBanner error={requests.error} />
        <div className="flex flex-wrap gap-2">
          <Button
            onPress={() => void requests.refetch()}
            isPending={requests.isFetching}
            className="min-h-11 md:min-h-0"
          >
            Retry
          </Button>
          {page > 0 ? (
            <Button
              onPress={() => setPage(page - 1)}
              className="min-h-11 md:min-h-0"
            >
              Previous page
            </Button>
          ) : null}
        </div>
      </div>
    )
  }

  const openDecline = (group: McpServerRequestGroup) => {
    decline.reset()
    setReason("")
    setDeclining(group)
  }

  return (
    <>
      <SettingsGroup
        title="Requests"
        count={requests.data?.count}
        description="Servers members asked for. Adding one tells everyone who asked."
        isBounded
      >
        {groups.map((group) => {
          const pick = pickFor(group)
          const notes = group.requesters.filter((requester) => requester.note)
          return (
            <SettingsBlock key={group.catalog_name}>
              <section
                aria-label={`Requests for ${group.title}`}
                className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between"
              >
                <div className="flex min-w-0 items-start gap-3">
                  <McpServerLogo
                    name={group.title}
                    logoUrl={pick.logoUrl}
                    tone={pick.logoTone}
                  />
                  <div className="flex min-w-0 flex-col gap-1">
                    <span className="flex flex-wrap items-center gap-2">
                      <span className="font-medium break-words">
                        {group.title}
                      </span>
                      {pick.isOfficial ? <OfficialChip pick={pick} /> : null}
                    </span>
                    <code className="font-mono text-caption break-all">
                      {group.url}
                    </code>
                    <span className="text-sm break-words">
                      {`Asked by ${whoAsked(group.requesters, group.requester_count)} on ${formatDate(group.first_requested_at)}`}
                    </span>
                    {notes.length > 0 ? (
                      <ul className="flex flex-col gap-1 text-caption">
                        {notes.map((requester) => (
                          <li
                            key={requester.request_id}
                            className="break-words"
                          >
                            <span className="font-medium">
                              {nameOf(requester)}:
                            </span>{" "}
                            {requester.note}
                          </li>
                        ))}
                      </ul>
                    ) : null}
                  </div>
                </div>
                <div className="flex shrink-0 flex-wrap gap-2">
                  <Button
                    variant="primary"
                    onPress={() => onAdd(pick)}
                    aria-label={`Add ${group.title}`}
                    className="min-h-11 md:min-h-0"
                  >
                    Add
                  </Button>
                  <Button
                    onPress={() => openDecline(group)}
                    aria-label={`Decline ${group.title}`}
                    className="min-h-11 md:min-h-0"
                  >
                    Decline
                  </Button>
                </div>
              </section>
            </SettingsBlock>
          )
        })}
        {servers !== undefined && servers > PAGE_SIZE_OPTIONS[0] ? (
          <SettingsBlock>
            <TablePagination
              page={page}
              pageSize={pageSize}
              total={servers}
              rowsOnPage={groups.length}
              onPageChange={setPage}
              onPageSizeChange={(size) => {
                setPageSize(size)
                setPage(0)
              }}
              isFetching={requests.isFetching}
            />
          </SettingsBlock>
        ) : null}
      </SettingsGroup>

      {declining ? (
        <FormDialog
          isOpen
          onOpenChange={(open) => {
            if (!open) setDeclining(undefined)
          }}
          title={`Decline ${declining.title}`}
          description={`${whoAsked(declining.requesters, declining.requester_count)} will see that the request was declined.`}
          submitLabel="Decline"
          isPending={decline.isPending}
          isSubmitDisabled={reason.length > MAX_REASON_LENGTH}
          error={decline.error}
          onSubmit={() => {
            decline.mutate(
              {
                workspaceId,
                body: {
                  catalog_name: declining.catalog_name,
                  ...(reason.trim() ? { reason: reason.trim() } : {}),
                },
              },
              { onSuccess: () => setDeclining(undefined) },
            )
          }}
        >
          <TextArea
            label="Reason"
            value={reason}
            onChange={setReason}
            rows={3}
            placeholder="Optional"
            description="Shown to everyone who asked."
            isInvalid={reason.length > MAX_REASON_LENGTH}
            errorMessage={`Keep it under ${MAX_REASON_LENGTH} characters.`}
          />
        </FormDialog>
      ) : null}
    </>
  )
}
