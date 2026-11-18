import { useState } from "react"
import type { ApiKey, EndUserCount } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { TablePagination } from "@/design-system/data/TablePagination"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Skeleton } from "@/design-system/feedback/Skeleton"
import { SearchField } from "@/design-system/forms/SearchField"
import { Badge } from "@/design-system/indicators/Badge"
import { budgetLabeler } from "@/features/budgets/budgetLabel"
import { useKeyEndUserCount, useKeyEndUsers } from "@/shared/api/apiKeys"
import { formatDateTimeMinutes, formatSpend } from "@/shared/helpers/format"
import { useDebounced } from "@/shared/hooks/useDebounced"
import { useEndUserBudgets } from "./ServiceKeyFields"

// The gateway's cap on an end user's id.
const EXTERNAL_ID_MAX_LENGTH = 256

/**
 * The end users a service key's requests have created, oldest first, with the
 * budget each is on and what it has used there.
 *
 * Budget names are shown where the caller can read the budgets: an operator,
 * or an owner or admin on an organization's service key. Anyone else sees ids.
 */
export function EndUsersDialog({
  apiKey,
  canReadBudgets,
  onClose,
}: {
  apiKey: ApiKey
  canReadBudgets: boolean
  onClose: () => void
}) {
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(25)
  const [query, setQuery] = useState("")
  // Untrimmed: an id is any string, spaces included.
  const externalId = useDebounced(query)
  const endUsers = useKeyEndUsers(apiKey.id, page, pageSize, externalId)
  const count = useKeyEndUserCount(apiKey.id)
  const budgets = useEndUserBudgets(
    apiKey.is_organization_service_key ? "organization" : "deployment",
    canReadBudgets,
  )
  const label = budgetLabeler(budgets.offered)
  const budgetName = (budgetId: string | null) => {
    if (budgetId === null) return "No budget"
    const budget = budgets.offered.find((row) => row.budget_id === budgetId)
    return budget ? label(budget) : budgetId
  }
  const rows = endUsers.data?.rows ?? []
  const keyLabel = apiKey.key_name ?? apiKey.id

  return (
    <Dialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title="End users"
      description={
        <span className="inline-flex flex-wrap items-center gap-2">
          <code>{keyLabel}</code>
          {count.data ? (
            <EndUserCountLine count={count.data} />
          ) : count.isError ? (
            <>
              <span className="text-caption">Could not read the cap</span>
              <Button size="sm" onPress={() => count.refetch()}>
                Retry
              </Button>
            </>
          ) : null}
        </span>
      }
      size="lg"
      actions={<Button onPress={onClose}>Close</Button>}
    >
      <DialogSection>
        <SearchField
          label="Find an end user by id"
          placeholder="End user id"
          value={query}
          maxLength={EXTERNAL_ID_MAX_LENGTH}
          onChange={(value) => {
            setQuery(value)
            setPage(0)
          }}
        />
        {endUsers.error ? (
          <ErrorBanner error={endUsers.error} />
        ) : endUsers.isPending ? (
          <Skeleton className="h-16 w-full" />
        ) : rows.length === 0 ? (
          <EmptyMessage>
            {externalId ? "No end user has this id." : "No end users yet."}
          </EmptyMessage>
        ) : (
          <ul
            aria-label={`End users of ${keyLabel}`}
            className="flex flex-col divide-y divide-border border-y border-border"
          >
            {rows.map((endUser) => (
              <li key={endUser.user_id} className="flex flex-col gap-1 py-3">
                <div className="flex items-start gap-2">
                  <p className="min-w-0 flex-1 break-all text-body">
                    {endUser.external_id}
                  </p>
                  {endUser.blocked ? (
                    <span className="shrink-0">
                      <Badge tone="warn">Blocked</Badge>
                    </span>
                  ) : null}
                </div>
                <p className="break-words text-caption">
                  {budgetName(endUser.budget_id)} · {formatSpend(endUser.spend)}{" "}
                  · {endUser.current_requests.toLocaleString("en-US")}{" "}
                  {endUser.current_requests === 1 ? "request" : "requests"} ·
                  since {formatDateTimeMinutes(endUser.created_at)}
                </p>
              </li>
            ))}
          </ul>
        )}
        {externalId ? null : (
          <TablePagination
            page={page}
            pageSize={pageSize}
            total={null}
            rowsOnPage={rows.length}
            hasNextFallback={endUsers.data?.hasMore ?? false}
            onPageChange={setPage}
            onPageSizeChange={(size) => {
              setPageSize(size)
              setPage(0)
            }}
            isFetching={endUsers.isFetching}
            label={`end users of ${keyLabel}`}
          />
        )}
      </DialogSection>
    </Dialog>
  )
}

/**
 * The end users the key's cap counts, against the cap. Those on budgets the key
 * does not assign are listed below but take no place, so the two can differ.
 */
function EndUserCountLine({ count }: { count: EndUserCount }) {
  const used = count.count.toLocaleString("en-US")
  if (count.max_end_users === null) {
    return (
      <span className="text-caption">
        {used} {count.count === 1 ? "end user" : "end users"}, no cap
      </span>
    )
  }
  const isFull = count.count >= count.max_end_users
  return (
    <>
      <span className="text-caption">
        {used} of {count.max_end_users.toLocaleString("en-US")} end users
      </span>
      {isFull ? <Badge tone="warn">Full</Badge> : null}
    </>
  )
}
