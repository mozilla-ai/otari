import { type ReactNode, useState } from "react"

import type { McpCatalogFeatured } from "@/client"
import { TablePagination } from "@/design-system/data/TablePagination"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SearchField } from "@/design-system/forms/SearchField"
import { Chip } from "@/design-system/indicators/Chip"
import { type CatalogPick, pickOf } from "@/features/tools/mcpCatalog"
import { useMcpCatalogServers } from "@/shared/api/mcpCatalog"
import { useDebounced } from "@/shared/hooks/useDebounced"

// The gateway's own bound on `q`.
const MAX_QUERY_LENGTH = 200

function CatalogRow({
  pick,
  control,
  note,
}: {
  pick: CatalogPick
  /** The row's one control, at its leading edge: a checkbox, or an action. */
  control: ReactNode
  /** A chip saying something about this row in the caller's context. */
  note?: ReactNode
}) {
  return (
    <li className="flex items-start gap-3 border-border border-t py-3 first:border-t-0">
      <div className="shrink-0">{control}</div>
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-medium break-words">{pick.title}</span>
          {pick.requiresToken ? <Chip tone="info">Needs a token</Chip> : null}
          {note}
        </div>
        <code className="font-mono text-caption break-all">{pick.url}</code>
        {pick.description ? (
          <p className="line-clamp-2 text-caption">{pick.description}</p>
        ) : null}
      </div>
    </li>
  )
}

/**
 * The catalog as a searchable list: the organization's featured servers first,
 * then every server, a page at a time. Used where servers are picked for a
 * workspace and where an admin chooses what to feature, so what each row's
 * control does is the caller's.
 *
 * The gateway filters and pages. Nothing here holds more than one page.
 */
export function McpCatalogBrowser({
  featured,
  renderControl,
  renderNote,
  emptyCatalog,
}: {
  /** Shown above the catalog while nothing is being searched for. */
  featured: McpCatalogFeatured[]
  renderControl: (pick: CatalogPick) => ReactNode
  renderNote?: (pick: CatalogPick) => ReactNode
  /** What to say when the catalog holds no server at all. */
  emptyCatalog: ReactNode
}) {
  const [typed, setTyped] = useState("")
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(25)
  const query = useDebounced(typed.trim())
  const servers = useMcpCatalogServers({ query, page, pageSize })

  const rows = servers.data?.data ?? []
  const isSearching = query !== ""

  return (
    <div className="flex flex-col gap-4">
      <SearchField
        label="Search the catalog"
        placeholder="Search by name or description"
        value={typed}
        maxLength={MAX_QUERY_LENGTH}
        onChange={(value) => {
          setTyped(value)
          setPage(0)
        }}
      />

      <ErrorBanner error={servers.error} />

      {!isSearching && featured.length > 0 ? (
        <section className="flex flex-col gap-1">
          <h3 className="text-title">Recommended by your organization</h3>
          <ul aria-label="Recommended by your organization">
            {featured.map((entry) => {
              const pick = pickOf(entry)
              return (
                <CatalogRow
                  key={entry.id}
                  pick={pick}
                  control={renderControl(pick)}
                  note={renderNote?.(pick)}
                />
              )
            })}
          </ul>
        </section>
      ) : null}

      <section className="flex flex-col gap-1">
        <h3 className="text-title">
          {isSearching ? "Results" : "All servers"}
        </h3>
        {servers.isPending && !servers.data ? (
          <EmptyMessage>Reading the catalog.</EmptyMessage>
        ) : (
          <>
            {rows.length === 0 ? (
              <EmptyMessage>
                {isSearching
                  ? `No server matches "${query}".`
                  : page === 0
                    ? emptyCatalog
                    : "Nothing on this page. The catalog changed since it was opened."}
              </EmptyMessage>
            ) : (
              <ul aria-label={isSearching ? "Results" : "All servers"}>
                {rows.map((server) => {
                  const pick = pickOf(server)
                  return (
                    <CatalogRow
                      key={server.id}
                      pick={pick}
                      control={renderControl(pick)}
                      note={renderNote?.(pick)}
                    />
                  )
                })}
              </ul>
            )}
            {/* Kept on an empty page past the first, so there is a way back
                when a sync shortened the catalog under the reader. */}
            {rows.length > 0 || page > 0 ? (
              <TablePagination
                label="catalog servers"
                page={page}
                pageSize={pageSize}
                total={servers.data?.count ?? null}
                rowsOnPage={rows.length}
                isFetching={servers.isFetching}
                onPageChange={setPage}
                onPageSizeChange={(size) => {
                  setPageSize(size)
                  setPage(0)
                }}
              />
            ) : null}
          </>
        )}
      </section>
    </div>
  )
}
