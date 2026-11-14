import { type ReactNode, useState } from "react"

import { Button } from "@/design-system/actions/Button"
import { TablePagination } from "@/design-system/data/TablePagination"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SearchField } from "@/design-system/forms/SearchField"
import { Chip } from "@/design-system/indicators/Chip"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import { McpServerLogo } from "@/features/tools/McpServerLogo"
import {
  type CatalogPick,
  hostOf,
  toCatalogPick,
} from "@/features/tools/mcpCatalog"
import {
  useMcpCatalogPopular,
  useMcpCatalogServers,
} from "@/shared/api/mcpCatalog"
import { formatCount } from "@/shared/helpers/format"
import { useDebounced } from "@/shared/hooks/useDebounced"

// The gateway's own bound on `q`.
const MAX_QUERY_LENGTH = 200

/**
 * "Official" on a server its maker runs on its own domain, so the real one
 * stands out among the gateways named after it. The gateway judges that from
 * names, so the tooltip gives the host for the reader to judge too. Nothing on
 * the others.
 */
export function OfficialChip({ pick }: { pick: CatalogPick }) {
  const said = `Runs on ${hostOf(pick)}, its publisher's own domain`
  return (
    <Tooltip content={said}>
      <Chip tone="success">
        Official<span className="sr-only">. {said}</span>
      </Chip>
    </Tooltip>
  )
}

function CatalogRow({
  pick,
  control,
  controlAt,
}: {
  pick: CatalogPick
  /** The row's one control. */
  control: ReactNode
  controlAt: "start" | "end"
}) {
  return (
    <li className="flex items-start gap-3 border-border border-t py-3 first:border-t-0">
      {controlAt === "start" ? <div className="shrink-0">{control}</div> : null}
      <McpServerLogo
        name={pick.title}
        logoUrl={pick.logoUrl}
        tone={pick.logoTone}
      />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-medium break-words">{pick.title}</span>
          {pick.isOfficial ? <OfficialChip pick={pick} /> : null}
          {pick.requiresToken ? <Chip tone="info">Needs a token</Chip> : null}
        </div>
        <code className="font-mono text-caption break-all">{pick.url}</code>
        {pick.description ? (
          <p className="line-clamp-2 text-caption">{pick.description}</p>
        ) : null}
      </div>
      {controlAt === "end" ? <div className="shrink-0">{control}</div> : null}
    </li>
  )
}

function CatalogList({
  label,
  picks,
  renderControl,
  controlAt,
}: {
  label: string
  picks: CatalogPick[]
  renderControl: (pick: CatalogPick) => ReactNode
  controlAt: "start" | "end"
}) {
  return (
    <ul aria-label={label}>
      {picks.map((pick) => (
        <CatalogRow
          key={pick.name}
          pick={pick}
          control={renderControl(pick)}
          controlAt={controlAt}
        />
      ))}
    </ul>
  )
}

/** The way out for a server the catalog does not list: the form that takes a URL. */
function AddByUrl({ onPress }: { onPress: () => void }) {
  return (
    <p className="flex flex-wrap items-center gap-2 text-caption">
      Not listed?
      <Button size="sm" onPress={onPress}>
        Add it by URL
      </Button>
    </p>
  )
}

/**
 * The catalog as something to browse: a short Popular shelf before anything is
 * typed, and the whole catalog, searched and paged by the gateway, once
 * something is. The whole catalog unsearched is tens of thousands of servers in
 * alphabetical order, which nobody can browse, so it is never listed.
 *
 * What is picked sits in a Selected list above both, so a pick made in a
 * search stays on screen once the search is cleared, and is listed once.
 */
export function McpCatalogBrowser({
  workspaceId,
  serverCount,
  isSyncing,
  selected,
  isAdded,
  renderControl,
  controlAt = "start",
  emptyPopular = "No other well-known servers to add. Search for more.",
  onAddByUrl,
}: {
  /** The workspace being added to; the gateway leaves out the servers it has. */
  workspaceId: string
  /** Servers in the catalog now, for the line that invites a search. */
  serverCount: number
  /** A sync is storing pages, so the lists are re-read as they grow. */
  isSyncing: boolean
  selected: CatalogPick[]
  /** The workspace has this server, as of a read newer than the catalog's. */
  isAdded: (pick: CatalogPick) => boolean
  renderControl: (pick: CatalogPick) => ReactNode
  /** Where a row's control sits: a checkbox leads, a button trails. */
  controlAt?: "start" | "end"
  /** What the Popular shelf says once it has nothing left to list. */
  emptyPopular?: string
  onAddByUrl?: () => void
}) {
  const [typed, setTyped] = useState("")
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(25)
  const query = useDebounced(typed.trim())
  const isSearching = query !== ""
  const popular = useMcpCatalogPopular(workspaceId, isSyncing)
  const results = useMcpCatalogServers({
    query,
    page,
    pageSize,
    workspaceId,
    enabled: isSearching,
    isLoading: isSyncing,
  })

  const picked = new Set(selected.map((pick) => pick.name))
  const isListed = (pick: CatalogPick) =>
    !picked.has(pick.name) && !isAdded(pick)
  const allPopular = (popular.data?.data ?? []).map(toCatalogPick)
  const popularPicks = allPopular.filter(isListed)
  const allRows = (results.data?.data ?? []).map(toCatalogPick)
  const rows = allRows.filter(isListed)

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

      {selected.length > 0 ? (
        <section className="flex flex-col gap-1">
          <h3 className="text-title">Selected</h3>
          <CatalogList
            label="Selected"
            picks={selected}
            renderControl={renderControl}
            controlAt={controlAt}
          />
        </section>
      ) : null}

      {isSearching ? (
        <section className="flex flex-col gap-1">
          <h3 className="text-title">Results</h3>
          <ErrorBanner error={results.error} />
          {results.isPending && !results.data ? (
            <EmptyMessage>Searching the catalog.</EmptyMessage>
          ) : (
            <>
              {rows.length === 0 ? (
                <EmptyMessage>
                  {allRows.length > 0
                    ? "Everything on this page is selected."
                    : page === 0
                      ? `No server matches "${query}".`
                      : "Nothing on this page. The catalog changed since it was opened."}
                </EmptyMessage>
              ) : (
                <CatalogList
                  label="Results"
                  picks={rows}
                  renderControl={renderControl}
                  controlAt={controlAt}
                />
              )}
              {/* Kept on an empty page past the first, so there is a way back
                  when a sync shortened the catalog under the reader. */}
              {allRows.length > 0 || page > 0 ? (
                <TablePagination
                  label="catalog servers"
                  page={page}
                  pageSize={pageSize}
                  total={results.data?.count ?? null}
                  rowsOnPage={rows.length}
                  isFetching={results.isFetching}
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
      ) : (
        <section className="flex flex-col gap-1">
          <h3 className="text-title">Popular</h3>
          <ErrorBanner error={popular.error} />
          {popularPicks.length > 0 ? (
            <CatalogList
              label="Popular"
              picks={popularPicks}
              renderControl={renderControl}
              controlAt={controlAt}
            />
          ) : selected.length > 0 && allPopular.length > 0 ? null : (
            <EmptyMessage>
              {popular.isPending ? "Reading the catalog." : emptyPopular}
            </EmptyMessage>
          )}
          {serverCount > 0 ? (
            <p className="text-caption">
              {`Search to find any of the ${formatCount(serverCount)} servers in the catalog.`}
            </p>
          ) : null}
        </section>
      )}

      {onAddByUrl ? <AddByUrl onPress={onAddByUrl} /> : null}
    </div>
  )
}
