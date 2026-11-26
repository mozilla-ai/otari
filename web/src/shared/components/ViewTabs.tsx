import { Link } from "@tanstack/react-router"
import { useEffect, useRef } from "react"
import type { DashboardSearch } from "@/shared/helpers/search"

interface ViewTab {
  to: string
  label: string
  /** What `current` names this tab by, where tabs share a path. */
  id?: string
  /** The query string the tab opens on, when it carries filters across. */
  search?: DashboardSearch
}

/**
 * Views of one page as links under its title, each its own route: Rules and
 * Routing, Sessions and Requests.
 */
export function ViewTabs({
  label,
  tabs,
  current,
}: {
  label: string
  tabs: readonly ViewTab[]
  current: string
}) {
  const rowRef = useRef<HTMLDivElement>(null)
  // The row scrolls rather than wrapping, so on a phone the current tab can
  // sit past its edge. Brought in by the row's own scroll, as `TabRow` does:
  // `scrollIntoView` would scroll the page to the row as well.
  useEffect(() => {
    const row = rowRef.current
    const tab = Array.from(
      row?.querySelectorAll<HTMLElement>("[data-tab]") ?? [],
    ).find((one) => one.dataset.tab === current)
    if (!row || !tab) return
    const tabBox = tab.getBoundingClientRect()
    const rowBox = row.getBoundingClientRect()
    if (tabBox.left < rowBox.left) row.scrollLeft -= rowBox.left - tabBox.left
    else if (tabBox.right > rowBox.right)
      row.scrollLeft += tabBox.right - rowBox.right
  }, [current])
  return (
    <nav aria-label={label}>
      <div
        ref={rowRef}
        className="flex max-w-full items-center gap-6 overflow-x-auto border-b border-border"
      >
        {tabs.map((tab) => {
          const id = tab.id ?? tab.to
          const isActive = id === current
          return (
            <Link
              key={id}
              to={tab.to}
              search={tab.search}
              // Tabs on one path differ only by their query, so the router
              // has to read it to mark one current rather than all of them.
              activeOptions={{
                exact: true,
                includeSearch: tab.id !== undefined,
              }}
              aria-current={isActive ? "page" : undefined}
              data-tab={id}
              className={`-mb-px flex min-h-11 shrink-0 items-end border-b-2 pb-2 text-title whitespace-nowrap transition-colors motion-reduce:transition-none max-md:min-w-11 max-md:justify-center md:min-h-0 ${
                isActive
                  ? "border-foreground text-foreground"
                  : "border-transparent text-muted hover:text-foreground"
              }`}
            >
              {tab.label}
            </Link>
          )
        })}
      </div>
    </nav>
  )
}
