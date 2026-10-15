import { SearchField as AriaSearchField, Input } from "react-aria-components"
import {
  FiChevronsDown,
  FiChevronsUp,
  FiSearch,
  FiSidebar,
} from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { Tooltip } from "@/design-system/overlays/Tooltip"

import { TraceViewSwitch } from "./TraceViewSwitch"

export type NavView = "tree" | "timeline"

// Panel toggle, span search, collapse all, Tree / Timeline, after Langfuse's
// trace panel navigation header (see LANGFUSE-LICENSE.md).
export function NavHeader({
  isCollapsed,
  onToggleCollapsed,
  query,
  onQueryChange,
  everythingCollapsed,
  onToggleAll,
  view,
  onViewChange,
}: {
  isCollapsed: boolean
  onToggleCollapsed: () => void
  query: string
  onQueryChange: (value: string) => void
  everythingCollapsed: boolean
  onToggleAll: () => void
  view: NavView
  onViewChange: (view: NavView) => void
}) {
  const panelLabel = isCollapsed ? "Expand panel" : "Collapse panel"
  const panelButton = (
    <Tooltip content={panelLabel}>
      <IconButton
        label={panelLabel}
        size="sm"
        isIconOnly
        onPress={onToggleCollapsed}
      >
        <FiSidebar aria-hidden="true" className="size-4" />
      </IconButton>
    </Tooltip>
  )

  if (isCollapsed) {
    return (
      <div className="otari-toolbar flex min-h-11 shrink-0 items-center justify-center">
        {panelButton}
      </div>
    )
  }

  const allLabel = everythingCollapsed ? "Expand all" : "Collapse all"
  return (
    <div className="otari-toolbar grid min-h-11 shrink-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-1 border-b border-border pr-2 pl-1">
      {/* Below md the tree is the whole width, with nothing to give room to. */}
      <span className="max-md:hidden">{panelButton}</span>
      {/* Borderless, to sit flush in the header. Escape clears the query. */}
      <AriaSearchField
        aria-label="Search spans"
        value={query}
        onChange={onQueryChange}
        className="flex min-w-0 items-center gap-1.5"
      >
        <FiSearch
          aria-hidden="true"
          className="size-3.5 shrink-0 text-subtle"
        />
        <Input
          placeholder="Search"
          className="h-7 min-w-0 flex-1 bg-transparent text-caption outline-none placeholder:text-subtle data-focus-visible:otari-focus-ring [&::-webkit-search-cancel-button]:appearance-none"
        />
      </AriaSearchField>
      <div className="flex items-center gap-0.5">
        <Tooltip content={allLabel}>
          <IconButton
            label={allLabel}
            size="sm"
            isIconOnly
            onPress={onToggleAll}
          >
            {everythingCollapsed ? (
              <FiChevronsDown aria-hidden="true" className="size-4" />
            ) : (
              <FiChevronsUp aria-hidden="true" className="size-4" />
            )}
          </IconButton>
        </Tooltip>
        <span className="ml-2">
          <TraceViewSwitch value={view} onChange={onViewChange} />
        </span>
      </div>
    </div>
  )
}
