import {
  FiChevronsDown,
  FiChevronsUp,
  FiSearch,
  FiSidebar,
} from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { Segmented } from "@/design-system/navigation/Segmented"
import { Tooltip } from "@/design-system/overlays/Tooltip"

export type NavView = "tree" | "timeline"

const NAV_OPTIONS = [
  { value: "tree", label: "Tree" },
  { value: "timeline", label: "Timeline" },
]

const BUTTON = "md:min-h-7 md:min-w-7"

// One row: panel toggle, search, collapse all, Tree / Timeline. After
// Langfuse's TracePanelNavigationHeader (MIT, Copyright (c) 2023-2026
// ClickHouse, Inc.).
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
        className={BUTTON}
        onPress={onToggleCollapsed}
      >
        <FiSidebar aria-hidden="true" className="size-4" />
      </IconButton>
    </Tooltip>
  )

  if (isCollapsed) {
    return (
      <div className="otari-toolbar flex h-11 shrink-0 items-center justify-center md:h-9">
        {panelButton}
      </div>
    )
  }

  const allLabel = everythingCollapsed ? "Expand all" : "Collapse all"
  return (
    <div className="otari-toolbar grid h-11 shrink-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-1 border-b border-border pr-2 pl-1 md:h-9">
      {panelButton}
      <label className="flex min-w-0 items-center gap-1.5">
        <FiSearch
          aria-hidden="true"
          className="size-3.5 shrink-0 text-subtle"
        />
        <input
          type="text"
          aria-label="Search"
          placeholder="Search"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          className="h-7 min-w-0 flex-1 bg-transparent text-caption outline-none placeholder:text-subtle focus-visible:outline-2 focus-visible:outline-foreground"
        />
      </label>
      <div className="flex items-center gap-0.5">
        <Tooltip content={allLabel}>
          <IconButton
            label={allLabel}
            size="sm"
            className={BUTTON}
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
          <Segmented
            size="sm"
            label="Trace view"
            options={NAV_OPTIONS}
            value={view}
            onChange={(next) => onViewChange(next as NavView)}
          />
        </span>
      </div>
    </div>
  )
}
