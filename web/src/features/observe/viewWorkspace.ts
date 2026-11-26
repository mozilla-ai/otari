import { useSearch } from "@tanstack/react-router"
import { useEffect, useRef } from "react"

import { useWorkspaces } from "@/shared/api/workspaces"
import type { DashboardSearch } from "@/shared/helpers/search"
import { useUrlState } from "@/shared/helpers/urlState"
import {
  useNarrowingWorkspace,
  useSelectedWorkspace,
} from "@/shared/hooks/SelectedWorkspace"

/**
 * The `workspace` a link into Sessions carries for every workspace the caller
 * may read. Explicit, because a link that names no workspace opens on the
 * shell's selection instead.
 */
export const ALL_WORKSPACES = "all"

const URL_DEFAULTS = {
  workspace: "",
  page: "0",
  peek: "",
  peek_workspace: "",
  span: "",
  request: "",
}

/** What a link carries so a Sessions view opens on its scope: an id, or "" for every workspace. */
export const workspaceSearch = (workspaceId: string) => ({
  workspace: workspaceId || ALL_WORKSPACES,
})

/**
 * The workspace the URL names for a Sessions view: an id, "" for every
 * workspace, or undefined where it names none and the view follows the shell.
 */
export function useCarriedWorkspace(): string | undefined {
  const raw = useSearch({
    strict: false,
    select: (search: DashboardSearch) => search.workspace,
  })
  const value = (Array.isArray(raw) ? raw[0] : raw) ?? ""
  if (!value) return undefined
  return value === ALL_WORKSPACES ? "" : value
}

/**
 * The workspace a Sessions view reads: the one its URL names, else the shell's
 * (`useNarrowingWorkspace`), with "" for every workspace and null while the
 * shell's selection is not final.
 */
export function useViewWorkspace(): string | null {
  const carried = useCarriedWorkspace()
  const shell = useNarrowingWorkspace()
  return carried ?? shell
}

interface ViewWorkspaceFilter {
  value: string
  options: { value: string; label: string }[]
  onChange: (next: string) => void
}

/**
 * The control that names the workspace a Sessions view shows when its URL
 * names one, since the shell's switcher may name another. Null where the view
 * follows the shell, or the caller sees one workspace. Kept where the list of
 * workspaces fails, so the scope the URL names stays on screen.
 *
 * Picking a workspace in the shell is a choice of scope too, the one already
 * selected included, so it drops the URL's and the view follows the shell again.
 */
export function useViewWorkspaceFilter(): ViewWorkspaceFilter | null {
  const carried = useCarriedWorkspace()
  const { patch } = useUrlState(URL_DEFAULTS)
  const isCarried = carried !== undefined
  const workspaces = useWorkspaces(isCarried)
  const { selections } = useSelectedWorkspace()
  const seen = useRef(selections)

  useEffect(() => {
    if (seen.current === selections) return
    seen.current = selections
    if (isCarried) patch({ workspace: "", page: 0 })
  }, [selections, isCarried, patch])

  const rows = workspaces.data ?? []
  if (!isCarried || (rows.length <= 1 && !workspaces.isError)) return null
  return {
    value: carried,
    options: [
      { value: "", label: "All workspaces" },
      ...rows.map((one) => ({ value: one.id, label: one.name })),
    ],
    // An open session or request belongs to the scope being left.
    onChange: (next) =>
      patch({
        workspace: next || ALL_WORKSPACES,
        page: 0,
        peek: "",
        peek_workspace: "",
        span: "",
        request: "",
      }),
  }
}
