import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react"

import type { CallerWorkspaceMembership } from "@/client"
import { useOrganizationContext } from "@/shared/api/hooks"

const STORAGE_KEY = "otari.dashboard.selectedWorkspace"

interface SelectedWorkspace {
  /** The workspaces the caller belongs to, in the order the server returned. */
  memberships: readonly CallerWorkspaceMembership[]
  /** The selected workspace, or null before the context answers. */
  selected: CallerWorkspaceMembership | null
  select: (workspaceId: string) => void
  isLoading: boolean
}

const Context = createContext<SelectedWorkspace | null>(null)

function readStored(): string | null {
  if (typeof window === "undefined") return null
  try {
    return window.localStorage.getItem(STORAGE_KEY)
  } catch {
    // Private-mode Safari and a disabled-storage policy both throw here. A
    // remembered workspace is a convenience, so losing it is not worth failing
    // the shell over; the first membership is picked instead.
    return null
  }
}

/**
 * Which workspace the shell is looking at.
 *
 * Seeded from `workspace_memberships` on the organization context, so no extra
 * request is made to render the switcher. The selection is remembered per
 * browser, and falls back to the first membership when the stored id names a
 * workspace the caller is no longer in (removed from it, or the deployment was
 * rebuilt), which would otherwise leave the switcher pointing at nothing.
 */
export function SelectedWorkspaceProvider({
  children,
}: {
  children: ReactNode
}) {
  const context = useOrganizationContext()
  const memberships = useMemo(
    () => context.data?.workspace_memberships ?? [],
    [context.data],
  )
  const [chosen, setChosen] = useState<string | null>(readStored)

  const selected = useMemo(() => {
    if (memberships.length === 0) return null
    return memberships.find((m) => m.workspace_id === chosen) ?? memberships[0]
  }, [memberships, chosen])

  // Write the resolved id back, not the chosen one, so a stale stored id is
  // repaired rather than kept and re-resolved on every load.
  useEffect(() => {
    if (!selected) return
    try {
      window.localStorage.setItem(STORAGE_KEY, selected.workspace_id)
    } catch {
      // See readStored: a browser that refuses storage still gets a working
      // switcher, it just forgets the choice between loads.
    }
  }, [selected])

  const select = useCallback((workspaceId: string) => {
    setChosen(workspaceId)
  }, [])

  const value = useMemo(
    () => ({
      memberships,
      selected,
      select,
      isLoading: context.isLoading,
    }),
    [memberships, selected, select, context.isLoading],
  )

  return <Context.Provider value={value}>{children}</Context.Provider>
}

// What a caller outside the provider sees. Not an error: "no workspace
// selected" is a real state inside the shell too (a caller who belongs to none),
// and every consumer already handles it by falling back to the deployment-wide
// view. A page rendered on its own therefore behaves like one whose switcher has
// nothing to offer, rather than crashing.
const NO_WORKSPACE: SelectedWorkspace = {
  memberships: [],
  selected: null,
  select: () => {},
  isLoading: false,
}

export function useSelectedWorkspace(): SelectedWorkspace {
  return useContext(Context) ?? NO_WORKSPACE
}
