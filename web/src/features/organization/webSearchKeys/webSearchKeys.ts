/**
 * The rules the organization's Web search providers page shows, kept apart
 * from the rendering so they can be tested without mounting it: which keys a
 * table lists, and what a workspace without a pin uses.
 */

import type { OrgWebSearchKey } from "@/client"
import type { WebToolKind } from "@/features/webSearch/webTools"

export const KIND_COPY: Readonly<
  Record<
    WebToolKind,
    {
      title: string
      serves: string
      empty: string
      useForEveryWorkspace: string
    }
  >
> = {
  search: {
    title: "Search providers",
    serves: "What the model's web searches and the search API run on.",
    empty: "No search keys. Workspaces search with this deployment's own tool.",
    useForEveryWorkspace:
      "With no workspace default, workspaces search with this deployment's own tool, unless their admins pin a key.",
  },
  fetch: {
    title: "Fetch providers",
    serves:
      "What the model's fetches run on, and the pages a search reads for its results.",
    empty: "No fetch keys. Workspaces fetch with this deployment's own tool.",
    useForEveryWorkspace:
      "With no workspace default, workspaces fetch with this deployment's own tool, unless their admins pin a key.",
  },
}

/** One kind's keys, archived ones only when they are shown. */
export function keysOfKind(
  keys: readonly OrgWebSearchKey[],
  kind: WebToolKind,
  showArchived: boolean,
): OrgWebSearchKey[] {
  return keys.filter(
    (key) => key.kind === kind && (showArchived || !key.archived_at),
  )
}

/** The workspace default: the key every workspace without a pin uses, if the organization set one. */
export function keyForEveryWorkspace(
  keys: readonly OrgWebSearchKey[],
  kind: WebToolKind,
): OrgWebSearchKey | undefined {
  return keys.find(
    (key) => key.kind === kind && key.is_org_default && !key.archived_at,
  )
}
