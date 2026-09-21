import { useQuery } from "@tanstack/react-query"
import type { OverviewSummary } from "@/client"
import { apiFetch } from "@/shared/api/client"
import { OVERVIEW } from "@/shared/api/queryKeys"

/**
 * The overview's counts and budget strips, in one read.
 *
 * The page used to download keys, users, members, budgets and ceilings to
 * compute three integers and one worst-case row per strip (otari#1425). The
 * server does that scan now; what stays here is the wording.
 *
 * `budgets` and `ceilings` are null where the caller may not see them, which is
 * not the same as a strip with nothing in it: deployment budgets are the
 * operator's and ceilings are an organization owner's or admin's.
 */
export function useOverviewSummary(workspaceId?: string, enabled = true) {
  return useQuery({
    queryKey: [OVERVIEW, workspaceId ?? null],
    queryFn: () =>
      apiFetch<OverviewSummary>(
        workspaceId
          ? `/overview?workspace_id=${encodeURIComponent(workspaceId)}`
          : "/overview",
      ),
    staleTime: 60_000,
    enabled,
  })
}
