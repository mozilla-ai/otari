import { useQuery } from "@tanstack/react-query"

import type { SubagentSavings } from "@/client"
import { apiFetch } from "@/shared/api/client"
import { SUBAGENT_SAVINGS } from "@/shared/api/queryKeys"

// What the subagent router saved (`/organizations/me/subagent-savings`). The
// server derives the scope from the caller's membership: an owner or admin
// reads the organization, anyone else the workspaces they belong to. No
// previous answer stands in while a new one loads: a figure shown under a
// scope or a period it does not answer reads as that scope's own.

export interface SubagentSavingsFilters {
  start_date: string
  end_date?: string
  bucket: "hour" | "day"
  /** Narrows to one person's subagents. */
  user_id?: string
  /** Narrows to one workspace of the caller's scope. */
  workspace_id?: string
}

export function useSubagentSavings(
  filters: SubagentSavingsFilters,
  enabled = true,
) {
  return useQuery({
    queryKey: [SUBAGENT_SAVINGS, filters],
    queryFn: () => {
      const params = new URLSearchParams({
        start_date: filters.start_date,
        bucket: filters.bucket,
      })
      if (filters.end_date) params.set("end_date", filters.end_date)
      if (filters.user_id) params.set("user_id", filters.user_id)
      if (filters.workspace_id) {
        params.set("workspace_id", filters.workspace_id)
      }
      return apiFetch<SubagentSavings>(
        `/organizations/me/subagent-savings?${params.toString()}`,
      )
    },
    enabled,
    staleTime: 30_000,
  })
}
