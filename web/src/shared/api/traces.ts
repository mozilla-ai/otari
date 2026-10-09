import { keepPreviousData, useQuery } from "@tanstack/react-query"
import type { TraceCount, TraceDetail, TraceList, TraceSeries } from "@/client"
import { apiFetch } from "@/shared/api/client"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { TRACES } from "@/shared/api/queryKeys"

// What narrows the session list. Empty strings and arrays are "no filter", the
// same reading the URL state gives a cleared key. `workspaceId` is the shell's
// selected workspace, as the request log takes it; the server only lets it
// narrow the caller's scope, so a workspace outside it reads as empty.
export interface TraceFilters {
  workspaceId: string
  start: string
  q: string
  failedOnly: boolean
  harness: string[]
}

// An operator reads every workspace on the deployment, anyone else their own
// organization, by the same rule as the usage pages (`useUsageScope`). The
// server derives the scope; this only picks which of the two routes to ask.
export function useTraceScope(): { base: string; isReady: boolean } {
  const operator = useDeploymentOperator()
  return {
    base: operator.isOperator ? "/traces" : "/organizations/me/traces",
    isReady: operator.isSettled,
  }
}

export function traceParams(filters: TraceFilters): URLSearchParams {
  const params = new URLSearchParams()
  if (filters.workspaceId) params.set("workspace_id", filters.workspaceId)
  if (filters.start) params.set("start", filters.start)
  if (filters.q) params.set("q", filters.q)
  if (filters.failedOnly) params.set("has_error", "true")
  for (const harness of filters.harness) params.append("harness", harness)
  return params
}

// One page of sessions, most recently active first. `keepPreviousData` keeps the
// current page on screen while the next loads, so paging does not flash empty.
// Like the request log, the list is a snapshot an operator reads rather than a
// feed, so it refetches when asked (a mount, a filter, a page) and not on its own.
export function useTraces(
  filters: TraceFilters,
  page: number,
  pageSize: number,
) {
  const scope = useTraceScope()
  return useQuery({
    queryKey: [TRACES, "list", scope.base, filters, page, pageSize],
    queryFn: () => {
      const params = traceParams(filters)
      params.set("skip", String(page * pageSize))
      params.set("limit", String(pageSize))
      return apiFetch<TraceList>(`${scope.base}?${params.toString()}`)
    },
    enabled: scope.isReady,
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  })
}

export function useTraceCount(filters: TraceFilters) {
  const scope = useTraceScope()
  return useQuery({
    queryKey: [TRACES, "count", scope.base, filters],
    queryFn: () =>
      apiFetch<TraceCount>(
        `${scope.base}/count?${traceParams(filters).toString()}`,
      ),
    enabled: scope.isReady,
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  })
}

export function useTraceSeries(filters: TraceFilters, bucket: "hour" | "day") {
  const scope = useTraceScope()
  return useQuery({
    queryKey: [TRACES, "series", scope.base, filters, bucket],
    queryFn: () => {
      const params = traceParams(filters)
      params.set("bucket", bucket)
      return apiFetch<TraceSeries>(`${scope.base}/series?${params.toString()}`)
    },
    enabled: scope.isReady,
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  })
}

// One session with its turns and spans. An active session keeps growing, so an
// open one is refetched every few seconds; a settled one is not.
// A trace is keyed by its workspace and its id, so the read names the workspace
// its list entry carried; an empty one is a link from before the URL kept it.
export function useTrace(traceId: string, workspaceId = "") {
  const scope = useTraceScope()
  const query = workspaceId
    ? `?${new URLSearchParams({ workspace_id: workspaceId }).toString()}`
    : ""
  return useQuery({
    queryKey: [TRACES, "detail", scope.base, traceId, workspaceId],
    queryFn: () =>
      apiFetch<TraceDetail>(
        `${scope.base}/${encodeURIComponent(traceId)}${query}`,
      ),
    enabled: scope.isReady && traceId !== "",
    staleTime: 5_000,
    refetchInterval: (query) =>
      query.state.data?.state === "active" ? 5_000 : false,
  })
}
