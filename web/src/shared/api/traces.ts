import {
  keepPreviousData,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import type {
  TraceCount,
  TraceDetail,
  TraceList,
  TraceSeries,
  TraceSortKey,
  TraceSortOrder,
  TraceSummary,
} from "@/client"
import { apiFetch, retryUnlessNotFound } from "@/shared/api/client"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { TRACES, workspaceMeta } from "@/shared/api/queryKeys"

// What narrows the session list. Empty strings and arrays are "no filter", the
// same reading the URL state gives a cleared key. `workspaceId` is the shell's
// selected workspace, as the request log takes it; the server only lets it
// narrow the caller's scope, so a workspace outside it reads as empty.
export interface TraceFilters {
  workspaceId: string
  start: string
  // Optional, so a filter that names only a start reads to now.
  end?: string
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
  if (filters.end) params.set("end", filters.end)
  if (filters.q) params.set("q", filters.q)
  if (filters.failedOnly) params.set("has_error", "true")
  for (const harness of filters.harness) params.append("harness", harness)
  return params
}

// What the session list is ordered by. The server sorts, so the order holds
// across pages rather than within the one on screen.
export interface TraceSort {
  key: TraceSortKey
  direction: TraceSortOrder
}

export const DEFAULT_TRACE_SORT: TraceSort = {
  key: "last_activity",
  direction: "desc",
}

// One page of sessions, most recently active first unless `sort` says otherwise.
// `keepPreviousData` keeps the current page on screen while the next loads, so
// paging does not flash empty. Like the request log, the list is a snapshot an
// operator reads rather than a feed, so it refetches when asked (a mount, a
// filter, a page) and not on its own.
export function useTraces(
  filters: TraceFilters,
  page: number,
  pageSize: number,
  sort: TraceSort = DEFAULT_TRACE_SORT,
) {
  const scope = useTraceScope()
  return useQuery({
    queryKey: [TRACES, "list", scope.base, filters, page, pageSize, sort],
    meta: workspaceMeta(filters.workspaceId),
    queryFn: () => {
      const params = traceParams(filters)
      params.set("skip", String(page * pageSize))
      params.set("limit", String(pageSize))
      params.set("sort", sort.key)
      params.set("order", sort.direction)
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
    meta: workspaceMeta(filters.workspaceId),
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
    meta: workspaceMeta(filters.workspaceId),
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
    meta: workspaceMeta(workspaceId),
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

// The session a request from the usage log was traced into, by the
// `request_group_id` its rows carry. A request with no trace stored (served
// before traces were on, or expired since) answers 404, which is not retried.
export function useRequestTrace(requestId: string, enabled = true) {
  const scope = useTraceScope()
  return useQuery({
    queryKey: [TRACES, "request", scope.base, requestId],
    queryFn: () =>
      apiFetch<TraceSummary>(
        `${scope.base}/requests/${encodeURIComponent(requestId)}`,
      ),
    enabled: enabled && scope.isReady && requestId !== "",
    staleTime: 60_000,
    retry: retryUnlessNotFound(useQueryClient()),
  })
}
