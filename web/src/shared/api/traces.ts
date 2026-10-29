import {
  keepPreviousData,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import type {
  TraceBreakGlassReadList,
  TraceContentPolicy,
  TraceContentShare,
  TraceContentShareCreate,
  TraceContentShareList,
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
import type { ChartBucket } from "@/shared/helpers/timeRange"

// What narrows the session list. An empty string is "no filter", the same
// reading the URL state gives a cleared key. `workspaceId` is the shell's
// selected workspace, as the request log takes it; the server only lets it
// narrow the caller's scope, so a workspace outside it reads as empty.
export interface TraceFilters {
  workspaceId: string
  start: string
  // Optional, so a filter that names only a start reads to now.
  end?: string
  q: string
  // Owning users, API keys and models, each matching any of its values. A model
  // matches a session with a call on it.
  userIds?: string[]
  apiKeyIds?: string[]
  models?: string[]
  // "true" keeps sessions with an unrecovered error, "false" those without.
  hasError?: "" | "true" | "false"
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

// What a deployment with traces off answers, and what is shown until the policy
// loads: no content, so nothing is offered before the deployment says it keeps any.
const NO_CONTENT_POLICY: TraceContentPolicy = {
  content_available: false,
  platform_break_glass: false,
  retention_days: 30,
  session_max_age_days: 90,
}

// What the deployment keeps of agent sessions, and who besides a session's owner
// can open it. Behind a sign-in, not on the public bootstrap; a deployment with
// traces off has no such route, which reads as keeping no content.
export function useTraceContentPolicy(): {
  policy: TraceContentPolicy
  isPending: boolean
} {
  const queryClient = useQueryClient()
  const query = useQuery({
    queryKey: [TRACES, "content-policy"],
    queryFn: () => apiFetch<TraceContentPolicy>("/trace-content/policy"),
    retry: retryUnlessNotFound(queryClient),
    staleTime: 5 * 60 * 1000,
  })
  return { policy: query.data ?? NO_CONTENT_POLICY, isPending: query.isPending }
}

export function traceParams(filters: TraceFilters): URLSearchParams {
  const params = new URLSearchParams()
  if (filters.workspaceId) params.set("workspace_id", filters.workspaceId)
  if (filters.start) params.set("start", filters.start)
  if (filters.end) params.set("end", filters.end)
  if (filters.q) params.set("q", filters.q)
  for (const id of filters.userIds ?? []) params.append("user_id", id)
  for (const id of filters.apiKeyIds ?? []) params.append("api_key_id", id)
  for (const model of filters.models ?? []) params.append("model", model)
  if (filters.hasError) params.set("has_error", filters.hasError)
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

export function useTraceSeries(filters: TraceFilters, bucket: ChartBucket) {
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

// Sharing one of your own sessions' content. Always on the organization route,
// whichever route reads the session: only its own user shares it, as a member.
const SHARE_BASE = "/organizations/me/traces"

function sharesPath(
  traceId: string,
  workspaceId: string,
  shareId = "",
  page: Record<string, string> = {},
): string {
  const params = new URLSearchParams(page)
  if (workspaceId) params.set("workspace_id", workspaceId)
  const query = params.toString() ? `?${params.toString()}` : ""
  const tail = shareId ? `/${encodeURIComponent(shareId)}` : ""
  return `${SHARE_BASE}/${encodeURIComponent(traceId)}/shares${tail}${query}`
}

const SHARES_PAGE = 20

// The shares the session has had, a page at a time, running ones first, with
// the newest reads made through each. Only the session's own user may ask, so
// it is asked only for them.
export function useTraceShares(
  traceId: string,
  workspaceId: string,
  enabled: boolean,
) {
  return useInfiniteQuery({
    queryKey: [TRACES, "shares", traceId, workspaceId],
    meta: workspaceMeta(workspaceId),
    initialPageParam: 0,
    queryFn: ({ pageParam }) =>
      apiFetch<TraceContentShareList>(
        sharesPath(traceId, workspaceId, "", {
          skip: String(pageParam),
          limit: String(SHARES_PAGE),
        }),
      ),
    getNextPageParam: (last, pages) =>
      last.has_more ? pages.length * SHARES_PAGE : undefined,
    enabled: enabled && traceId !== "",
    staleTime: 10_000,
  })
}

export function useShareTrace() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      traceId,
      workspaceId,
      body,
    }: {
      traceId: string
      workspaceId: string
      body: TraceContentShareCreate
    }) =>
      apiFetch<TraceContentShare>(sharesPath(traceId, workspaceId), {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [TRACES, "shares"] })
    },
  })
}

export function useEndTraceShare() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      traceId,
      workspaceId,
      shareId,
    }: {
      traceId: string
      workspaceId: string
      shareId: string
    }) =>
      apiFetch<void>(sharesPath(traceId, workspaceId, shareId), {
        method: "DELETE",
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [TRACES, "shares"] })
    },
  })
}

// Opening a session's content without its owner's share, as an owner of the
// organization. The opening is recorded with its reason before anything opens.
export function useBreakGlass() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      traceId,
      workspaceId,
      reason,
    }: {
      traceId: string
      workspaceId: string
      reason: string
    }) =>
      apiFetch<TraceDetail>(
        `${SHARE_BASE}/${encodeURIComponent(traceId)}/break-glass?${new URLSearchParams({ workspace_id: workspaceId }).toString()}`,
        { method: "POST", body: JSON.stringify({ reason }) },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [TRACES, "break-glass"] })
    },
  })
}

const BREAK_GLASS_PAGE = 50

// The organization's owners' break-glass openings, newest first, a page at a
// time, so every opening stays reachable however many follow it. Owners only.
export function useBreakGlassLog(enabled: boolean) {
  return useInfiniteQuery({
    queryKey: [TRACES, "break-glass"],
    initialPageParam: 0,
    queryFn: ({ pageParam }) =>
      apiFetch<TraceBreakGlassReadList>(
        `${SHARE_BASE}/audit/break-glass?${new URLSearchParams({
          skip: String(pageParam),
          limit: String(BREAK_GLASS_PAGE),
        }).toString()}`,
      ),
    getNextPageParam: (last, pages) =>
      last.has_more ? pages.length * BREAK_GLASS_PAGE : undefined,
    enabled,
    staleTime: 30_000,
  })
}
