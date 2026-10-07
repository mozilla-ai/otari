import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import type {
  BreakGlassRequest,
  ContentAccessList,
  SpanContent,
  TraceCount,
  TraceDetail,
  TraceList,
  TraceSeries,
  TraceSettings,
  TraceSettingsUpdate,
} from "@/client"
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

const ORGANIZATION_TRACES = "/organizations/me/traces"

// An operator reads every workspace on the deployment, anyone else their own
// organization, by the same rule as the usage pages (`useUsageScope`). The
// server derives the scope; this only picks which of the two routes to ask.
export function useTraceScope(): {
  base: string
  isReady: boolean
  /** The operator's deployment-wide view, where content opens only by break-glass. */
  isDeploymentWide: boolean
} {
  const operator = useDeploymentOperator()
  return {
    base: operator.isOperator ? "/traces" : ORGANIZATION_TRACES,
    isReady: operator.isSettled,
    isDeploymentWide: operator.isOperator,
  }
}

function contentPath(traceId: string, spanId: string): string {
  return `/${encodeURIComponent(traceId)}/spans/${encodeURIComponent(spanId)}/content`
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
export function useTrace(traceId: string) {
  const scope = useTraceScope()
  return useQuery({
    queryKey: [TRACES, "detail", scope.base, traceId],
    queryFn: () =>
      apiFetch<TraceDetail>(`${scope.base}/${encodeURIComponent(traceId)}`),
    enabled: scope.isReady && traceId !== "",
    staleTime: 5_000,
    refetchInterval: (query) =>
      query.state.data?.state === "active" ? 5_000 : false,
  })
}

// One span's captured content, read as a signed-in member: the session's own
// user, or an organization admin where the workspace allows it. Read only when
// the span has some and someone opened it, since every read is recorded on the
// server, and never from the operator's deployment-wide view, where content
// opens only through `useBreakGlassContent`.
export function useSpanContent(
  traceId: string,
  spanId: string,
  hasContent: boolean,
) {
  const scope = useTraceScope()
  return useQuery({
    queryKey: [TRACES, "content", traceId, spanId],
    queryFn: () =>
      apiFetch<SpanContent>(
        `${ORGANIZATION_TRACES}${contentPath(traceId, spanId)}`,
      ),
    enabled: scope.isReady && !scope.isDeploymentWide && hasContent,
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  })
}

// A platform operator's read of one span's content, for a stated reason the
// server records with the read. A mutation rather than a query: it is an act
// someone confirms, never something a render repeats.
export function useBreakGlassContent() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      traceId,
      spanId,
      reason,
    }: { traceId: string; spanId: string } & BreakGlassRequest) =>
      apiFetch<SpanContent>(
        `/traces${contentPath(traceId, spanId)}/break-glass`,
        { method: "POST", body: JSON.stringify({ reason }) },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: [TRACES, "content-access"],
      })
    },
  })
}

// A workspace's content capture. Only its admins may read it, so a refusal here
// means "not yours to change" rather than a failure, and is not retried.
export function useTraceSettings(workspaceId: string) {
  return useQuery({
    queryKey: [TRACES, "settings", workspaceId],
    queryFn: () =>
      apiFetch<TraceSettings>(`/workspaces/${workspaceId}/trace-settings`),
    enabled: workspaceId !== "",
    retry: false,
    staleTime: 30_000,
  })
}

// Changes one or both of a workspace's trace settings; a field left out keeps
// its value.
export function useUpdateTraceSettings(workspaceId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (update: TraceSettingsUpdate) =>
      apiFetch<TraceSettings>(`/workspaces/${workspaceId}/trace-settings`, {
        method: "PUT",
        body: JSON.stringify(update),
      }),
    onSuccess: (settings) => {
      queryClient.setQueryData([TRACES, "settings", workspaceId], settings)
    },
  })
}

// One page of the workspace's recorded content reads, newest first. Only its
// admins may list them. The endpoint does not count, so a full page is what
// says there may be another.
export function useContentAccessLog(
  workspaceId: string,
  page: number,
  pageSize: number,
) {
  return useQuery({
    queryKey: [TRACES, "content-access", workspaceId, page, pageSize],
    queryFn: () =>
      apiFetch<ContentAccessList>(
        `/workspaces/${workspaceId}/trace-settings/content-access?skip=${page * pageSize}&limit=${pageSize}`,
      ),
    enabled: workspaceId !== "",
    placeholderData: keepPreviousData,
    retry: false,
    staleTime: 10_000,
  })
}

export function usePurgeTraceContent(workspaceId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () =>
      apiFetch<{ removed: number }>(
        `/workspaces/${workspaceId}/trace-settings/purge-content`,
        { method: "POST" },
      ),
    onSuccess: () => {
      // Every open session's content flags are now stale.
      void queryClient.invalidateQueries({ queryKey: [TRACES, "detail"] })
      void queryClient.invalidateQueries({ queryKey: [TRACES, "content"] })
    },
  })
}
