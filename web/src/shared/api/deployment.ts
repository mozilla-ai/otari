import {
  keepPreviousData,
  type QueryClient,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import { useEffect, useRef, useState, useSyncExternalStore } from "react"
import type {
  BlockedEmailCreated,
  BlockedEmails,
  CreateBlockedEmailRequest,
  DashboardBuild,
  DeploymentAdminAccess,
  DeploymentUser,
  DeploymentUsers,
  PasswordLink,
  UpdateDeploymentUserRequest,
} from "@/client"
import {
  ApiError,
  apiFetch,
  DASHBOARD_BUILD_PATH,
  probeGateway,
  siteFetch,
} from "@/shared/api/client"
import {
  BUILD,
  BUILD_POLL_MS,
  DEPLOYMENT_ADMIN,
  GATEWAY_LIVENESS,
  GATEWAY_LIVENESS_RECHECK_MS,
  ORGANIZATION_MEMBERS,
} from "@/shared/api/queryKeys"

export function useDashboardBuild() {
  return useQuery({
    queryKey: [BUILD],
    // `siteFetch`, not `apiFetch`: the gateway serves this beside the dashboard
    // at its own root rather than under `API_ROOT`, so the helper that prepends
    // the root asks for a path nothing mounts.
    queryFn: () => siteFetch<DashboardBuild>(DASHBOARD_BUILD_PATH),
    refetchInterval: BUILD_POLL_MS,
    // A tab left open in the background is the one most likely to be stale, so
    // check again the moment someone comes back to it.
    refetchOnWindowFocus: true,
    staleTime: 0,
    // A failed check is not worth reporting: the tab keeps working, and the next
    // poll retries anyway.
    retry: false,
  })
}

// apiFetch normalizes an unreachable gateway to ApiError status 0 ("Network
// error: could not reach the gateway."). A 401/403 is a different failure: the
// backend answered, it just rejected the key, and that already bounces to
// sign-in, so it is not "can't connect".
function isUnreachable(error: unknown): boolean {
  return error instanceof ApiError && error.status === 0
}

// When a request last failed to reach the gateway, or 0 before any has. Kept
// from the cache's events rather than scanned off each query's state, so a page
// the operator has moved on from is not re-read on every notification. The
// probe's own failures are left out: they are the verdict, not a new suspicion.
// `error` fires once a query's retries are spent.
//
// The cache is an external store and is read as one. Subscribing in an effect
// and calling `setState` from the listener does not hold here: the cache emits
// synchronously when an observer is created, and an observer is created when a
// component calls `useQuery` during its render, so the listener runs inside
// another component's render pass. `useSyncExternalStore` is the primitive for
// this shape.
function trackUnreachable(queryClient: QueryClient) {
  let since = 0
  return {
    subscribe: (onStoreChange: () => void) =>
      queryClient.getQueryCache().subscribe((event) => {
        if (
          event.type !== "updated" ||
          event.action.type !== "error" ||
          event.query.queryKey[0] === GATEWAY_LIVENESS ||
          !isUnreachable(event.action.error)
        ) {
          return
        }
        since = Math.max(since, event.query.state.errorUpdatedAt)
        onStoreChange()
      }),
    getSnapshot: () => since,
  }
}

// Per attempt, rather than `apiFetch`'s 30s: the route does no I/O, so a
// gateway that has not answered it in a few seconds is not answering, and three
// full-length attempts would hold the alarm back for a minute and a half.
const GATEWAY_LIVENESS_TIMEOUT_MS = 5_000

/**
 * True once a request has failed to reach the gateway and a liveness probe,
 * retries included, has failed too. Clears when the probe next succeeds.
 *
 * One request failing at the network layer is not evidence the gateway is down:
 * a laptop waking from sleep, a Wi-Fi or VPN change, or an edge proxy dropping a
 * pooled connection each fail a single fetch against a healthy backend. So a
 * failed query only starts the probe, and the probe's answer is the verdict.
 * Rechecking with `GET /health/liveness` rather than re-running the failed
 * queries keeps a recheck cheap: the failed query may be a heavy usage read.
 *
 * The probe runs when the newest unreachable failure is newer than its own last
 * success, and keeps asking every few seconds for as long as it fails. It runs
 * whatever the browser believes about being online, because an offline browser
 * cannot reach the gateway either and the answer should say so. Any response
 * is an answer: only no response at all counts as down, and the probe does not
 * go through `apiFetch`, so a 401 from something in front of the gateway does
 * not sign the operator out.
 *
 * Each time the probe answers, the queries on screen that failed to connect
 * are run again, so a page's own error clears whether the gateway was down
 * for a minute or a request merely dropped. A query that fails that way again
 * while the gateway answers is left alone until it next succeeds, rather than
 * re-run on every probe; the gateway coming back from a confirmed outage runs
 * all of them regardless.
 *
 * "Down" is read from the timestamps, not `status`: a query that has never held
 * data goes back to `pending` on every refetch, so `isError` would drop the
 * banner for the length of each recheck while the gateway is still down.
 */
export function useGatewayUnreachable(): boolean {
  const queryClient = useQueryClient()
  // Built once per client: `useSyncExternalStore` identity-checks `subscribe`
  // and re-subscribes when it changes, and the tracker holds the state.
  const [tracker] = useState(() => trackUnreachable(queryClient))
  const unreachableSince = useSyncExternalStore(
    tracker.subscribe,
    tracker.getSnapshot,
  )
  const liveness = useQuery({
    queryKey: [GATEWAY_LIVENESS],
    queryFn: () =>
      probeGateway(
        "/health/liveness",
        AbortSignal.timeout(GATEWAY_LIVENESS_TIMEOUT_MS),
      ),
    enabled: (query) =>
      isDown(query.state) || unreachableSince > query.state.dataUpdatedAt,
    retry: 2,
    networkMode: "always",
    refetchInterval: (query) =>
      isDown(query.state) ? GATEWAY_LIVENESS_RECHECK_MS : false,
    refetchOnWindowFocus: true,
    staleTime: 0,
  })
  const down = isDown(liveness)
  const answeredAt = liveness.dataUpdatedAt
  // Answering: re-run what failed to reach it, so a page's own error clears
  // instead of waiting for focus or navigation. When each query was last re-run
  // this way, keyed by its hash.
  const wasDown = useRef(false)
  const reRunAt = useRef(new Map<string, number>())
  useEffect(() => {
    const recovered = wasDown.current && !down
    wasDown.current = down
    if (down || answeredAt === 0) {
      return
    }
    const due = queryClient
      .getQueryCache()
      .findAll({
        type: "active",
        predicate: (query) =>
          query.queryKey[0] !== GATEWAY_LIVENESS &&
          query.state.status === "error" &&
          isUnreachable(query.state.error),
      })
      .filter((query) => {
        const at = reRunAt.current.get(query.queryHash)
        return recovered || at === undefined || query.state.dataUpdatedAt > at
      })
    if (due.length === 0) {
      return
    }
    const now = Date.now()
    for (const query of due) {
      reRunAt.current.set(query.queryHash, now)
    }
    void queryClient.refetchQueries({
      type: "active",
      predicate: (query) => due.includes(query),
    })
  }, [answeredAt, down, queryClient])
  return down
}

function isDown(state: {
  errorUpdatedAt: number
  dataUpdatedAt: number
}): boolean {
  return state.errorUpdatedAt > state.dataUpdatedAt
}

// Every model the configured credentials can reach, per provider. Distinct from
// useModels: that is the catalog served to API callers (curated by
// model_discovery, aliases listed, targets withheld), while this is what an
// operator could pick from. A provider that failed is reported rather than
// dropped, so the picker can say why a list is empty.
//
// Live provider calls, cached gateway-side; kept fresh for the length of a
// session rather than refetched per open, since the set of models a key can
// reach does not move minute to minute.
//
// `enabled` is the `useToolSettings` composition: the read is
// deployment-operator-only, so a tenant-facing page declines to ask.

// ---------------------------------------------------------------------------
// Deployment administration
//
// The one management surface scoped to the deployment rather than to an
// organization. `useDeploymentAdminAccess` is its gate, which is why it answers
// 200 with a boolean where the rest of the prefix refuses a non-operator with
// 404: a gate cannot be built on a failed request, and hiding a destination
// grants nothing either way.
//
// Read by the deployment accounts, hosted providers and subagent router pages, to word their own
// refusal, and each holds that decision until this lands. Nothing else asks: the rail and the
// Overview index need the answer before their first paint, so they take
// `deployment_operator` off the organization context, which is the same
// server-side predicate on a read the shell already makes (otari#836).
// ---------------------------------------------------------------------------

// No `enabled` parameter: both callers sit behind routes whose surfaces
// (`admin`, `hosted_providers`, `agent_recommenders`) every gateway publishes. Not inferred from a
// 404 either: the ordinary retry policy applies, and a failure is a failure.
const accessQuery = {
  queryKey: [DEPLOYMENT_ADMIN, "access"],
  queryFn: () => apiFetch<DeploymentAdminAccess>("/admin/access"),
  staleTime: 60_000,
}

export function useDeploymentAdminAccess() {
  return useQuery({ ...accessQuery, select: (body) => body.granted })
}

/**
 * Whether the caller is the bootstrap operator, the one account that may hand
 * that role on. A gateway older than the field answers `false`.
 */
export function useIsBootstrapOperator() {
  return useQuery({
    ...accessQuery,
    select: (body) => body.is_bootstrap_operator ?? false,
  })
}

/**
 * One page of the deployment's accounts, narrowed by `search` on the server.
 *
 * Debounce the term before it gets here (`useDebounced`), or every keystroke is
 * a request.
 */
export function useDeploymentUsers(
  {
    page,
    pageSize,
    search,
  }: { page: number; pageSize: number; search: string },
  enabled = true,
) {
  return useQuery({
    queryKey: [DEPLOYMENT_ADMIN, "users", page, pageSize, search],
    queryFn: () => {
      const params = new URLSearchParams({
        skip: String(page * pageSize),
        limit: String(pageSize),
      })
      if (search) params.set("search", search)
      return apiFetch<DeploymentUsers>(`/admin/users?${params}`)
    },
    staleTime: 60_000,
    placeholderData: keepPreviousData,
    enabled,
  })
}

export function useUpdateDeploymentUser() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      id,
      body,
    }: {
      id: string
      body: UpdateDeploymentUserRequest
    }) =>
      apiFetch<DeploymentUser>(`/admin/users/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [DEPLOYMENT_ADMIN] })
      // Deactivating an account ends its sessions and changes what the
      // organization roster may offer it, so the tenancy reads go stale with it.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATION_MEMBERS] })
    },
  })
}

/**
 * Make another operator the bootstrap operator.
 *
 * Invalidates the access read as well as the list: the caller stops being the
 * bootstrap operator, so the control that did this goes away.
 */
export function useHandOverBootstrapOperator() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      id,
      currentPassword,
    }: {
      id: string
      currentPassword: string
    }) =>
      apiFetch<void>(
        `/admin/users/${encodeURIComponent(id)}/bootstrap-operator`,
        {
          method: "POST",
          body: JSON.stringify({ current_password: currentPassword }),
        },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [DEPLOYMENT_ADMIN] })
    },
  })
}

/**
 * Give an account a single-use link that sets its password.
 *
 * The gateway mails it where it can and otherwise returns it to pass on, so the
 * result says which. No cached read changes, so nothing is invalidated.
 */
export function useIssuePasswordLink() {
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<PasswordLink>(
        `/admin/users/${encodeURIComponent(id)}/password-link`,
        { method: "POST" },
      ),
  })
}

/** The blocklist reads' key prefix, which every blocklist write invalidates. */
const BLOCKED_EMAILS = [DEPLOYMENT_ADMIN, "blocked-emails"] as const

/** One page of the deployment's email blocklist, newest entry first. */
export function useBlockedEmails(
  { page, pageSize }: { page: number; pageSize: number },
  enabled = true,
) {
  return useQuery({
    queryKey: [...BLOCKED_EMAILS, page, pageSize],
    queryFn: () => {
      const params = new URLSearchParams({
        skip: String(page * pageSize),
        limit: String(pageSize),
      })
      return apiFetch<BlockedEmails>(`/admin/blocked-emails?${params}`)
    },
    staleTime: 60_000,
    placeholderData: keepPreviousData,
    enabled,
  })
}

export function useBlockEmail() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateBlockedEmailRequest) =>
      apiFetch<BlockedEmailCreated>("/admin/blocked-emails", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    // Blocking ends the matching accounts' sessions, which the accounts list
    // does not show, so only the blocklist itself goes stale.
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: BLOCKED_EMAILS })
    },
  })
}

export function useUnblockEmail() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/admin/blocked-emails/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: BLOCKED_EMAILS })
    },
  })
}
