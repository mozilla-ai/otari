import {
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import type {
  CreateWorkspaceBudgetDefaultRequest,
  CreateWorkspaceRequest,
  SetWorkspaceProviderKeyOverrideRequest,
  UpdateWorkspaceBudgetDefaultRequest,
  UpdateWorkspaceRequest,
  Workspace,
  WorkspaceBudgetDefault,
  WorkspaceMember,
  WorkspaceMemberRole,
  WorkspaceProviderKeyOverride,
} from "@/client"
import { apiFetch } from "@/shared/api/client"
import { fetchAllPaged } from "@/shared/api/paging"
import { ORGANIZATIONS, WORKSPACES } from "@/shared/api/queryKeys"

export function useWorkspaces(enabled = true) {
  return useQuery({
    queryKey: [WORKSPACES],
    queryFn: () => fetchAllPaged<Workspace>("/v1/workspaces"),
    staleTime: 60_000,
    enabled,
  })
}

// One workspace's roster. Nested under the workspaces key so deleting a
// workspace drops its roster with it.
export function useWorkspaceMembers(workspaceId: string | null) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "members"],
    queryFn: () =>
      fetchAllPaged<WorkspaceMember>(
        `/v1/workspaces/${encodeURIComponent(workspaceId as string)}/members`,
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

export function useCreateWorkspace() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateWorkspaceRequest) =>
      apiFetch<Workspace>("/v1/workspaces", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a workspace created
      // here would not be offered and a deleted one would stay selected.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

export function useUpdateWorkspace() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdateWorkspaceRequest }) =>
      apiFetch<Workspace>(`/v1/workspaces/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a workspace created
      // here would not be offered and a deleted one would stay selected.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

export function useDeleteWorkspace() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/v1/workspaces/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a workspace created
      // here would not be offered and a deleted one would stay selected.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

export function useAddWorkspaceMember() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      userId,
      role,
    }: {
      workspaceId: string
      userId: string
      role: WorkspaceMemberRole
    }) =>
      // The role travels as a query parameter, not a body: that is the wire
      // contract these endpoints were rehomed with.
      apiFetch<WorkspaceMember>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}?role=${encodeURIComponent(role)}`,
        { method: "POST" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a roster change that moves
      // the caller in or out of a workspace has to refresh it too.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

export function useUpdateWorkspaceMemberRole() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      userId,
      role,
    }: {
      workspaceId: string
      userId: string
      role: WorkspaceMemberRole
    }) =>
      apiFetch<WorkspaceMember>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}?role=${encodeURIComponent(role)}`,
        { method: "PATCH" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a roster change that moves
      // the caller in or out of a workspace has to refresh it too.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}

// A workspace's budget-default templates. Nested under the workspaces key so
// deleting a workspace drops them with it, same as `useWorkspaceMembers`.
export function useWorkspaceBudgetDefaults(workspaceId: string | null) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "budget-defaults"],
    queryFn: () =>
      fetchAllPaged<WorkspaceBudgetDefault>(
        `/v1/workspaces/${encodeURIComponent(workspaceId as string)}/member-budget-policies`,
      ),
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

/**
 * Every workspace's roster, as one list, each row paired with its workspace.
 *
 * Same fan-out as `useAllWorkspaceBudgetDefaults` and for the same reason: a
 * roster is only served per workspace, and a standalone deployment has few. It
 * is what lets the organization roster answer "which workspaces is this person
 * in", which is otherwise only answerable one workspace at a time.
 */
export function useAllWorkspaceMembers(workspaceIds: string[]) {
  return useQueries({
    queries: workspaceIds.map((workspaceId) => ({
      queryKey: [WORKSPACES, workspaceId, "members"],
      queryFn: () =>
        fetchAllPaged<WorkspaceMember>(
          `/v1/workspaces/${encodeURIComponent(workspaceId)}/members`,
        ),
      staleTime: 60_000,
    })),
    combine: (results) => ({
      data: results.flatMap((result, index) =>
        (result.data ?? []).map((row) => ({
          workspaceId: workspaceIds[index],
          member: row,
        })),
      ),
      isLoading: results.some((result) => result.isLoading),
      // The first failure, surfaced rather than swallowed: a rejected read
      // contributes nothing to `data`, so without this the caller cannot tell a
      // workspace with no rows from one whose read failed, and a lost membership
      // or a lost ceiling looks exactly like a deliberate absence.
      error: results.find((result) => result.error)?.error ?? null,
      isSuccess: results.every((result) => result.isSuccess),
    }),
  })
}

/**
 * Every workspace's budget defaults, as one list.
 *
 * A fan-out rather than one call: defaults are only served per workspace
 * (`/v1/workspaces/{id}/member-budget-policies`), and a standalone deployment
 * has few workspaces, so N small cached reads beat adding a route. Each shares
 * the cache entry `useWorkspaceBudgetDefaults` uses, so opening a workspace
 * afterwards costs nothing.
 *
 * This is what lets the budgets list say a budget is a workspace's default:
 * without it the page would know the budget and not the assignment.
 */
export function useAllWorkspaceBudgetDefaults(workspaceIds: string[]) {
  return useQueries({
    queries: workspaceIds.map((workspaceId) => ({
      queryKey: [WORKSPACES, workspaceId, "budget-defaults"],
      queryFn: () =>
        fetchAllPaged<WorkspaceBudgetDefault>(
          `/v1/workspaces/${encodeURIComponent(workspaceId)}/member-budget-policies`,
        ),
      staleTime: 60_000,
    })),
    combine: (results) => ({
      // Paired with its workspace on the way out: a default names a workspace by
      // id, and the caller wants the name.
      data: results.flatMap((result, index) =>
        (result.data ?? []).map((row) => ({
          workspaceId: workspaceIds[index],
          default: row,
        })),
      ),
      isLoading: results.some((result) => result.isLoading),
      // The first failure, surfaced rather than swallowed: a rejected read
      // contributes nothing to `data`, so without this the caller cannot tell a
      // workspace with no rows from one whose read failed, and a lost membership
      // or a lost ceiling looks exactly like a deliberate absence.
      error: results.find((result) => result.error)?.error ?? null,
      isSuccess: results.every((result) => result.isSuccess),
    }),
  })
}

export function useCreateWorkspaceBudgetDefault() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      body,
    }: {
      workspaceId: string
      body: CreateWorkspaceBudgetDefaultRequest
    }) =>
      apiFetch<WorkspaceBudgetDefault>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/member-budget-policies`,
        { method: "POST", body: JSON.stringify(body) },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "budget-defaults"],
      })
    },
  })
}

export function useUpdateWorkspaceBudgetDefault() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      defaultId,
      body,
    }: {
      workspaceId: string
      defaultId: string
      body: UpdateWorkspaceBudgetDefaultRequest
    }) =>
      apiFetch<WorkspaceBudgetDefault>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/member-budget-policies/${encodeURIComponent(defaultId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "budget-defaults"],
      })
    },
  })
}

export function useDeleteWorkspaceBudgetDefault() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      defaultId,
    }: {
      workspaceId: string
      defaultId: string
    }) =>
      apiFetch<void>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/member-budget-policies/${encodeURIComponent(defaultId)}`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId }) => {
      void queryClient.invalidateQueries({
        queryKey: [WORKSPACES, workspaceId, "budget-defaults"],
      })
    },
  })
}

/**
 * One workspace's view of its organization's provider keys.
 *
 * Every non-archived organization key, each carrying this workspace's departure
 * from it: `is_default`/`disabled` are the stored flags, `is_effective_*` the
 * resolution once the provider's other keys are taken into account. The response
 * names keys by id only, so the caller pairs it with `useOrgProviderKeys` for the
 * provider and the name.
 *
 * Not paged: the route serves the whole set in one body, because it is bounded by
 * the organization's key count rather than by anything a workspace accumulates.
 */
export function useWorkspaceProviderKeys(workspaceId: string | null) {
  return useQuery({
    queryKey: [WORKSPACES, workspaceId, "provider-keys"],
    queryFn: async () =>
      (
        await apiFetch<{ data: WorkspaceProviderKeyOverride[] }>(
          `/v1/workspaces/${encodeURIComponent(workspaceId as string)}/provider-keys`,
        )
      ).data,
    enabled: workspaceId !== null,
    staleTime: 60_000,
  })
}

/**
 * The model allow-list each of a workspace's keys carries, as one map.
 *
 * A fan-out for the reason the two above are: the allow-list is only served per
 * key (`GET /v1/workspaces/{id}/provider-keys/{key}/models`), and an organization
 * holds a handful of keys, so N small cached reads beat adding a route. An empty
 * list is the common answer and a meaningful one: no rows means every model the
 * key serves is allowed, not that none is.
 */
export function useWorkspaceProviderKeyModels(
  workspaceId: string | null,
  keyIds: string[],
) {
  return useQueries({
    queries: (workspaceId === null ? [] : keyIds).map((keyId) => ({
      queryKey: [WORKSPACES, workspaceId, "provider-keys", keyId, "models"],
      queryFn: async () =>
        (
          await apiFetch<{ models: string[] }>(
            `/v1/workspaces/${encodeURIComponent(workspaceId as string)}/provider-keys/${encodeURIComponent(keyId)}/models`,
          )
        ).models,
      staleTime: 60_000,
    })),
    combine: (results) => ({
      // Three states per key, not two. `models` is `undefined` rather than `[]`
      // until the read answers, because an empty list is the answer "every model
      // is allowed" and defaulting to one would report a narrowed workspace as
      // open. `failed` is what separates a read still in flight from one that
      // will never answer, so the caller can say which rather than showing a
      // refusal as a wait that does not end.
      data: new Map(
        results.map((result, index) => [
          keyIds[index],
          { models: result.data, failed: result.error !== null },
        ]),
      ),
      // The first failure rather than a swallowed one, as the fan-outs above do.
      error: results.find((result) => result.error)?.error ?? null,
    }),
  })
}

// Pinning a key clears whichever of the provider's other keys this workspace had
// pinned, and disabling one deletes its model allow-list server-side, so every
// write here re-reads the workspace's whole provider-key subtree rather than
// patching the row it acted on. The key is a prefix of each per-key allow-list
// key, so invalidating it takes those with it.
function invalidateWorkspaceProviderKeys(
  queryClient: ReturnType<typeof useQueryClient>,
  workspaceId: string,
): void {
  void queryClient.invalidateQueries({
    queryKey: [WORKSPACES, workspaceId, "provider-keys"],
  })
}

export function useSetWorkspaceProviderKeyOverride() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      keyId,
      body,
    }: {
      workspaceId: string
      keyId: string
      body: SetWorkspaceProviderKeyOverrideRequest
    }) =>
      apiFetch<WorkspaceProviderKeyOverride>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/provider-keys/${encodeURIComponent(keyId)}`,
        { method: "PATCH", body: JSON.stringify(body) },
      ),
    onSuccess: (_data, { workspaceId }) =>
      invalidateWorkspaceProviderKeys(queryClient, workspaceId),
  })
}

/** Drop the override entirely, so the workspace inherits the organization again. */
export function useResetWorkspaceProviderKeyOverride() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      keyId,
    }: {
      workspaceId: string
      keyId: string
    }) =>
      apiFetch<{ message: string }>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/provider-keys/${encodeURIComponent(keyId)}`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId }) =>
      invalidateWorkspaceProviderKeys(queryClient, workspaceId),
  })
}

export function useAddWorkspaceProviderKeyModel() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      keyId,
      model,
    }: {
      workspaceId: string
      keyId: string
      model: string
    }) =>
      apiFetch<{ message: string }>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/provider-keys/${encodeURIComponent(keyId)}/models`,
        { method: "POST", body: JSON.stringify({ model }) },
      ),
    onSuccess: (_data, { workspaceId }) =>
      invalidateWorkspaceProviderKeys(queryClient, workspaceId),
  })
}

export function useRemoveWorkspaceProviderKeyModel() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      keyId,
      model,
    }: {
      workspaceId: string
      keyId: string
      model: string
    }) =>
      apiFetch<{ message: string }>(
        // The model id is the last path segment and the route declares it
        // `:path`, so a provider that spells one with a slash still addresses
        // its own row: the escape survives the match and the gateway unquotes it.
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/provider-keys/${encodeURIComponent(keyId)}/models/${encodeURIComponent(model)}`,
        { method: "DELETE" },
      ),
    onSuccess: (_data, { workspaceId }) =>
      invalidateWorkspaceProviderKeys(queryClient, workspaceId),
  })
}

// The guardrails the caller's organization mandates over its workspaces. A
// small hand-edited list rather than a growing table, but paged through like
// the rest of the tenancy surface so a backend that ignored `skip` cannot spin
// this either.

export function useRemoveWorkspaceMember() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workspaceId,
      userId,
    }: {
      workspaceId: string
      userId: string
    }) =>
      apiFetch<void>(
        `/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [WORKSPACES] })
      // The switcher reads its list from `workspace_memberships` on the
      // organization context, not from this key, so a roster change that moves
      // the caller in or out of a workspace has to refresh it too.
      void queryClient.invalidateQueries({ queryKey: [ORGANIZATIONS] })
    },
  })
}
