import {
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import type {
  CreateWorkspaceBudgetDefaultRequest,
  CreateWorkspaceRequest,
  UpdateWorkspaceBudgetDefaultRequest,
  UpdateWorkspaceRequest,
  Workspace,
  WorkspaceBudgetDefault,
  WorkspaceMember,
  WorkspaceMemberRole,
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
